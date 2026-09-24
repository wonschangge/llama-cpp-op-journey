#!/usr/bin/env python3
"""覆盖域定义 —— 保真/覆盖两个门禁的共同基准。

视角：算子从模型定义到 CPU/GPU/NPU 等后端执行。

覆盖域 = 这条视角上真实承载"算子"的源文件集合，按 git 索引枚举（不用 os.walk，
避免把 build/ 产物算进来）。

核心层（CORE_PATTERNS）是必须 100% 覆盖的部分；
后端层（BACKEND_*）是 17 个后端家族，按家族逐课覆盖。
"""

import os
import subprocess

UPSTREAM = os.environ.get(
    "LLAMA_UPSTREAM", "/data/WORKSPACE/llama.cpp-project/llama.cpp"
)

# 上游仓库锚定版本（v0.5.0）
UPSTREAM_TAG = "v0.5.0"
UPSTREAM_COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"

# ---------------------------------------------------------------- 覆盖域

# 核心层：ggml 抽象 + llama 核心 + CPU 后端
CORE_PATTERNS = [
    "ggml/include/",       # ggml 公共头（张量/op/backend/各后端 API）
    "ggml/src/",           # ggml 顶层实现（ggml.c / backend / alloc / quants / gguf）
    "ggml/src/ggml-cpu/",  # CPU 后端
    "src/",                # llama.cpp 核心（模型定义 / 图构建 / 上下文）
    "include/",            # llama.h 公共 API
]

# 后端家族：目录名 -> 显示名
BACKEND_FAMILIES = {
    "ggml-cuda":     "CUDA (NVIDIA GPU)",
    "ggml-sycl":     "SYCL (Intel GPU)",
    "ggml-vulkan":   "Vulkan (跨厂商 GPU)",
    "ggml-metal":    "Metal (Apple GPU)",
    "ggml-hip":      "HIP (AMD GPU)",
    "ggml-musa":     "MUSA (Moore Threads GPU)",
    "ggml-opencl":   "OpenCL (移动 GPU)",
    "ggml-webgpu":   "WebGPU (浏览器 GPU)",
    "ggml-cann":     "CANN (Ascend NPU)",
    "ggml-hexagon":  "Hexagon (Qualcomm NPU/DSP)",
    "ggml-openvino": "OpenVINO (Intel NPU/GPU)",
    "ggml-et":       "ExecuTorch (边缘/移动端)",
    "ggml-virtgpu":  "VirtGPU (虚拟化 GPU)",
    "ggml-zdnn":     "zDNN (IBM Z 加速器)",
    "ggml-zendnn":   "ZenDNN (AMD EPYC CPU)",
    "ggml-blas":     "BLAS (厂商数学库)",
    "ggml-rpc":      "RPC (远程后端)",
}

# 计入覆盖域的后缀（按本视角：算子的宿主语言 + GPU/NPU 内核语言）
SOURCE_SUFFIXES = (
    ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cu", ".cuh",          # CUDA / HIP / MUSA
    ".m", ".mm",            # Metal (ObjC++)
    ".metal",               # Metal 内核
    ".comp",                # Vulkan GLSL 计算着色器
)


def git_ls_files(upstream=UPSTREAM):
    """用 git 索引枚举上游受控文件（相对路径）。"""
    out = subprocess.run(
        ["git", "-C", upstream, "ls-files"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def classify(path):
    """返回 (tier, family)。tier ∈ {'core','backend'}；family 为后端目录名或 None。"""
    for pat in CORE_PATTERNS:
        if path.startswith(pat):
            # ggml/src/ggml-<family>/ 属于后端层，不是 core
            if path.startswith("ggml/src/ggml-") and not path.startswith(
                "ggml/src/ggml-cpu/"
            ):
                rest = path[len("ggml/src/"):]
                head = rest.split("/", 1)[0]
                if "/" in rest and head in BACKEND_FAMILIES:
                    return "backend", head
            return "core", None
    for fam in BACKEND_FAMILIES:
        if path.startswith(f"ggml/src/{fam}/"):
            return "backend", fam
    return None, None


def universe(upstream=UPSTREAM):
    """覆盖域全集（已过滤后缀）。返回排序后的相对路径列表。"""
    files = []
    for path in git_ls_files(upstream):
        if not path.endswith(SOURCE_SUFFIXES):
            continue
        tier, _fam = classify(path)
        if tier is None:
            continue
        files.append(path)
    return sorted(files)


def universe_with_tier(upstream=UPSTREAM):
    """返回 [(path, tier, family)]。"""
    out = []
    for path in universe(upstream):
        tier, fam = classify(path)
        out.append((path, tier, fam))
    return out


if __name__ == "__main__":
    items = universe_with_tier()
    core = [p for p, t, _ in items if t == "core"]
    back = [p for p, t, _ in items if t == "backend"]
    print(f"上游根目录 : {UPSTREAM}")
    print(f"覆盖域总数 : {len(items)}")
    print(f"  核心层   : {len(core)}")
    print(f"  后端层   : {len(back)}")
    print()
    for fam in BACKEND_FAMILIES:
        n = sum(1 for _, t, f in items if f == fam)
        print(f"  {fam:<16} {n:5d}  {BACKEND_FAMILIES[fam]}")
