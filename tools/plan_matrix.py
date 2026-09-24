#!/usr/bin/env python3
"""生成覆盖矩阵 plan/COVERAGE.md，并断言覆盖域中无未指派文件。

用法:
    python3 tools/plan_matrix.py            # 打印矩阵摘要 + 断言
    python3 tools/plan_matrix.py --md       # 写出 plan/COVERAGE.md
    python3 tools/plan_matrix.py --files    # 打印每个文件的归属课
"""

import os
import re
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repo_universe as ru          # noqa: E402
import plan_model as pm             # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 层目录名（用于生成课件路径）
LAYER_DIR = {
    "L1": "L1-operator-representation",
    "L2": "L2-model-to-graph",
    "L3": "L3-backend-registry",
    "L4": "L4-memory-and-scheduling",
    "L5": "L5-cpu-backend",
    "L6": "L6-gpu-backend",
    "L7": "L7-npu-backend",
    "L8": "L8-end-to-end",
}

# 每课的 URL 片段（ASCII，便于 GitHub Pages 路径可读）
SLUGS = {
    "L1-01": "tensor-data-plane",       "L1-02": "op-enum-and-nargs",
    "L1-03": "graph-and-toposort",      "L1-04": "quant-block-layout",
    "L1-05": "context-threads-opt",     "L1-06": "gguf-container",
    "L2-01": "llama-h-api",             "L2-02": "arch-table-hparams",
    "L2-03": "weight-loading-mmap",     "L2-04": "kv-cache-and-memory",
    "L2-05": "batch-and-model-build",   "L2-06": "graph-skeleton",
    "L2-07": "context-and-decode",      "L2-08": "sampler-and-vocab",
    "L2-09": "quant-export-adapter",    "L2-10": "models-multimodal",
    "L2-11": "models-ssm-linear",       "L2-12": "models-moe-a",
    "L2-13": "models-moe-b",            "L2-14": "models-dense-a",
    "L2-15": "models-dense-b",
    "L3-01": "backend-contract",        "L3-02": "registry-and-devices",
    "L3-03": "dynamic-loading",         "L3-04": "backend-features",
    "L4-01": "ggml-alloc",              "L4-02": "scheduler-split",
    "L4-03": "buffers-and-types",       "L4-04": "graph-compute-entry",
    "L5-01": "cpu-backend-skeleton",    "L5-02": "unary-binary-ops",
    "L5-03": "simd-infrastructure",     "L5-04": "quants-and-repack",
    "L5-05": "multiarch-and-vendor",
    "L6-01": "cuda-skeleton",           "L6-02": "cuda-quant-matmul",
    "L6-03": "cuda-flash-attention",    "L6-04": "cuda-other-ops",
    "L6-05": "sycl-backend",            "L6-06": "vulkan-host",
    "L6-07": "vulkan-shaders",          "L6-08": "metal-host-fusion",
    "L6-09": "metal-kernels",
    "L7-01": "cann-ascend-npu",         "L7-02": "hexagon-qualcomm",
    "L7-03": "openvino-intel-npu",      "L7-04": "executorch-edge",
    "L7-05": "virtgpu-virtualized",     "L7-06": "small-backends",
    "L8-01": "journey-of-mul-mat",      "L8-02": "offload-decision",
    "L8-03": "writing-a-new-backend",
}


def lesson_dir(les):
    """课件目录（相对仓库根）。"""
    return f"{LAYER_DIR[les['layer']]}/{les['id']}-{SLUGS[les['id']]}"

# src/models 家族判定用的图原语标记（静态扫描，可复现）
MODEL_MARKERS = [
    ("multimodal", re.compile(r"clip_|vision|mmproj|image_")),
    ("ssm",        re.compile(r"build_ssm|ssm_conv|ssm_scan|gated_delta|build_rwkv|wkv")),
    ("moe",        re.compile(r"build_moe_ffn|ffn_gate_exps|ffn_gate_inp")),
]


def _read(rel):
    with open(os.path.join(ru.UPSTREAM, rel), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def classify_models(paths):
    """把 src/models/*.cpp 按图原语分类。返回 {group: [paths]}。

    优先级：multimodal > ssm > moe > dense。models.h 归 dense（它只是声明）。
    """
    out = collections.defaultdict(list)
    for p in paths:
        if not p.endswith(".cpp"):
            out["dense"].append(p)
            continue
        text = _read(p)
        for name, rx in MODEL_MARKERS:
            if rx.search(text):
                out[name].append(p)
                break
        else:
            out["dense"].append(p)
    for k in out:
        out[k].sort()
    return out


def _halve(lst):
    mid = (len(lst) + 1) // 2
    return lst[:mid], lst[mid:]


def resolve():
    """返回 (assign, unassigned)。assign: lesson_id -> [paths]（已排序去重）。"""
    uni = set(ru.universe())
    model_paths = sorted(p for p in uni if p.startswith("src/models/"))
    fams = classify_models(model_paths)
    moe_a, moe_b = _halve(fams["moe"])
    dense_a, dense_b = _halve(fams["dense"])
    model_split = {
        "L2-10": fams["multimodal"] + ["src/models/models.h"],
        "L2-11": fams["ssm"],
        "L2-12": moe_a,
        "L2-13": moe_b,
        "L2-14": dense_a,
        "L2-15": dense_b,
    }

    assign = collections.OrderedDict()
    for les in pm.LESSONS:
        lid = les["id"]
        if les.get("group") == "models":
            picked = [p for p in model_split[lid] if p in uni]
        else:
            picked = []
            for pat in les["paths"]:
                if pat.endswith("/"):
                    picked += [p for p in uni if p.startswith(pat)]
                else:
                    picked += [p for p in uni if p == pat or p.startswith(pat)]
        assign[lid] = sorted(set(picked))

    declared = set()
    for v in assign.values():
        declared |= set(v)
    unassigned = sorted(uni - declared)
    return assign, unassigned


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""usage: plan_matrix.py [--md] [--plan] [--files]

生成覆盖矩阵与分层课程计划，并断言覆盖域中无未指派文件。

options:
  --md       写出 plan/COVERAGE.md（文件到课的映射矩阵）
  --plan     写出 plan/TODOLIST.md（勾选状态由磁盘文件齐备度自动生成）
  --files    逐行打印 课号<TAB>文件
  -h, --help 显示本帮助
""")
        return 0
    assign, unassigned = resolve()
    uni = ru.universe()
    total = len(uni)
    covered = total - len(unassigned)

    if "--files" in sys.argv:
        for lid, paths in assign.items():
            for p in paths:
                print(f"{lid}\t{p}")
        return 0 if not unassigned else 1

    if "--plan" in sys.argv:
        missing = [l["id"] for l in pm.LESSONS if l["id"] not in SLUGS]
        if missing:
            print(f"缺少 slug: {missing}", file=sys.stderr)
            return 2
        os.makedirs(os.path.join(HERE, "plan"), exist_ok=True)
        path = os.path.join(HERE, "plan", "TODOLIST.md")
        done = 0
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# 分层课程计划 —— llama.cpp 算子之旅\n\n")
            fh.write("> **视角**：一个算子如何从模型定义出发，"
                     "落到 CPU / GPU / NPU 等后端上真正执行。\n")
            fh.write("> **上游**：`ggml-org/llama.cpp` @ "
                     f"`{ru.UPSTREAM_TAG}` (`{ru.UPSTREAM_COMMIT[:12]}`)\n\n")
            fh.write(f"**总课数 {len(pm.LESSONS)}　覆盖域 {total} 个源文件**"
                     "（核心层 330 + 后端层 960）\n\n")
            fh.write("勾选状态由 `tools/plan_matrix.py --plan` 依据磁盘上四个文件是否齐备"
                     "自动生成，不靠肉眼。\n\n")
            fh.write("---\n\n")
            for layer, name, desc in pm.LAYERS:
                les = [l for l in pm.LESSONS if l["layer"] == layer]
                n = sum(len(assign[l["id"]]) for l in les)
                fh.write(f"## {layer} · {name}\n\n")
                fh.write(f"> {desc}\n>\n> {len(les)} 课，覆盖 {n} 个源文件。\n\n")
                for l in les:
                    d = lesson_dir(l)
                    ok = all(os.path.isfile(os.path.join(HERE, d, f))
                             for f in ("index.html", "lesson.js", "source.md", "README.md"))
                    done += ok
                    fh.write(f"- [{'x' if ok else ' '}] **`{l['id']}`** "
                             f"{l['title']}　`{l['prio']}`　"
                             f"{len(assign[l['id']])} 文件\n")
                    fh.write(f"  - 讲解要点：{l['ideas']}\n")
                    fh.write(f"  - 验收点：{l['accept']}\n")
                    fh.write(f"  - 目录：`{d}/`\n")
                fh.write("\n")
        print(f"已写出 {path}（{done}/{len(pm.LESSONS)} 课已齐备）")
        return 0 if not unassigned else 1

    if "--md" in sys.argv:
        os.makedirs(os.path.join(HERE, "plan"), exist_ok=True)
        path = os.path.join(HERE, "plan", "COVERAGE.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# 覆盖矩阵\n\n")
            fh.write(f"上游：`{ru.UPSTREAM}` @ `{ru.UPSTREAM_TAG}` "
                     f"(`{ru.UPSTREAM_COMMIT[:12]}`)\n\n")
            fh.write(f"覆盖域源文件总数：**{total}**　"
                     f"已指派：**{covered}**　未指派：**{len(unassigned)}**\n\n")
            for layer, name, desc in pm.LAYERS:
                les = [l for l in pm.LESSONS if l["layer"] == layer]
                n = sum(len(assign[l["id"]]) for l in les)
                fh.write(f"## {layer} {name}（{len(les)} 课 / {n} 文件）\n\n")
                fh.write(f"> {desc}\n\n")
                fh.write("| 课 | 优先级 | 主题 | 文件数 |\n|---|---|---|---|\n")
                for l in les:
                    fh.write(f"| `{l['id']}` | {l['prio']} | {l['title']} | "
                             f"{len(assign[l['id']])} |\n")
                fh.write("\n")
            if unassigned:
                fh.write("## 未指派文件\n\n```\n")
                fh.write("\n".join(unassigned))
                fh.write("\n```\n")
        print(f"已写出 {path}")
        return 0 if not unassigned else 1

    print(f"覆盖域源文件总数 : {total}")
    print(f"已指派           : {covered}  ({covered/total*100:.1f}%)")
    print(f"未指派           : {len(unassigned)}")
    print(f"课数             : {len(pm.LESSONS)}")
    print()
    for layer, name, _ in pm.LAYERS:
        les = [l for l in pm.LESSONS if l["layer"] == layer]
        n = sum(len(assign[l["id"]]) for l in les)
        flag = "  <-- 有文件数为 0 的课" if any(not assign[l["id"]] for l in les) else ""
        print(f"  {layer} {name:<16} {len(les):2d} 课 / {n:5d} 文件{flag}")
    if unassigned:
        print("\n未指派（前 40）:")
        for p in unassigned[:40]:
            print("   ", p)
        return 1
    print("\n✓ 覆盖域已全部指派到课")
    return 0


if __name__ == "__main__":
    sys.exit(main())
