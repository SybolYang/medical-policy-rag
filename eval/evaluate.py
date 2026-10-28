"""【所属文件夹：eval/ = 评估（第7步）。eval 和 src 平级，专门放「质检」相关代码和题库】
【本文件角色：第7步 评估 —— 跑评测集，算「检索层」指标，检验 RAG 检索得准不准】

为什么放在 eval/ 而不是 src/：评估是「质检台」，不是生产流水线的一环，单独隔开职责更清晰。

检索层（脚本自动算，客观）：
  命中率 hit@k = 检索回来的 top-k 里，至少包含一个标准出处的题数 / 计分题数
  召回率      = 每题（命中的标准出处数 / 该题标准出处总数）的平均
  Top1 命中   = 「第一条就对」的比例（用户体感最关键的指标）
生成层（忠实度 / 引用准确率 / 未覆盖题是否老实拒答）需要大模型或人工判断，
脚本只把这些题标出来提示人工复核，不自动打分、不编造数字（守住「先真跑出来再写简历」的红线）。

用法：.venv/Scripts/python -m eval.evaluate [--top-k 4]
"""

from __future__ import annotations    # 固定咒语：类型注解延迟求值

import sys                            # 拿命令行参数、改控制台编码
import json                           # 读评测集 JSON

try:                                  # 又是 GBK 坑：强制控制台 UTF-8，否则打印 ✓ ✗ 这类符号会崩
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import config                # 注意：这里用 src.config 而不是 .config，因为 eval 和 src 是两个平级的包
from src.embedder import embed_query  # 复用 src 的向量化（把评测题的问题转向量）
from src.vectorstore import VectorStore   # 复用 src 的检索


def run_eval(top_k: int = config.TOP_K) -> None:
    """跑整个评测集，打印检索层指标报告。"""
    if not VectorStore.exists():      # 先确认建过库（没库没法检索）
        print("还没建库，先跑：python -m src.main build")
        return

    with open(config.EVAL_DATASET, encoding="utf-8") as f:   # 打开评测集题库 dataset.json
        dataset = json.load(f)                               # 读成 Python 列表（每题一个字典）

    store = VectorStore.load()        # 把建好的索引读回内存
    scored = []                       # 收集「有标准出处的题」（能算检索分的）
    uncovered = []                    # 收集「未覆盖题」（golden_sources 为空，考模型会不会拒答，不计检索分）

    for item in dataset:              # 逐题评测
        q = item["question"]                             # 取问题
        goldens = set(item.get("golden_sources", []))    # 取标准出处，转成集合（方便算交集）；.get 取不到就用空列表
        hits = store.search(embed_query(q), k=top_k)     # 把问题向量化后检索 top-k
        retrieved = [c.filename for _s, c in hits]       # 取出检索到的文件名列表（_s=不要相似度分）
        top1 = retrieved[0] if retrieved else None       # 第一条命中的文件名（没有就 None）

        if not goldens:               # 标准出处为空 = 这是道「未覆盖」题
            uncovered.append((item, retrieved))   # 单独收起来，不进计分
            continue

        hit_set = goldens & set(retrieved)   # & = 集合交集：标准出处里「确实被检索到」的那些
        scored.append(                        # 记录这题的评分明细
            {
                "id": item["id"],
                "category": item.get("category", "事实"),   # 题目类别（事实/跨文档），没标就当「事实」
                "question": q,
                "hit": len(hit_set) > 0,          # 命中：交集非空 = 至少搜到一个标准出处
                "top1_hit": top1 in goldens,      # Top1命中：第一条就是标准出处之一
                "recall": len(hit_set) / len(goldens),   # 召回率：命中的标准出处 / 全部标准出处
                "retrieved_top1": top1,           # 记下实际第一条搜到啥（没命中时用来对比诊断）
                "goldens": list(goldens),         # 记下标准答案出处
            }
        )

    n = len(scored)                   # 计分题总数
    if n == 0:                        # 一道计分题都没有（全是未覆盖题）
        print("评测集里没有可计分的题（golden_sources 都为空）。")
        return

    # 三个总指标：把每题的布尔/数值汇总求平均
    hit_rate = sum(1 for s in scored if s["hit"]) / n          # 命中率 = 命中的题数 / 总题数（sum(1 for...if) = 计数满足条件的）
    top1_rate = sum(1 for s in scored if s["top1_hit"]) / n    # Top1 命中率
    avg_recall = sum(s["recall"] for s in scored) / n          # 平均召回率

    print(f"=== 检索层评估（top_k={top_k}, 计分题 {n} 道）===")
    print(f"命中率 hit@{top_k} : {hit_rate:.1%}  ({sum(1 for s in scored if s['hit'])}/{n})")   # :.1% = 转成百分数保留1位小数
    print(f"Top1 命中       : {top1_rate:.1%}  ({sum(1 for s in scored if s['top1_hit'])}/{n})")
    print(f"平均召回率      : {avg_recall:.1%}")

    print("\n--- 分类别 ---")          # 按 category 分组看命中率（比如「跨文档」类是不是明显更低）
    cats = sorted(set(s["category"] for s in scored))   # set 去重出所有类别，sorted 排序
    for cat in cats:
        rows = [s for s in scored if s["category"] == cat]   # 筛出该类别的所有题
        ch = sum(1 for r in rows if r["hit"]) / len(rows)    # 该类别命中率
        print(f"  {cat}: 命中率 {ch:.1%} ({len(rows)}题)")

    print("\n--- 逐题 ---")             # 一题题看结果，方便定位哪题没搜准
    for s in scored:
        flag = "✓" if s["top1_hit"] else ("○" if s["hit"] else "✗")   # ✓=第一条就对，○=命中但不在第一条，✗=没命中
        print(f"  {flag} #{s['id']} [{s['category']}] {s['question'][:26]}")
        if not s["top1_hit"]:           # 不是第一条就对的，打印诊断信息：实际搜到啥 vs 应该搜到啥
            print(f"       top1检索到: {s['retrieved_top1']}")
            print(f"       应命中    : {s['goldens'][0][:40]}")

    if uncovered:                       # 如果有未覆盖题，单独提示（不计检索分）
        print("\n--- 未覆盖题（需人工/LLM 复核是否老实拒答，不计入检索分）---")
        for item, retrieved in uncovered:
            print(f"  #{item['id']} {item['question'][:30]}")
            print(f"       检索仍返回了 {len(retrieved)} 段 → 生成层应触发「未覆盖」拒答")

    print("\n注：生成层忠实度/引用准确率需配 DEEPSEEK_API_KEY 后人工或 LLM-as-judge 评。")


if __name__ == "__main__":              # 直接运行本文件时执行
    k = config.TOP_K                    # 默认 top_k
    if "--top-k" in sys.argv:           # 如果命令行给了 --top-k 参数
        k = int(sys.argv[sys.argv.index("--top-k") + 1])   # index 找到 --top-k 的位置，+1 取它后面那个值转成整数
    run_eval(k)
