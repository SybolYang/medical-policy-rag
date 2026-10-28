"""【所属文件夹：src/ = 源代码】
【本文件角色：总指挥（CLI 入口）—— 把前面所有模块串成三个可运行的命令】

它本身不做 RAG 的具体计算，只负责「按顺序调用别的模块」和「命令行分发」。
等于前端的 main.js / 路由入口：把各功能模块接起来，对外暴露几个命令。

用法（在项目根目录）：
  .venv/Scripts/python -m src.main build            # 离线：第1-4步 加载→切分→向量化→入库
  .venv/Scripts/python -m src.main ask "你的问题"    # 在线：第5-7步 检索→(重排)→生成→带出处
  .venv/Scripts/python -m src.main ask "问题" --rerank   # 启用重排
  .venv/Scripts/python -m src.main demo            # 跑一个内置示例问题
"""

from __future__ import annotations    # 固定咒语：类型注解延迟求值

import sys                            # 系统库：拿命令行参数 argv、改控制台输出编码

# 【又是 GBK 坑】Windows 控制台默认 GBK 编码，打印中文/✓ 这种符号会崩。
# 开头强制把标准输出/错误改成 UTF-8。try/except pass = 万一这方法在老 Python 不存在就忽略，不让它崩。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # errors="replace"：遇到编不了的字符用 ? 代替而不是抛异常
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 下面把各模块要用的函数/类导入进来（等于 JS 顶部的 import）
from . import config                             # 配置
from .loader import load_documents               # 第1步 加载
from .chunker import build_chunks                # 第2步 切分
from .embedder import embed_texts, embed_query   # 第3步 向量化（建库用 embed_texts，问答用 embed_query）
from .vectorstore import VectorStore             # 第4步入库 + 第5步检索
from .cache import Cache                         # 技巧③ 缓存
from .generator import call_llm, build_prompt    # 第6步 生成
from . import rerank as rerank_mod               # 可选重排（as = 起别名，避免和局部变量重名）


def cmd_build() -> None:
    """命令 build：离线建库，按顺序跑第1-4步，每步打印进度让你看见数据在流。"""
    print("=== 离线建库（七步链路 1-4）===")
    print("[1/4] 加载 data/raw ...")
    docs = load_documents()                     # 第1步：读 55 篇政策 → Document 列表
    print(f"      {len(docs)} 篇文档")

    print("[2/4] 切分（技巧①：按句子+重叠）...")
    chunks = build_chunks(docs)                 # 第2步：Document → Chunk 列表（560 段）
    print(f"      {len(chunks)} 个切片")

    print("[3/4] 向量化（bge-small-zh，首次会下载模型）...")
    vectors = embed_texts([c.text for c in chunks])   # 第3步：把每段文字转向量；[c.text for c in chunks]=取出所有段的文字
    print(f"      向量矩阵 {vectors.shape}")          # shape=(560,512)

    print("[4/4] 入库（Faiss IndexFlatIP）...")
    store = VectorStore.build(chunks, vectors)  # 第4步：造 Faiss 索引并灌入向量
    store.save()                                # 落盘到 data/processed/
    print(f"      已存 {config.INDEX_PATH.name} + {config.CHUNKS_PATH.name}")
    print("建库完成。现在可以用 ask 提问了。")


def _retrieve(question: str, use_rerank: bool):
    """【🔴第5步检索】把问题向量化 → 搜 top-k →（可选）重排，返回检索结果。前缀 _ = 内部函数。"""
    store = VectorStore.load()                  # 从硬盘把建好的索引读回内存
    qv = embed_query(question)                  # 把「问题」也向量化（必须用同一个模型，否则向量空间对不上）
    hits = store.search(qv, k=config.TOP_K)     # 搜出最相近的 TOP_K 个切片
    if use_rerank:                              # 如果用户加了 --rerank
        hits = rerank_mod.rerank(question, hits)   # 再重排一次序
    return hits


def cmd_ask(question: str, use_rerank: bool = False) -> None:
    """命令 ask：在线问答，跑第5-7步。这是用户真正提问时走的路径。"""
    if not VectorStore.exists():                # 先检查建过库没有
        print("还没建库，先跑：python -m src.main build")
        return                                  # 没建库直接返回，不往下走

    print(f"=== 在线问答（七步链路 5-7）===\n问题：{question}\n")

    cache = Cache()                             # 技巧③：先查缓存
    cached = cache.get(question)
    if cached:                                  # 命中缓存 → 直接返回上次答案，不检索不生成（省钱省时间）
        print("[缓存命中·技巧③] 直接返回上次答案：\n")
        print(cached["answer"])
        _print_sources(cached["sources"])
        return

    print(f"[5/7] 检索 top-{config.TOP_K}" + ("（含重排）" if use_rerank else "") + " ...")
    hits = _retrieve(question, use_rerank)      # 第5步：检索
    for i, (score, c) in enumerate(hits, 1):    # 打印每条命中：序号、相似度、标题
        print(f"      {i}. 相似度{score:.3f} 《{c.title[:24]}》")   # :.3f=3位小数；[:24]=标题前24字

    print("[6/7] 生成（技巧②：引用约束 prompt）...")
    answer = call_llm(question, hits)           # 第6步：调大模型（无 key 返回 None）
    sources = [{"title": c.title, "url": c.source_url, "date": c.date} for _s, c in hits]   # 收集出处列表（_s=不要分数）

    if answer is None:                          # 没配 API key → dry-run：只打印拼好的约束 prompt，让你看清技巧②长啥样
        print("      未配置 DEEPSEEK_API_KEY → dry-run，只展示拼好的约束 prompt：\n")
        print("----- SYSTEM -----")
        from .generator import SYSTEM_PROMPT    # 局部导入那段系统提示词
        print(SYSTEM_PROMPT)
        print("\n----- USER（拼了检索到的原文段）-----")
        print(build_prompt(question, hits))     # 打印拼好的 user prompt
        answer = "[dry-run] 未调用大模型。配置 .env 里的 DEEPSEEK_API_KEY 后会生成带出处的答案。"
    else:                                       # 配了 key、真拿到了答案
        print("\n----- 答案 -----")
        print(answer)

    cache.set(question, {"answer": answer, "sources": sources})   # 第7步：把这次问答存进缓存，备下次命中
    print("\n[7/7] 已写入缓存。")
    _print_sources(sources)                     # 打印出处


def _print_sources(sources: list[dict]) -> None:
    """小工具：把出处列表逐条打印（标题/日期/网址）。"""
    print("\n出处：")
    for s in sources:
        print(f"  · 《{s['title'][:40]}》 {s.get('date','')} {s.get('url','')}")   # .get(k,'')=取不到就用空串，避免 KeyError


def cmd_demo() -> None:
    """命令 demo：跑一个内置示例问题，方便一键看效果。"""
    cmd_ask("浙江省生育津贴从2025年7月1日起怎么发放？")


def main(argv: list[str]) -> None:
    """命令行分发：根据第一个参数决定跑哪个命令（等于前端的 switch(route)）。argv=命令行参数数组（等于 process.argv）。"""
    if len(argv) < 2:                           # 没给任何命令（argv[0] 是脚本名，argv[1] 才是命令）
        print(__doc__)                          # __doc__ = 本文件顶部那段三引号文档字符串，打印出来当帮助
        return
    cmd = argv[1]                               # 取命令名
    if cmd == "build":
        cmd_build()
    elif cmd == "ask":
        if len(argv) < 3:                       # ask 后面必须跟问题
            print('用法：python -m src.main ask "你的问题" [--rerank]')
            return
        use_rerank = "--rerank" in argv         # 检查参数里有没有 --rerank 这个 flag（等于 argv.includes('--rerank')）
        cmd_ask(argv[2], use_rerank)            # argv[2] = 问题文本
    elif cmd == "demo":
        cmd_demo()
    else:
        print(f"未知命令：{cmd}\n")
        print(__doc__)


if __name__ == "__main__":      # 直接运行本文件时（python -m src.main）才执行；被 import 时不执行
    main(sys.argv)              # sys.argv = 命令行传入的所有参数
