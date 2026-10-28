"""【所属文件夹：src/ = 源代码，七步链路的每个模块都在这】
【本文件角色：全局配置中心 —— 相当于前端的 constants.ts / .env，所有模块都 import 它】

为什么单独抽一个 config：路径、模型名、切分参数这些是「以后调参天天要改的数」，
集中放一处 = 单一事实来源，改一个数全项目生效，不用翻遍每个文件。
"""

import os                              # Python 标准库：读写环境变量、和操作系统打交道（等于 Node 的 process）
from pathlib import Path              # 面向对象的路径工具（等于 Node 的 path 模块），比裸字符串拼路径安全

# ── 一、定位项目根目录 ──────────────────────────────────────────────
# __file__ = 当前这个文件的路径（等于 Node 的 __filename）
# .resolve() = 转成绝对路径（消掉 ../ 和软链接）
# .parent = 上一级目录；.parent.parent = 上两级
# 从 src/config.py 往上两级 → 项目根目录 medical-policy-rag/
# 为什么这么写：不管你在哪个目录敲命令运行，算出来的路径都对，不依赖「当前工作目录」
ROOT = Path(__file__).resolve().parent.parent

# ── 二、各个数据文件的路径（都用 Path 的 / 运算符拼，自动适配 Win 的 \ 和 Mac 的 /）──
RAW_DIR = ROOT / "data" / "raw"            # 原始语料目录：55 篇本地医保政策 .md（唯一的真数据，进 git）
PROCESSED_DIR = ROOT / "data" / "processed"  # 加工产物目录：向量索引等（可重建，不进 git）
INDEX_PATH = PROCESSED_DIR / "index.faiss"   # Faiss 向量索引文件（第4步入库的产物）
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"  # 切片的文字+元数据（和 index.faiss 按行号一一对齐）
CACHE_PATH = PROCESSED_DIR / "cache.json"    # 问答缓存文件（技巧③，存问过的问题和答案）

EVAL_DATASET = ROOT / "eval" / "dataset.json"  # 评测集题库（第7步评估用）

# ── 三、环境变量默认值 ─────────────────────────────────────────────
# setdefault(k, v) = 「如果 k 还没设，就设成 v；已经设了就不动」
# HF_ENDPOINT：HuggingFace 国内镜像，让模型下载走 hf-mirror（否则国内被墙下不动）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关掉 Windows 上 huggingface 缓存的 symlink（软链接）警告——无害，只是刷屏烦人
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── 四、模型与检索参数 ─────────────────────────────────────────────
# getenv(k, default) = 读环境变量，没设就用默认值。允许你日后不改代码、只改 .env 就换模型
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")  # 中文嵌入模型，输出 512 维向量
TOP_K = 4   # 检索时召回最相近的前 4 段（越大召回越全但噪音越多、越费 token）

# ── 五、切分参数（技巧①：切分策略）─────────────────────────────────
CHUNK_SIZE = 400     # 每段目标字数：太短丢上下文，太长召回不准 + 费 token
CHUNK_OVERLAP = 60   # 段间重叠字数：把上一段尾巴 60 字带进下一段，防止关键句正好被切断
