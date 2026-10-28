"""【所属文件夹：src/ = 源代码】
【本文件角色：第3步 向量化 —— 把「文字」翻译成计算机能算相似度的「数字向量」】

为什么需要：计算机不懂文字，只懂数字。「报销比例」和「报销标准」字面不同但意思近，
要把它们转成向量（一串数字），让「意思近的向量在空间里距离近」，这样才能按语义搜索
而不是按关键词死匹配。模型 BAAI/bge-small-zh-v1.5：中文、512维、CPU 就能跑。
首次运行会从 HuggingFace 镜像下载到本地缓存，之后离线可用。跟做：B P36 向量化。
"""

from __future__ import annotations    # 固定咒语：类型注解延迟求值

import numpy as np                    # numpy：Python 的数值计算库，处理向量/矩阵（np.ndarray 就是多维数组）

from . import config                  # 拿模型名 EMBED_MODEL

_MODEL = None                         # 模块级全局变量，缓存已加载的模型（初始为 None，还没加载）


def get_model():
    """懒加载单例：第一次调用才真正加载模型，之后复用同一个，不重复加载。
    为什么懒加载：模型加载要几秒 + 占几百 MB 内存，每次 embed 都重载会慢死。
    等于前端的 let instance=null; function get(){ return instance ??= new HeavyModel(); }"""
    global _MODEL                     # 声明「我要修改模块级的那个 _MODEL 全局变量」（不加 global 会被当成新的局部变量）
    if _MODEL is None:                # is None = 判断还没加载过
        from sentence_transformers import SentenceTransformer   # 把重库的 import 放函数里：只有真用到才导入，让不需要模型的模块启动更快
        _MODEL = SentenceTransformer(config.EMBED_MODEL)        # 加载模型（首次会触发下载），存进全局缓存
    return _MODEL                     # 返回模型实例


def embed_texts(texts: list[str]) -> np.ndarray:
    """把一批文字转成向量矩阵。输入 N 条文字 → 输出形状 (N, 512) 的矩阵。"""
    model = get_model()               # 拿到（已缓存的）模型
    vecs = model.encode(              # encode = 模型的核心方法：文字进、向量出
        texts,                        # 要向量化的文字列表
        normalize_embeddings=True,    # 【关键】L2 归一化：把每个向量长度缩放成 1。归一化后「内积=余弦相似度」，
                                      #   这样下游 Faiss 用最简单的内积索引就能算余弦，省一步除法
        show_progress_bar=len(texts) > 50,   # 超过 50 条才显示进度条（少量就不刷屏）
        batch_size=32,                # 32 条一批喂给模型，比一条条快得多（CPU/GPU 都吃批量）
    )
    return np.asarray(vecs, dtype="float32")   # 转成 numpy 数组，dtype=float32（32位浮点，比64位省一半内存，精度对检索够用）


def embed_query(query: str) -> np.ndarray:
    """把「单个问题」向量化（在线问答第5步用）。
    复用 embed_texts：把单条问题包成 [query] 列表，encode 后取第 0 个结果。"""
    return embed_texts([query])[0]


if __name__ == "__main__":            # 直接运行本文件的自测
    v = embed_texts(["医保报销比例", "异地就医备案"])   # 向量化两句测试文字
    # np.linalg.norm = 算向量的模长（长度）；归一化后应≈1，用来验证 normalize 生效了
    print(f"向量维度={v.shape}, 已归一化(模长≈1)={np.linalg.norm(v[0]):.3f}")   # :.3f = 保留3位小数
