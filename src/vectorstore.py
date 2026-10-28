"""【所属文件夹：src/ = 源代码】
【本文件角色：第4步 入库 + 第5步 检索 —— 一个文件干两步，因为「存」和「查」操作的是同一个 Faiss 索引】

为什么需要向量库：560 个向量，提问时要快速找出「最像的 4 个」。自己写循环算也行，
但数据上万就慢。Faiss 是 Facebook 开源的向量检索库，专门干这个。
入库：把所有切片向量存进 IndexFlatIP；向量存 index.faiss，文字/元数据存 chunks.json，两者按行号对齐。
检索：把问题也向量化，去索引里搜 top-k 最相近的切片。跟做：B P57 Faiss 索引。
"""

from __future__ import annotations   # 固定咒语：类型注解延迟求值

import json                          # JSON 读写（存/取 chunks.json 里的切片文字和元数据）

import faiss                         # Facebook 的向量检索库
import numpy as np                   # 数值库，处理向量数组

from . import config                 # 拿路径、TOP_K 等配置
from .chunker import Chunk           # 拿上一步的 Chunk 结构（检索出来要还原成 Chunk 返回）


class VectorStore:
    """向量库的封装：内部同时持有「Faiss 索引（存向量）」和「chunks 列表（存文字+来源）」，靠行号对齐。
    为什么要同时存两份：Faiss 只存向量、搜出来只给你「第几号」（行号），它不存文字。
    所以要自己留一份 chunks，用行号 self.chunks[i] 取回对应的文字和出处。"""

    def __init__(self, index: faiss.Index, chunks: list[Chunk]):   # 构造函数（等于 JS 的 constructor）；self = this
        self.index = index      # Faiss 索引对象（存向量、负责搜）
        self.chunks = chunks    # 切片列表（存文字+来源，靠下标和 index 里的向量一一对齐）

    @classmethod                # 类方法（等于 TS 的 static）：用 VectorStore.build(...) 调，不用先 new 对象
    def build(cls, chunks: list[Chunk], vectors: np.ndarray) -> "VectorStore":   # cls = 类本身
        """建库：拿切片和它们的向量，造一个 Faiss 索引并灌入向量。"""
        dim = vectors.shape[1]              # shape=(560,512)，shape[1]=512，即向量维度
        index = faiss.IndexFlatIP(dim)      # Flat=暴力精确搜索(不加速但100%准)，IP=内积(Inner Product)
                                            #   为什么用 Flat 不用 IVF/HNSW(近似加速)：只有560条，暴力搜毫秒级，
                                            #   上近似索引是过度设计还牺牲精度。数据上万再换——这个取舍面试能加分
        index.add(vectors)                  # 把 560 个向量灌进索引
        return cls(index, chunks)           # 用索引+切片造一个 VectorStore 对象返回

    def save(self) -> None:
        """把索引和切片存到硬盘（data/processed/）。"""
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)   # 确保目录存在；parents=连带建父目录，exist_ok=已存在也不报错
        # 【踩过的坑，重点】本来该用一行 faiss.write_index(index, path)，但它在中文路径下崩：
        #   faiss 底层是 C++，用 fopen 打开文件；中文 Windows 的 fopen 按 GBK 解析路径，
        #   而 Python 传过去的是 UTF-8 字节，「工作」两字对不上 → C++ 报「文件不存在」。
        #   修法：绕开 C++ 的文件读写。serialize_index 只在内存里把索引转成字节（不碰文件系统），
        #   再用 Python 的 open（能正确处理 Unicode 路径）写盘。让 C++ 只管算、Python 管文件。
        blob = faiss.serialize_index(self.index)      # 把索引序列化成字节（内存操作，不落盘）
        with open(config.INDEX_PATH, "wb") as f:      # with...as = 上下文管理器，出块自动关文件（等于 try/finally close）；"wb"=写+二进制
            f.write(np.asarray(blob).tobytes())       # 把字节写进 index.faiss
        data = [c.to_dict() for c in self.chunks]     # 把每个 Chunk 转成字典（列表推导式）
        with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:   # "w"=写文本，显式 utf-8 防中文崩
            json.dump(data, f, ensure_ascii=False, indent=2)   # 存 JSON；ensure_ascii=False 让中文原样存（不转成 \uXXXX）；indent=2 美化缩进

    @classmethod
    def load(cls) -> "VectorStore":
        """从硬盘把索引和切片读回来（在线问答时用）。是 save 的逆操作。"""
        with open(config.INDEX_PATH, "rb") as f:                  # "rb"=读+二进制
            blob = np.frombuffer(f.read(), dtype=np.uint8)        # 把读到的字节还原成 numpy 数组（uint8=每个元素1字节）
        index = faiss.deserialize_index(blob)                     # 反序列化回 Faiss 索引对象
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            data = json.load(f)                                   # 读回切片字典列表
        chunks = [Chunk(**d) for d in data]                       # Chunk(**d)：把字典 d 拆成关键字参数传给构造函数还原成 Chunk 对象
        return cls(index, chunks)

    @staticmethod             # 静态方法：不依赖实例（不用 self），就是个挂在类下的普通函数
    def exists() -> bool:
        """判断库有没有建过（两个产物文件都在才算）。ask 之前先检查，没建库就提示先 build。"""
        return config.INDEX_PATH.exists() and config.CHUNKS_PATH.exists()

    def search(self, query_vec: np.ndarray, k: int = config.TOP_K) -> list[tuple[float, Chunk]]:
        """【🔴核心·第5步检索】拿问题向量搜出最相近的 k 个切片，返回 [(相似度分, Chunk), ...]。"""
        q = np.asarray([query_vec], dtype="float32")   # 把单个查询向量包成「1行」的二维数组（Faiss 要求批量输入，哪怕只查一条）
        scores, idxs = self.index.search(q, k)         # Faiss 搜索：返回两个数组——scores(相似度分) 和 idxs(命中的行号)
        results = []
        for score, i in zip(scores[0], idxs[0]):       # zip=把两个数组并排配对遍历（等于 JS 的 scores[0].map((s,i)=>[s, idxs[0][i]])）；[0] 是取「第0条查询」的结果
            if i < 0:                                  # 库里不足 k 条时，Faiss 用 -1 占位，跳过（防御性处理）
                continue
            results.append((float(score), self.chunks[i]))   # 用行号 i 从 chunks 取回文字+来源，和分数打包
        return results
