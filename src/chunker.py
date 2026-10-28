"""【所属文件夹：src/ = 源代码】
【本文件角色：第2步 切分（技巧①）—— 把整篇长政策切成一小段一小段】

为什么必须切：①大模型上下文长度有限，整篇塞不下；②检索以「段」为单位，整篇丢进去
问「起付线」却召回一整篇无关内容，召回不准。所以先切成小段再向量化。
为什么段间要重叠：避免一句关键规定正好被切在两段边界上、导致语义丢失。
切法：按句子边界（。！？；和换行）累加到 CHUNK_SIZE 字，比无脑按字数硬切召回更准。
跟做：B P33 文档切分。
"""

from __future__ import annotations              # 固定咒语：类型注解延迟求值（见 loader.py 说明）

import re                                       # 正则库，用来按标点切句子
from dataclasses import dataclass, asdict       # dataclass=数据结构装饰器；asdict=把 dataclass 对象转成字典（存 JSON 用）

from . import config                            # 拿切分参数 CHUNK_SIZE / CHUNK_OVERLAP
from .loader import Document                    # 拿上一步定义好的 Document 结构（本模块吃 Document、吐 Chunk）

# 句子分隔正则：(?<=[。！？；\n]) 叫「后行断言」——在这些标点的「后面」切一刀，但不吃掉标点本身
# 为什么按句子而不是按固定字数硬切：硬切会把「支付期自2023年1月6日起至 | 2023年3月31日止」劈成两段，
# 检索到前半段就答不出完整日期。按句子切保证「一个完整语义单元不被劈开」。
# 变量名全大写 = 约定它是常量；re.compile 预编译好复用，省性能
_SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


@dataclass
class Chunk:
    """一个「切片」的数据结构（一篇文档会切出很多个 Chunk）。
    注意它比 Document 多了 chunk_id，还「冗余」存了 title/filename/source_url/date。
    为什么冗余存来源：检索时只召回「段」，这段必须自带「我出自哪篇、url 是啥」，
    否则答完没法给用户标出处。这叫反规范化——牺牲一点存储空间，换查询时的方便。"""
    chunk_id: int       # 这是全库第几个切片（0,1,2...，也是它在向量库里的行号）
    text: str           # 这一小段的文字内容
    title: str          # 出自哪篇（冗余存，方便标出处）
    filename: str       # 出自哪个文件（评估时比对用）
    source_url: str     # 原文网址（标出处用）
    date: str           # 发布日期

    def to_dict(self) -> dict:
        """把自己转成普通字典，存成 JSON 时用（asdict 是 dataclass 自带的工具函数）。"""
        return asdict(self)


def _split_sentences(text: str) -> list[str]:
    """把一整段正文按句子切成「句子的列表」。前缀 _ = 内部私有函数。"""
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]   # 列表推导式（等于 JS 的 text.split(正则).map(p=>p.trim())）
    return [p for p in parts if p]                         # 再过滤掉空字符串（连续标点会切出空串）


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,   # 默认每段 400 字
    overlap: int = config.CHUNK_OVERLAP,   # 默认重叠 60 字
) -> list[str]:
    """【🔴核心】按句子累加成段，段尾保留 overlap 个字带入下一段。返回切好的段落列表。"""
    sentences = _split_sentences(text)   # 先把正文拆成一句句
    chunks: list[str] = []               # 收集切好的段
    buf = ""                             # buffer 缓冲区：正在攒的当前段（等于你 JS 里攒字符串的临时变量）
    for sent in sentences:               # 一句句往 buf 里塞
        # 边界情况：单独一句就超过 chunk_size（少见，比如一整段没标点）→ 退化成按字数硬切
        if len(sent) > chunk_size:
            if buf:                      # 如果 buf 里已经攒了东西，先把它收成一段
                chunks.append(buf)
                buf = ""                 # 清空重开
            # range(起点, 终点, 步长)：每隔 (chunk_size-overlap) 取一个切点（等于 JS 的 for(i=0;i<len;i+=step)）
            for i in range(0, len(sent), chunk_size - overlap):
                chunks.append(sent[i : i + chunk_size])   # sent[i:i+400] = 从 i 开始取 400 字（切片，等于 JS 的 substr）
            continue                     # 这句处理完，看下一句
        # 正常情况：如果「当前 buf + 这句」会超过 chunk_size，且 buf 非空 → 该切段了
        if len(buf) + len(sent) > chunk_size and buf:
            chunks.append(buf)           # 把攒满的 buf 收成一段
            tail = buf[-overlap:] if overlap else ""   # buf[-60:] = 取 buf 最后 60 字（负数索引=从末尾数，等于 JS 的 slice(-60)）
            buf = tail + sent            # 新段 = 上一段尾巴60字（重叠！）+ 当前这句 → 保证边界句子在两段都出现，不漏召回
        else:
            buf += sent                  # 还没满，继续往当前段塞这句
    if buf:                              # 循环结束后 buf 里可能还剩最后一段没满的，收尾
        chunks.append(buf)
    return chunks


def build_chunks(docs: list[Document]) -> list[Chunk]:
    """把「文档列表」整体切成「切片列表」：遍历每篇文档，切分，给每段编号并附上来源信息。"""
    chunks: list[Chunk] = []   # 收集所有切片
    cid = 0                    # 全局切片计数器（chunk_id），跨文档连续递增
    for doc in docs:           # 逐篇文档
        for piece in chunk_text(doc.text):   # 把这篇的正文切成若干段，逐段处理
            chunks.append(
                Chunk(
                    chunk_id=cid,            # 全局唯一编号 = 它在向量库里的行号
                    text=piece,              # 这段文字
                    title=doc.title,         # 以下 4 个都从所属文档「冗余」复制过来，方便日后标出处
                    filename=doc.filename,
                    source_url=doc.source_url,
                    date=doc.date,
                )
            )
            cid += 1                         # 编号 +1
    return chunks


if __name__ == "__main__":                   # 直接运行本文件时的自测（被 import 时不跑）
    from .loader import load_documents       # 局部导入加载函数

    docs = load_documents()                  # 先加载全部文档
    all_chunks = build_chunks(docs)          # 再切分
    print(f"{len(docs)} 篇文档 → {len(all_chunks)} 个切片")
    lens = [len(c.text) for c in all_chunks] # 列表推导：取出每段的字数
    print(f"每段字数：min={min(lens)} max={max(lens)} avg={sum(lens)//len(lens)}")   # //=整除（向下取整的平均值）
    print("--- 示例切片 ---")
    print(all_chunks[0].text[:120], "...")   # 打印第一段的前 120 字看看效果
