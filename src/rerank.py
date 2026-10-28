"""【所属文件夹：src/ = 源代码】
【本文件角色：（可选）重排 —— 检索回来的 top-k 再排一次序，把最相关的顶到第一条】

为什么需要：向量检索快但粗。重排 = 用更细的方法给检索结果重新打分排序。
这里用最轻量的「关键词重叠」打分做 baseline，不依赖额外模型（离线可跑）。
⚠️它只是教学 baseline，不是真本事：生产中应换成 cross-encoder（如 bge-reranker）效果更好。
重排不是七步链路的必经步骤，main 里默认关闭，加 --rerank 才启用。跟做：B P44 重排。
"""

from __future__ import annotations    # 固定咒语：类型注解延迟求值

from .chunker import Chunk            # 拿 Chunk 结构（重排的是检索回来的 Chunk）


def _keywords(text: str) -> set[str]:
    """从一段文字里抠出「关键词集合」（用于算重叠度）。前缀 _ = 内部私有函数。"""
    import re                          # 局部导入正则库（只这函数用）

    # findall(r"[...]{2,}") = 找出所有「长度≥2 的连续中文/字母/数字片段」当关键词，返回 list；set() 转成集合（自动去重）
    # \u4e00-\u9fa5 = Unicode 里中文汉字的范围
    return set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", text))


def rerank(query: str, hits: list[tuple[float, Chunk]], top_n: int | None = None):
    """给检索结果 hits 重新打分排序。top_n=None 表示不截断、全返回。"""
    qk = _keywords(query)              # 先把「问题」的关键词抠出来
    scored = []                        # 收集重新打分后的结果
    for vec_score, chunk in hits:      # 遍历每个 (原向量分, 切片)
        ck = _keywords(chunk.text)     # 抠出这段切片的关键词
        overlap = len(qk & ck) / (len(qk) or 1)   # qk & ck = 集合取交集（问题和切片共有的关键词）；
                                                  #   除以问题关键词总数 = 重叠比例；(len(qk) or 1) 防止除以 0（qk 为空时当 1）
        # 加权融合：向量分为主(0.7) + 关键词重叠微调(0.3)。为什么不全用关键词：那就退回老式关键词搜索、丢了语义能力
        scored.append((vec_score * 0.7 + overlap * 0.3, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)   # 按新分（元组第0个 x[0]）降序排；lambda=箭头函数（等于 JS 的 .sort((a,b)=>b[0]-a[0])）
    return scored[:top_n] if top_n else scored      # 传了 top_n 就只返回前 top_n 条，否则全返回
