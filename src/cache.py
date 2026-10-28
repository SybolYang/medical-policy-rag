"""【所属文件夹：src/ = 源代码】
【本文件角色：技巧③ 缓存 —— 相同问题不重复调大模型，直接返回上次答案】

为什么要缓存：医保高频问题就那几类（报销比例/异地就医/门诊慢特病），命中缓存既省 API 钱又降延迟。
本质就是「一个能存到硬盘的字典」：key=问题文本，value=上次的答案+出处。
为什么用 JSON 文件不用 Redis：单机、学习期、量小，Redis 是分布式高并发才需要的重武器，现在上是自找部署麻烦。
文件缓存重启不丢、能直接打开看，够用；上线多用户了再换 Redis。
"""

from __future__ import annotations    # 固定咒语：类型注解延迟求值

import json                           # JSON 读写（缓存存成 cache.json）

from . import config                  # 拿缓存文件路径 CACHE_PATH


class Cache:
    """问答缓存：内存里一个字典，每次写入同步落盘到 cache.json。"""

    def __init__(self, path=None):    # 构造函数；path=None 默认参数
        self.path = path or config.CACHE_PATH   # 没传路径就用 config 里默认的 cache.json（?? 的 Python 版）
        self._data: dict[str, dict] = {}        # 前缀 _ = 私有字段，外面别直接碰，都走 get/set
        if self.path.exists():                  # 如果缓存文件已存在（之前跑过）
            try:                                # try/except = 捕获异常（等于 JS 的 try/catch）
                self._data = json.loads(self.path.read_text(encoding="utf-8"))   # 读回来加载进内存
            except json.JSONDecodeError:        # 万一文件损坏/不是合法 JSON
                self._data = {}                 # 就当空缓存，不让程序崩（防御性处理）

    def get(self, key: str):
        """按问题取缓存。dict.get(k) 取不到返回 None 而不报错（直接 d[k] 取不到会崩），等于 JS 的 obj[k] 返回 undefined。"""
        return self._data.get(key)

    def set(self, key: str, value: dict) -> None:
        """存一条缓存并立刻落盘。"""
        self._data[key] = value                 # 先写进内存字典
        self.path.parent.mkdir(parents=True, exist_ok=True)   # 确保父目录存在
        self.path.write_text(                   # 整个字典写成 JSON 落盘（每次 set 都全量重写，量小无所谓）
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"   # ensure_ascii=False 中文原样存；utf-8 防崩
        )

    def __len__(self) -> int:
        """让 len(cache) 能返回缓存条数（__len__ 是 Python 的魔术方法，定义 len() 的行为）。"""
        return len(self._data)
