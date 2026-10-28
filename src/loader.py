"""【所属文件夹：src/ = 源代码】
【本文件角色：第1步 加载 —— 七步链路的入口，把硬盘上的政策文件读成程序能用的对象】

干什么：把 data/raw/ 下的 .md 政策文件读进来，解析出「标题/来源URL/日期」当元数据，
正文留给下一步切分。产出一组 Document 对象，交给 chunker.py。
为什么：RAG 的第一件事就是「把知识装进程序」。跟做：B P32 文档加载。
"""

from __future__ import annotations   # 固定咒语：让类型注解延迟求值，list[Document] 这种写法在旧 Python 也不报错

import re                            # 正则表达式库（等于 JS 的 RegExp），用来从文本里抠出标题/来源/日期
from dataclasses import dataclass    # dataclass 装饰器：自动生成构造函数和打印格式（近似 TS 的 interface + class）

from . import config                 # 相对导入同目录的 config（. 表示当前包 src/），拿路径等配置


@dataclass                           # 加上它，下面就自动获得 Document(...) 构造函数、__repr__ 等
class Document:
    """一篇完整政策文档的数据结构 —— 相当于 TS 的 interface Document{...}，但能真的 new 出来。
    为什么用 dataclass 而不是 dict：dict 拼错 key（如 "titel"）运行时才崩，dataclass 拼错字段 IDE 立刻标红。"""
    title: str          # 标题
    source_url: str     # 政策原文的政府网址（用来给用户标出处）
    date: str           # 发布日期
    text: str           # 正文（去掉头部元信息后的纯内容）
    filename: str       # 文件名（评估时用它比对「检索到的 vs 标准答案」）


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """把文件头部的元信息（# 标题 / > 来源 / > 日期）拆出来，返回 (元数据字典, 正文)。
    函数名前缀 _ = 约定的「内部私有函数」，只给本文件用，不从外面调（等于 JS 里不 export 的函数）。
    返回类型 tuple[dict, str] = 一个元组打包两个值：第0个是 dict、第1个是 str。"""
    meta: dict[str, str] = {}          # 准备一个空字典收集元数据
    lines = raw.splitlines()           # 按换行把整篇拆成「行的列表」（等于 JS 的 raw.split("\n")）
    body_start = 0                     # 记录正文从第几行开始（头部元信息之后的那一行）
    for i, line in enumerate(lines):   # 遍历每一行，i 是行号、line 是行内容（等于 JS 的 lines.forEach((line,i)=>...)）
        if line.startswith("# "):      # 以 "# " 开头 = Markdown 一级标题
            meta.setdefault("title", line[2:].strip())   # line[2:] 去掉开头的 "# "，strip() 去首尾空格；setdefault=没有才设
            body_start = i + 1         # 标题之后可能是正文起点
            continue                   # continue = 跳过本次循环剩下的判断，直接看下一行
        m = re.match(r">\s*来源：\s*(.+)", line)   # 匹配形如 "> 来源：http://..." 的行；\s*=任意空格，(.+)=捕获后面的内容
        if m:                          # 匹配成功（m 不是 None）
            meta["source_url"] = m.group(1).strip()   # group(1) = 取第1个括号 (.+) 捕获到的内容，即 url（等于 JS 的 m[1]）
            continue
        m = re.match(r">\s*日期：\s*(.+)", line)   # 同理匹配 "> 日期：2025-07-16"
        if m:
            meta["date"] = m.group(1).strip()      # 取出日期
            continue
        if line.strip() == "---":      # 遇到分隔线 --- 说明头部结束
            body_start = i + 1         # 分隔线的下一行就是正文起点
            continue
    body = "\n".join(lines[body_start:]).strip()   # 把 body_start 之后的所有行重新用换行拼回正文，去掉首尾空白
    return meta, body                  # 返回元数据和正文两个值


def load_documents(raw_dir=None) -> list[Document]:
    """加载 raw_dir 目录下所有 .md → Document 列表。raw_dir=None 是默认参数，不传就用 config 里的默认目录。"""
    raw_dir = raw_dir or config.RAW_DIR   # 「没传就用默认」的惯用写法（等于 JS 的 dir = dir ?? config.RAW_DIR）
    docs: list[Document] = []             # 准备空列表收集结果
    for path in sorted(raw_dir.glob("*.md")):   # glob("*.md")=列出所有 .md 文件；sorted 排序保证每次顺序一致（重要！向量库靠顺序对齐）
        raw = path.read_text(encoding="utf-8")  # 读文件全部内容，显式指定 utf-8（否则 Windows 默认 GBK，中文会崩）
        meta, body = _parse_front_matter(raw)   # 拆出元数据和正文
        if not body:                     # 正文为空（比如空文件）就跳过，不浪费
            continue
        docs.append(                     # 组装成一个 Document 对象加进列表
            Document(
                title=meta.get("title", path.stem),       # .get(k, 默认) = 取不到就用文件名兜底；path.stem=不含扩展名的文件名
                source_url=meta.get("source_url", ""),    # 取不到来源就用空字符串
                date=meta.get("date", ""),                # 取不到日期就用空字符串
                text=body,                                # 正文
                filename=path.name,                       # path.name=完整文件名（含 .md），评估比对用
            )
        )
    return docs                          # 返回所有文档


if __name__ == "__main__":               # 经典写法：只有「直接运行本文件」时才执行下面，被别的文件 import 时不执行（等于 Node 的 require.main===module）
    loaded = load_documents()            # 自测：加载一遍
    print(f"加载 {len(loaded)} 篇文档")   # f-string = 格式化字符串，{...} 里塞变量（等于 JS 的模板串 `加载 ${n} 篇`）
    d = loaded[0]                        # 取第一篇当示例
    print(f"示例：{d.title[:30]} | 日期={d.date} | 正文 {len(d.text)} 字")   # d.title[:30]=标题前30个字符（切片，等于 JS 的 slice(0,30)）
