"""【所属文件夹：src/ = 源代码】
【本文件角色：第6步 生成（技巧②：引用约束 prompt）—— 把「问题 + 检索到的原文段」拼成 prompt，让大模型照着答】

核心是那段 SYSTEM_PROMPT：把「只依据参考资料作答、逐句标来源、没覆盖就说不知道、不得编造」写死。
医疗场景这是合规命门——大模型天生爱编（幻觉），编错报销比例=事故，所以宁可拒答不可幻觉。
有 API Key（.env 里的 DEEPSEEK_API_KEY）就真调大模型；没有就进 dry-run，只把拼好的 prompt 打印出来，
让你不花钱也能看清「引用约束」长什么样。跟做：B P11 生成 + 引用。
"""

from __future__ import annotations    # 固定咒语：类型注解延迟求值

import os                             # 读环境变量（API key 等）

import requests                       # HTTP 请求库（等于 JS 的 fetch），用来调大模型接口
from dotenv import load_dotenv         # 从 .env 文件加载环境变量到程序里（让 key 不写死在代码里）

from .chunker import Chunk            # 拿 Chunk 结构（检索结果里带的是 Chunk）

load_dotenv()                          # 执行加载：读取项目根目录的 .env 文件（没有也不报错）

# 【🔴核心·技巧②】系统提示词：定义大模型的角色和铁律。这是整个医疗 RAG 最关键的一段。
# 为什么单独拎成模块级大写常量、还写这么死：①显眼、可审计（合规/面试官一眼看到你的安全约束）；
# ②不埋在 f-string 里方便日后单独调。三引号 """...""" = 多行字符串（等于 JS 的反引号模板串）。
# 下面三引号里是「原样喂给大模型」的字符串，不能塞注释（会把 ← 这些字符也当成提示词发出去）。
# 四条铁律的意图逐条说明放在这里（字符串外）：
#   1. 只依据参考资料 → 防幻觉：把模型锁死在检索到的原文里
#   2. 每个结论标 [来源N] → 可溯源：给用户出处
#   3. 资料不足就说「未覆盖」 → 拒答能力：不知道就说不知道
#   4. 不给诊断/用药建议 → 合规边界：不越界给医疗建议
SYSTEM_PROMPT = """你是医保政策信息检索助手，只做政策信息的检索与转述。
严格要求：
1. 只依据下方【参考资料】作答，不得使用资料以外的任何知识。
2. 每个结论后用 [来源N] 标注它来自哪份资料（N 对应参考资料编号）。
3. 若【参考资料】不足以回答问题，直接说「现有政策语料未覆盖此问题」，不得编造。
4. 不提供任何诊断、用药、诊疗建议，只做政策条文的信息检索与转述。"""


def build_prompt(question: str, hits: list[tuple[float, Chunk]]) -> str:
    """把「问题 + 检索到的切片」拼成一段带编号参考资料的 user prompt（喂给大模型的用户消息）。"""
    refs = []                                          # 收集拼好的每条参考资料
    for i, (_score, c) in enumerate(hits, 1):          # enumerate(hits,1)=遍历时同时拿到序号 i，从 1 开始（等于 JS 的 map 但序号从1起）
                                                       #   _score 前缀 _ = 「这个值我拿到了但不用」（只要 chunk c，不要相似度分）
        refs.append(
            f"[来源{i}]《{c.title}》({c.date})\n{c.text}"   # 拼成「[来源1]《标题》(日期)\n正文」；这个编号 i 就是让模型标 [来源N] 用的
        )
    refs_block = "\n\n".join(refs) if refs else "（无检索结果）"   # 把各条参考资料用空行连起来；一条都没有就用占位文字
    return (                                           # 拼出最终 user prompt：参考资料 + 问题 + 作答要求
        f"【参考资料】\n{refs_block}\n\n"
        f"【问题】\n{question}\n\n"
        f"请依据参考资料回答，并逐条标注 [来源N]。"
    )


def call_llm(question: str, hits: list[tuple[float, Chunk]]) -> str | None:
    """调 DeepSeek 生成答案。返回 str=答案；返回 None=没配 key（触发上游 dry-run）。
    返回类型 str | None 表示「要么字符串要么 None」（等于 TS 的 string | null）。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")            # 从环境变量读 key（key 从 .env 读、不写死在代码，密钥绝不进 git）
    if not api_key:                                    # 没配 key
        return None                                    # 返回 None → main.py 会转而打印 prompt 做 dry-run
    base = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")   # 接口地址（getenv 第二参=没设就用默认）
    model = os.getenv("LLM_MODEL", "deepseek-chat")                # 模型名
    resp = requests.post(                              # 发 POST 请求（等于 fetch(url,{method:'POST',...})）
        f"{base}/chat/completions",                    # DeepSeek 用 OpenAI 兼容格式，所以路径是 /chat/completions
        headers={"Authorization": f"Bearer {api_key}"},   # 鉴权头：Bearer + key
        json={                                         # 请求体（json= 会自动序列化并设 Content-Type）
            "model": model,
            "messages": [                              # 对话消息数组：先 system（定规矩）再 user（给资料和问题）
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits)},
            ],
            "temperature": 0.1,                        # 温度调低=输出更确定、更少发挥（政策问答要稳，不要创意）
        },
        timeout=60,                                    # 60 秒超时，防止卡死
    )
    resp.raise_for_status()                            # HTTP 状态码非 2xx 就抛异常（等于 JS 里 if(!res.ok) throw）
    return resp.json()["choices"][0]["message"]["content"]   # 从返回 JSON 层层取出答案文本（等于 JS 的 data.choices[0].message.content）
