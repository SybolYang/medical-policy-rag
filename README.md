# medical-policy-rag · 医保/医疗政策 RAG 知识库助手

基于**本地医保政策**（浙江省 / 杭州市，政府门户主动公开文件）的检索问答系统。

## ★ 边界声明（合规红线，README 和面试都要主动说清）
本项目只做**信息检索与流程辅助**：依据公开政策文件检索、并给出**带出处**的答复。
**不提供任何诊断建议、用药建议或诊疗决策**。医疗判断以执业医师和官方经办口径为准。

## 架构（RAG 两条线）
- **离线建库**：加载(data/raw) → 清洗 → 切分 → 向量化 → 入库(Faiss)
- **在线问答**：提问 → 向量化 → 检索 top-k →（重排）→ 拼 prompt → 生成 → **附政策出处**

## 目录
```
data/raw/         原始政策语料（.md，抓取脚本可复现）
data/processed/   切分/向量索引产物（不进 git）
src/loader.py     加载语料            （跟 B P32）
src/chunker.py    切分：固定/语义/表格 （跟 B P33 / P57）
src/embedder.py   向量化              （跟 B P10 / P36）
src/vectorstore.py Faiss 入库+检索     （跟 B P11）
src/rerank.py     重排（可选）         （跟 B P56）
src/generator.py  拼 prompt+生成+出处  （跟 B P44）
src/cache.py      缓存                （跟 B P38 / P39）
src/main.py       问答入口            （跟 B P40）
eval/dataset.json 30 题评测集（人工标答案）
eval/evaluate.py  命中率/召回率脚本
docs/决策记录.md  每个选型的依据（面试素材）
```

## 如何跑
```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env               # 然后填 API key
python src/main.py
```

## 语料来源与复现
`data/raw/` 为抓取的公开政策文件。抓取脚本：上级目录 `../抓取医保语料.py`
（Playwright 渲染浙江省局/杭州市局列表页 → requests 抓正文，含限速/去重/断点续抓）。

## 成本（跑通后填真实数字，简历/面试要能复算）
- 建库耗时 / 文档数 / chunk 数：
- 单次问答 token 数 / 费用：
- 检索 top-k 命中率（eval 结果）：
