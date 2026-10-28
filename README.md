# medical-policy-rag · 医保/医疗政策 RAG 知识库助手

基于**本地医保政策**（浙江省 / 杭州市，政府门户主动公开文件）的检索问答系统。

> 📐 目录含义、数据流、七步链路对应哪些代码，见 [docs/架构说明.md](docs/架构说明.md)。

## ★ 边界声明（合规红线，README 和面试都要主动说清）
本项目只做**信息检索与流程辅助**：依据公开政策文件检索、并给出**带出处**的答复。
**不提供任何诊断建议、用药建议或诊疗决策**。医疗判断以执业医师和官方经办口径为准。

## 架构（RAG 两条线）
- **离线建库**：加载(data/raw) → 清洗 → 切分 → 向量化 → 入库(Faiss)
- **在线问答**：提问 → 向量化 → 检索 top-k →（重排）→ 拼 prompt → 生成 → **附政策出处**

## 七步链路与代码对应
| 步骤 | 模块 | 说明 |
|---|---|---|
| 1 加载 | `src/loader.py` | 读 `data/raw/*.md`，解析 front matter（标题/来源/日期） |
| 2 切分 | `src/chunker.py` | 技巧①：按句子边界累加到 400 字、段间重叠 60 字 |
| 3 向量化 | `src/embedder.py` | `BAAI/bge-small-zh-v1.5`，512 维，L2 归一化 |
| 4 入库 | `src/vectorstore.py` | Faiss `IndexFlatIP`（归一化后内积=余弦） |
| 5 检索 | `src/vectorstore.py` | query 向量化 → top-k；`src/rerank.py` 可选重排 |
| 6 生成 | `src/generator.py` | 技巧②：引用约束 prompt（只依据资料/逐句标来源/未覆盖即拒答） |
| 7 评估 | `eval/evaluate.py` | 检索层命中率/召回；`src/cache.py` 为技巧③缓存 |

## 目录
```
data/raw/         原始政策语料（55 篇 .md，抓取脚本可复现）
data/processed/   向量索引产物 index.faiss + chunks.json + cache.json（不进 git）
src/config.py     路径/模型/切分参数集中配置
src/loader.py     第1步 加载语料
src/chunker.py    第2步 切分（技巧①）
src/embedder.py   第3步 向量化
src/vectorstore.py 第4步入库 + 第5步检索
src/rerank.py     重排（可选，--rerank 启用）
src/generator.py  第6步 引用约束 prompt + 调 LLM（技巧②）
src/cache.py      生成层缓存（技巧③）
src/main.py       CLI：build / ask / demo
eval/dataset.json 评测集（种子6题，golden 逐字取自语料；待扩到30题）
eval/evaluate.py  第7步 检索层命中率/召回率
docs/决策记录.md  每个选型的依据（面试素材）
```

## 如何跑
```bash
# 环境（首次）
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env                   # 填 DEEPSEEK_API_KEY 才会真调大模型

# 离线建库（第1-4步）——首次会自动下载 bge 模型（走 hf-mirror 镜像）
python -m src.main build               # → 55篇 / 560切片 / 512维，产物落 data/processed

# 在线问答（第5-7步）
python -m src.main ask "浙江省生育津贴从2025年7月1日起怎么发放？"
python -m src.main ask "..." --rerank  # 启用重排
python -m src.main demo                # 内置示例问题
# 没配 API key 时进 dry-run：只打印拼好的「引用约束 prompt」+ 检索出处，不烧钱

# 评估（第7步）
python -m eval.evaluate --top-k 4
```

## 语料来源与复现
`data/raw/` 为抓取的公开政策文件。抓取脚本：上级目录 `../抓取医保语料.py`
（Playwright 渲染浙江省局/杭州市局列表页 → requests 抓正文，含限速/去重/断点续抓）。

## 成本与效果（真实跑出来的数字，可复算）
- 建库：55 篇文档 → 560 个切片，向量矩阵 (560, 512)，CPU 向量化约 1 分钟；索引 1.1 MB。
- 检索质量（`eval/dataset.json` 6 题种子，其中 5 题计分）：
  **hit@4 = 100%、Top1 命中 = 100%、平均召回 = 100%**。
- 单次问答 token 数 / 费用：待配 DEEPSEEK_API_KEY 后补真实数字。
- ⚠️ 种子题偏简单（答案原文直给），命中率虚高；扩到 30 题、加入跨文档/口语化/未覆盖后才有说服力——**简历里只写扩到 30 题后的真实数字**。
