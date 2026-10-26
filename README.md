# JobPilot

JobPilot 是一个本地优先的求职岗位匹配 Agent。它读取岗位 JD、候选人技能和简历资料，通过工具调用与向量检索生成有证据约束的岗位匹配报告。

## 项目亮点

- 使用 DeepSeek 的 OpenAI-compatible API
- 基于 Deep Agents Harness 构建 Agent 主循环
- Tool Calling：岗位读取、候选人技能、简历检索、技能匹配
- Session / Profile Memory：保存用户长期偏好
- RAG：Markdown 分块、中文分词、TF-IDF 向量化和余弦相似度 Top-K 检索
- 结构化输出：Pydantic Schema + 两阶段报告生成，最终保存 JSON
- 失败可追踪：保留真实故障案例和修复记录

## 架构

```text
User
  -> jobpilot.py
  -> Deep Agents Harness
  -> Tools
       get_job_description
       get_candidate_skills
       get_resume_evidence
       calculate_skill_match
  -> Analysis Markdown
  -> Structured Formatter
  -> report_v05.json
```

## 技术栈

- Python 3.12
- Deep Agents 0.7.32
- LangChain OpenAI 1.7.2
- DeepSeek API
- jieba
- scikit-learn
- Pydantic

## 快速开始

### 1. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 配置 API Key

复制模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写：

```text
DEEPSEEK_API_KEY=你的DeepSeek密钥
```

不要提交 `.env`。

### 3. 准备输入数据

```powershell
New-Item -ItemType Directory -Force data
Copy-Item examples\profile.example.json data\profile.json
Copy-Item examples\resume.example.md data\resume.md
Copy-Item examples\job.example.md data\job.md
Copy-Item examples\skills.example.json data\skills.json
```

然后把你自己的简历、岗位 JD、技能证据替换进去。

### 4. 运行

```powershell
.\.venv\Scripts\python.exe jobpilot.py
```

输出文件：

```text
data/report_v05.md
data/report_v05.json
```

## Docker

```powershell
docker build -t jobpilot .
docker run --rm --env-file .env -v "${PWD}/data:/app/data" jobpilot
```

## 当前限制

- 当前检索器是 TF-IDF 稀疏向量基线，不是语义 Embedding。
- 尚未实现向量数据库持久化和 Rerank。
- 尚未提供 Web 前端。
- 尚未实现自动重试、限流和完整的模型评测集。
- 结构化格式化阶段是独立模型调用，会增加一次 Token 成本。

## 后续计划

- FastAPI 服务层
- Vue 或简单 Web 界面
- 语义 Embedding 与向量数据库
- Top-K 召回评估和答案引用评测
- Tool Retry、幂等保存与人工确认
- 投递记录持久化

## 真实故障案例

详见 [docs/FAILURE_LOG.md](docs/FAILURE_LOG.md)。

## 简历项目描述示例

> 独立开发求职岗位匹配 Agent，使用 Python、Deep Agents 和 DeepSeek API，实现 Agent 工具调用、简历 RAG 检索、候选技能去重与结构化 JSON 输出。将简历切分为 14 个语义 Chunk，使用 jieba + TF-IDF 完成批量 Top-K 检索；通过两阶段生成解决主 Agent 工具循环与结构化输出冲突，并记录环境、数据清洗、标题分块和输出校验等故障案例。

## 说明

本项目用于学习与求职展示。所有候选人技能和经历必须来自真实资料，不应为了匹配岗位而虚构。