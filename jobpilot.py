import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import jieba
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROFILE_PATH = DATA_DIR / "profile.json"
RESUME_PATH = DATA_DIR / "resume.md"
JOB_PATH = DATA_DIR / "job.md"
SKILLS_PATH = DATA_DIR / "skills.json"
REPORT_JSON_PATH = DATA_DIR / "report_v05.json"
REPORT_MD_PATH = DATA_DIR / "report_v05.md"

load_dotenv(BASE_DIR / ".env")

profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
candidate_skill_records = json.loads(
    SKILLS_PATH.read_text(encoding="utf-8")
)["candidate_skills"]
CURRENT_DATE = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def split_markdown_chunks(text: str) -> list[dict]:
    chunks = []
    current_h2 = ""
    current_h3 = ""
    current_time = ""
    current_role = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("### "):
            current_h3 = line.removeprefix("### ").strip()
            current_time = ""
            current_role = ""
            continue

        if line.startswith("## "):
            current_h2 = line.removeprefix("## ").strip()
            current_h3 = ""
            current_time = ""
            current_role = ""
            continue

        if line.startswith("# "):
            continue

        if line.startswith(("- ", "* ")) or re.match(r"^\d+\.\s+", line):
            content = re.sub(r"^(?:- |\* |\d+\.\s+)", "", line).strip()

            if content.startswith("时间："):
                current_time = content.removeprefix("时间：").strip()
            elif content.startswith("角色："):
                current_role = content.removeprefix("角色：").strip()

            heading = f"{current_h2} / {current_h3}".strip(" /")
            metadata = []
            if current_time:
                metadata.append(f"时间：{current_time}")
            if current_role:
                metadata.append(f"角色：{current_role}")

            metadata_text = "；".join(metadata)
            source = f"{heading}（{metadata_text}）" if metadata_text else heading

            chunks.append(
                {
                    "heading": heading,
                    "metadata": metadata_text,
                    "content": content,
                    "text": f"{source}\n{content}",
                }
            )

    return chunks


def tokenize(text: str) -> list[str]:
    return [
        token.strip().lower()
        for token in jieba.cut(text)
        if token.strip()
    ]


resume_chunks = split_markdown_chunks(
    RESUME_PATH.read_text(encoding="utf-8")
)
resume_vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
resume_vectors = resume_vectorizer.fit_transform(
    [chunk["text"] for chunk in resume_chunks]
)


@tool
def get_job_description() -> str:
    """Read and return the target job description."""
    return JOB_PATH.read_text(encoding="utf-8")


@tool
def get_candidate_skills() -> list[dict]:
    """Return the canonical evidence-backed candidate skill list."""
    return candidate_skill_records


@tool
def get_resume_evidence(queries: list[str], top_k: int = 5) -> list[dict]:
    """Retrieve the most relevant resume chunks for multiple search queries."""
    best_results = {}

    for query in queries:
        query_vector = resume_vectorizer.transform([query])
        scores = cosine_similarity(query_vector, resume_vectors).flatten()

        for index in scores.argsort()[::-1]:
            score = float(scores[index])
            if score <= 0.08:
                continue

            chunk = resume_chunks[index]
            existing = best_results.get(chunk["text"])
            if existing is None or score > existing["score"]:
                best_results[chunk["text"]] = {
                    "heading": chunk["heading"],
                    "content": chunk["content"],
                    "source": chunk["text"].splitlines()[0],
                "score": round(score, 4),
                }

    ranked = sorted(
        best_results.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranked[:top_k]


SKILL_ALIASES = {
    "mysql": "sql",
    "sql server": "sql",
    "spring boot": "spring",
    "rest": "api",
    "restful": "api",
    "llm": "大语言模型",
    "large language model": "大语言模型",
    "大模型": "大语言模型",
    "function calling": "函数调用",
    "tool calling": "工具调用",
    "tool use": "工具调用",
    "llm api": "大语言模型 api 接入",
    "大语言模型 api 调用": "大语言模型 api 接入",
    "大模型 api 调用": "大语言模型 api 接入",
    "llm api 调用": "大语言模型 api 接入",
}


def normalize_skill(skill: str) -> str:
    normalized = re.sub(r"[\s_\-/]+", " ", skill.strip().lower())
    return SKILL_ALIASES.get(normalized, normalized)


def match_skill_lists(skills: list[str], candidate: set[str]) -> dict:
    normalized_skills = [normalize_skill(skill) for skill in skills]
    matched = []
    missing = []

    for skill in normalized_skills:
        if any(
            skill == candidate_skill
            or skill in candidate_skill
            or candidate_skill in skill
            for candidate_skill in candidate
        ):
            matched.append(skill)
        else:
            missing.append(skill)

    percentage = (
        round(len(matched) / len(normalized_skills) * 100, 1)
        if normalized_skills
        else 0.0
    )

    return {
        "matched_skills": sorted(set(matched)),
        "missing_skills": sorted(set(missing)),
        "keyword_coverage": percentage,
    }


@tool
def calculate_skill_match(
    required_skills: list[str],
    preferred_skills: list[str],
) -> dict:
    """Calculate separate keyword coverage for hard requirements and preferred conditions."""
    candidate = {
        normalize_skill(record["name"])
        for record in candidate_skill_records
    }

    return {
        "hard_requirements": match_skill_lists(required_skills, candidate),
        "preferred_conditions": match_skill_lists(preferred_skills, candidate),
        "note": "这是关键词覆盖率，不是综合能力评分。",
    }


class JobMatchReport(BaseModel):
    job_title: str = Field(description="岗位名称")
    location: str = Field(description="岗位地点")
    current_date: str = Field(description="分析报告使用的当前日期")
    hard_requirement_coverage: float = Field(
        description="硬性要求关键词覆盖率，范围 0 到 100"
    )
    preferred_condition_coverage: float = Field(
        description="优先条件关键词覆盖率，范围 0 到 100"
    )
    matched_skills: list[str] = Field(description="有明确证据的已具备技能")
    missing_skills: list[str] = Field(description="没有明确证据的能力")
    evidence: list[str] = Field(description="关键要求的证据摘要及判断")
    risks: list[str] = Field(description="投递前需要确认或补强的风险")
    resume_suggestions: list[str] = Field(description="简历优化建议")
    next_actions: list[str] = Field(description="接下来可以执行的行动")
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

agent = create_deep_agent(
    model=model,
    tools=[
        get_job_description,
        get_candidate_skills,
        get_resume_evidence,
        calculate_skill_match,
    ],
    system_prompt=(
        "你是 JobPilot，一名严谨的求职岗位匹配助手。\n"
        "用户长期偏好如下：\n"
        f"{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
        f"当前日期：{CURRENT_DATE}\n\n"
        "工作规则：\n"
        "1. 必须先调用 get_job_description。\n"
        "2. 必须调用 get_candidate_skills，不能自己编造候选人技能。\n"
        "3. 必须只调用一次 get_resume_evidence，并一次传入覆盖主要要求的查询列表。\n"
        "4. 必须调用一次 calculate_skill_match，分别传入 required_skills 和 preferred_skills。\n"
        "5. required_skills 对应 JD 的任职要求，preferred_skills 对应优先条件，二者不能混在一起。\n"
        "6. 只写原子技术能力，不要把整句职责当成技能。\n"
        "7. 关键词覆盖率不等于真实能力，不得把它作为面试通过率。\n"
        "8. 简历没有证据的能力不能写成已经具备。\n"
        "9. 所有优势判断必须引用 skills.json 或检索到的 Chunk。\n"
        "10. 判断项目时间是否合理时必须参考当前日期。\n\n"
        "最终输出 Markdown 报告，必须包含：\n"
        "# 岗位匹配报告\n"
        "## 硬性要求覆盖情况\n"
        "## 优先条件覆盖情况\n"
        "## 已具备能力与证据\n"
        "## 能力缺口\n"
        "## 简历优化建议\n"
        "## 投递前的行动清单"
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "请分析我当前简历与这个 AI 全栈应用实习岗位的匹配程度。"
                    "读取 JD 和候选人技能，检索简历证据，"
                    "分别计算硬性要求与优先条件的关键词覆盖率，最后生成报告。"
                ),
            }
        ]
    }
)

for message in result["messages"]:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"工具调用：{message.tool_calls}")

analysis_report = result["messages"][-1].content
REPORT_MD_PATH.write_text(analysis_report, encoding="utf-8")

formatter_agent = create_deep_agent(
    model=model,
    response_format=JobMatchReport,
    system_prompt=(
        "你是结构化数据转换器。"
        "只把用户提供的岗位匹配报告转换成 JobMatchReport 字段。"
        "不得新增报告中没有的事实，不得修改覆盖率数字。"
        "没有明确依据的内容要保留为风险或缺口。"
    ),
)

formatter_result = formatter_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "请把下面的岗位匹配报告转换为结构化结果：\n\n"
                    f"{analysis_report}"
                ),
            }
        ]
    },
    config={"recursion_limit": 20},
)

structured = formatter_result.get("structured_response")
if structured is None:
    print("格式化阶段没有获得 structured_response。最后几条消息如下：")
    for message in formatter_result["messages"][-3:]:
        print(type(message).__name__, message.content)
    raise RuntimeError("格式化阶段没有获得 structured_response")

report_json = structured.model_dump_json(indent=2)
REPORT_JSON_PATH.write_text(report_json, encoding="utf-8")

print(report_json)
print(f"\nMarkdown 分析已保存到：{REPORT_MD_PATH}")
print(f"结构化报告已保存到：{REPORT_JSON_PATH}")