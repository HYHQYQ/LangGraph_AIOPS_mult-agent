"""第十课精简版：直接运行本文件，会调用一次真实模型。"""

from pathlib import Path
from typing import Literal
import sys

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

# 改这一行，就能尝试不同的用户请求。
QUESTION = "请给我一个 LangGraph 条件边的最小代码示例。"


# 1. 模型应该返回的数据格式。
class LearningRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    topic: str = Field(min_length=1, max_length=80)
    output_format: Literal["explanation", "code"]
    needs_clarification: bool
    clarifying_question: str = Field(max_length=160)


# 2. 图中的节点共同使用这份 State。
class RequestState(TypedDict):
    question: str
    decision: dict
    status: str
    output: str


SYSTEM_PROMPT = """你负责分类 LangGraph 学习请求，只输出 JSON，不回答知识问题。
用户要求代码时 output_format 为 code，否则为 explanation。
只有缺少关键信息、无法确定主题时才需要澄清，并填写具体的 clarifying_question。
无需澄清时 clarifying_question 必须为空字符串，topic 必须是非空主题。
下面仅为格式示例；字段值要根据本次用户问题填写：
{"topic":"LangGraph 条件边","output_format":"code","needs_clarification":false,"clarifying_question":""}
"""


# 3. 只读取脚本旁的 key.env，使用 model、api-key、base_url 三个字段。
def load_config():
    config = {}
    path = Path(__file__).with_name("key.env")
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        separator = "=" if "=" in line else ":"
        name, found, value = line.partition(separator)
        if not found or name.strip() not in ("model", "api-key", "base_url"):
            raise ValueError("key.env 只支持 model、api-key、base_url 三个字段")
        config[name.strip()] = value.strip().strip("\"'")
    for name in ("model", "api-key", "base_url"):
        if not config.get(name):
            raise ValueError("key.env 缺少非空的 model、api-key 或 base_url")
    return config


# 4. 普通函数：传入问题，直接返回模型生成的 JSON 文本。
def ask_model(question):
    config = load_config()
    response = httpx.post(
        config["base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + config["api-key"]},
        json={
            "model": config["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 1024,
        },
        timeout=60,
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("模型输出未正常结束")
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("模型正文必须是字符串")
    return content


# 5. classify 节点：请求模型、检查数据，然后更新 State。
def classify(state):
    raw = ask_model(state["question"])
    try:
        decision = LearningRequest.model_validate_json(raw)
        if not decision.topic.strip():
            raise ValueError("主题不能为空白")
        if decision.needs_clarification and not decision.clarifying_question.strip():
            raise ValueError("需要澄清时必须提供问题")
        if not decision.needs_clarification and decision.clarifying_question.strip():
            raise ValueError("无需澄清时问题必须为空")
    except (ValidationError, ValueError):
        return {"decision": {}, "status": "invalid_output"}
    return {"decision": decision.model_dump(), "status": "classified"}


# 6. 条件边：返回下一站的节点名称。
def route(state):
    if state["status"] == "invalid_output":
        return "invalid"
    if state["decision"]["needs_clarification"]:
        return "clarify"
    if state["decision"]["output_format"] == "code":
        return "code"
    return "explain"


# 7. 四个分支：当前只输出提示，便于观察实际选中了哪一条路。
def code(state):
    return {"status": "code_selected", "output": "进入代码示例分支，主题：" + state["decision"]["topic"]}


def explain(state):
    return {"status": "explanation_selected", "output": "进入概念讲解分支，主题：" + state["decision"]["topic"]}


def clarify(state):
    return {"status": "needs_clarification", "output": state["decision"]["clarifying_question"]}


def invalid(state):
    return {"status": "invalid_output", "output": "模型返回的数据未通过校验。"}


# 8. 建图：把每个节点、执行顺序和分支关系写清楚。
builder = StateGraph(RequestState)
builder.add_node("classify", classify)
builder.add_node("code", code)
builder.add_node("explain", explain)
builder.add_node("clarify", clarify)
builder.add_node("invalid", invalid)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, {
    "code": "code", "explain": "explain", "clarify": "clarify", "invalid": "invalid",
})
builder.add_edge("code", END)
builder.add_edge("explain", END)
builder.add_edge("clarify", END)
builder.add_edge("invalid", END)
graph = builder.compile()


# 9. 从这里开始运行；直接点击运行即可，无需 --live 参数。
if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("精简版无需参数，请清空脚本参数。原版选项请使用 structured_output_full.py。")
    try:
        result = graph.invoke({"question": QUESTION, "decision": {}, "status": "new", "output": ""})
        print("解析结果：", result["decision"])
        print("最终状态：", result["status"])
        print("输出：", result["output"])
    except httpx.HTTPStatusError as exc:
        raise SystemExit(f"模型平台返回 HTTP {exc.response.status_code}，请检查 key.env 或平台状态。") from None
    except httpx.RequestError as exc:
        raise SystemExit(f"网络请求失败：{type(exc).__name__}") from None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        raise SystemExit("配置或模型响应格式不正确，请检查 key.env 的三个字段及接口响应。") from None
