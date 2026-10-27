"""第十课：兼容 Chat Completions 的模型 → JSON 校验 → LangGraph 路由。

离线校验：python -B -X utf8 structured_output_full.py --check
真实调用：python -B -X utf8 structured_output_full.py --live
默认优先读取脚本旁的 key.env；未填写的字段才使用环境变量。
--model、--base-url 和 --prompt-key 可显式覆盖配置。
"""

import argparse
from collections.abc import Callable
import getpass
import json
import os
from pathlib import Path
import re
import sys
from typing import Literal, Self
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph


DEFAULT_MODEL = "deepseek-flash"  # 2026-10-23 DeepSeek 官方 JSON Output 示例使用的模型。
DEFAULT_BASE_URL = "https://api.deepseek.com"
ENDPOINT = DEFAULT_BASE_URL + "/chat/completions"


def load_local_config(path: Path) -> dict[str, str]:
    """读取本课的本地配置（key=value 或 key: value），不执行文件内容。"""
    if not path.exists():
        return {}
    aliases = {"model": "model", "model_name": "model", "deepseek_model": "model",
               "base_url": "base_url", "base-url": "base_url", "deepseek_base_url": "base_url",
               "api-key": "api_key", "api_key": "api_key", "deepseek_api_key": "api_key"}
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([\w-]+)\s*[:=]\s*(.*)", line)
        if not match or match[1].lower() not in aliases:
            raise ValueError(f"本地配置第 {number} 行格式或字段不受支持（内容已隐藏）")
        value = match[2].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        name = aliases[match[1].lower()]
        if name in result:
            raise ValueError(f"本地配置第 {number} 行字段重复（内容已隐藏）")
        result[name] = value.strip()
    return result


class LearningRequest(BaseModel):
    """用于路由的学习请求，不是最终问题答案。"""

    model_config = ConfigDict(strict=True, extra="forbid")
    topic: str = Field(min_length=1, max_length=80, description="用户要学习的主题")
    output_format: Literal["explanation", "code"]
    needs_clarification: bool
    clarifying_question: str = Field(max_length=160)

    @model_validator(mode="after")
    def check_question(self) -> Self:
        if not self.topic.strip():
            raise ValueError("topic 不能只包含空白")
        if self.needs_clarification and not self.clarifying_question.strip():
            raise ValueError("需要澄清时必须提供具体问题")
        if not self.needs_clarification and self.clarifying_question.strip():
            raise ValueError("无需澄清时 clarifying_question 必须为空")
        return self


class RequestState(TypedDict):
    question: str
    raw_json: str
    decision: dict
    validation_errors: list[str]
    status: str
    output: str


class ModelResponseError(Exception):
    """模型服务响应不符合本示例需要的协议。"""


def system_prompt() -> str:
    example = {
        "topic": "LangGraph 条件边",
        "output_format": "code",
        "needs_clarification": False,
        "clarifying_question": "",
    }
    return (
        "你负责分类 LangGraph 学习请求，不回答知识问题。只输出一个 JSON 对象。"
        "用户明确需要代码则 output_format=code，否则为 explanation。"
        "只有缺少关键上下文而无法确定学习主题时才需要澄清；不要猜测缺失信息。"
        "needs_clarification 为 true 时必须提供具体澄清问题，否则该问题字段必须为空。"
        "JSON 示例：" + json.dumps(example, ensure_ascii=False)
        + "\n必须符合以下 JSON Schema："
        + json.dumps(LearningRequest.model_json_schema(), ensure_ascii=False)
    )


def chat_generator(api_key: str, model: str, transport=None,
                       base_url: str = DEFAULT_BASE_URL) -> Callable[[str], str]:
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def generate(question: str) -> str:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": question},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 1024,
        }
        # 第三方平台先使用通用字段，不假定其支持 DeepSeek 官方的扩展参数。
        if urlsplit(base_url).hostname == "api.deepseek.com":
            body["thinking"] = {"type": "disabled"}
        with httpx.Client(timeout=30.0, transport=transport) as client:
            response = client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
            response.raise_for_status()
        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError("响应缺少有效的 choices/message/content") from exc
        if finish_reason != "stop":
            raise ModelResponseError(f"模型未正常结束：finish_reason={finish_reason!r}")
        if not isinstance(content, str):
            raise ModelResponseError("模型正文不是字符串")
        return content

    return generate


def build_graph(generate_json: Callable[[str], str]):
    def classify(state: RequestState):
        raw = generate_json(state["question"])
        try:
            decision = LearningRequest.model_validate_json(raw)
        except ValidationError as exc:
            errors = [
                f"{'.'.join(map(str, item['loc'])) or 'schema'}: {item['msg']}"
                for item in exc.errors(include_input=False, include_url=False)
            ]
            return {"raw_json": raw, "decision": {}, "validation_errors": errors,
                    "status": "invalid_output"}
        return {"raw_json": raw, "decision": decision.model_dump(),
                "validation_errors": [], "status": "classified"}

    def route(state: RequestState) -> Literal["clarify", "code", "explain", "invalid"]:
        if state["status"] == "invalid_output":
            return "invalid"
        if state["decision"]["needs_clarification"]:
            return "clarify"
        return "code" if state["decision"]["output_format"] == "code" else "explain"

    def clarify(state: RequestState):
        return {"status": "needs_clarification", "output": state["decision"]["clarifying_question"]}

    def code(state: RequestState):
        return {"status": "code_selected", "output": f"进入代码示例分支，主题：{state['decision']['topic']}"}

    def explain(state: RequestState):
        return {"status": "explanation_selected", "output": f"进入概念讲解分支，主题：{state['decision']['topic']}"}

    def invalid(state: RequestState):
        return {"output": "模型输出未通过校验，本次不执行正常业务分支。"}

    builder = StateGraph(RequestState)
    for name, node in [("classify", classify), ("clarify", clarify), ("code", code),
                       ("explain", explain), ("invalid", invalid)]:
        builder.add_node(name, node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route)
    for name in ("clarify", "code", "explain", "invalid"):
        builder.add_edge(name, END)
    return builder.compile()


def run_once(generate_json, question: str):
    result = build_graph(generate_json).invoke({
        "question": question, "raw_json": "", "decision": {},
        "validation_errors": [], "status": "new", "output": "",
    })
    print("解析结果：", result["decision"])
    if result["validation_errors"]:
        print("校验错误：", result["validation_errors"])
    print("最终状态：", result["status"])
    print("输出：", result["output"])
    return result


def offline_check():
    print("离线校验：以下数据均为预设样本，没有调用真实模型。")
    base = {"topic": "LangGraph State", "output_format": "explanation",
            "needs_clarification": False, "clarifying_question": ""}
    cases = [
        ("概念请求", base, "explanation_selected"),
        ("代码请求", {**base, "output_format": "code"}, "code_selected"),
        ("澄清请求", {**base, "needs_clarification": True,
                      "clarifying_question": "你想了解哪一个节点的行为？"}, "needs_clarification"),
        ("布尔值错误地写成字符串", {**base, "needs_clarification": "false"}, "invalid_output"),
        ("需要澄清却没有问题", {**base, "needs_clarification": True}, "invalid_output"),
        ("JSON 被截断", '{"topic":', "invalid_output"),
    ]
    for label, payload, expected in cases:
        print(f"\n=== {label} ===")
        raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        result = run_once(lambda question, raw=raw: raw, "离线学习请求")
        assert result["status"] == expected, label

    def mock_handler(request):
        body = json.loads(request.content)
        assert str(request.url) == ENDPOINT
        assert body["response_format"] == {"type": "json_object"}
        assert body["thinking"] == {"type": "disabled"}
        assert body["messages"][-1]["content"] == "模拟接口请求"
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(base)},
        }]})

    print("\n=== 模拟 HTTP：验证真实请求函数的构造与响应解析，无外部联网 ===")
    generate = chat_generator("offline-placeholder", DEFAULT_MODEL,
                                  transport=httpx.MockTransport(mock_handler))
    assert run_once(generate, "模拟接口请求")["status"] == "explanation_selected"

    def compatible_handler(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://example.invalid/v1/chat/completions"
        assert "thinking" not in body
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(base)},
        }]})

    print("\n=== 模拟第三方兼容接口：验证 /v1 路径和通用参数 ===")
    generate = chat_generator("offline-placeholder", "glm-5.3",
                                  transport=httpx.MockTransport(compatible_handler),
                                  base_url="https://example.invalid/v1/")
    assert run_once(generate, "模拟接口请求")["status"] == "explanation_selected"
    print("\n离线校验通过：6 个 schema/路由场景，2 个模拟 HTTP 场景。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="只运行离线校验")
    modes.add_argument("--live", action="store_true", help="真实调用所配置的平台模型一次")
    parser.add_argument("--prompt-key", action="store_true", help="在自己的交互式终端中隐藏输入 API Key")
    parser.add_argument("--model", help="覆盖环境变量和 key.env 中的模型名称")
    parser.add_argument("--base-url", help="模型平台的兼容 API 基础地址，可包含 /v1")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name("key.env"),
                        help="本地配置路径，默认是脚本旁的 key.env")
    parser.add_argument("--question", default="请给我一个 LangGraph 条件边的最小代码示例。")
    args = parser.parse_args()
    if args.check:
        offline_check()
        return
    if not args.question.strip():
        parser.error("question 不能为空")
    try:
        local_config = load_local_config(args.env_file)
    except (OSError, UnicodeError, ValueError):
        parser.error("无法读取本地配置，请检查文件编码、字段和格式（内容已隐藏）")
    model = (args.model or local_config.get("model") or os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]*", model):
        parser.error("请使用平台提供的有效模型标识，不要包含空白")
    base_url = (args.base_url or local_config.get("base_url")
                or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    url = urlsplit(base_url)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        parser.error("base_url 必须是无凭证、查询参数和片段的 HTTPS 基础地址")
    file_key = local_config.get("api_key", "").strip()
    api_key = file_key or os.getenv("DEEPSEEK_API_KEY", "").strip()
    key_source = f"配置文件 {args.env_file.resolve()}" if file_key else "环境变量 DEEPSEEK_API_KEY"
    if args.prompt_key:
        if not sys.stdin.isatty():
            parser.error("隐藏输入需要在你自己的交互式终端运行；也可使用本机环境变量")
        api_key = getpass.getpass("DeepSeek API Key（隐藏输入，不保存到文件）：").strip()
        key_source = "本次隐藏输入（--prompt-key）"
    if not api_key:
        parser.error("未配置 API Key；请配置 key.env、DEEPSEEK_API_KEY 或使用 --live --prompt-key")
    print(f"真实模型调用：{model}；只发送本次 question 与本课分类提示词。")
    print(f"接口平台：{url.hostname}")
    print(f"密钥来源：{key_source}（密钥内容不显示）")
    try:
        result = run_once(chat_generator(api_key, model, base_url=base_url), args.question)
        if result["status"] == "invalid_output":
            raise SystemExit("真实请求已完成，但模型输出未通过校验。")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise SystemExit("模型服务返回 HTTP 401：平台未接受本次密钥。请核对上方密钥来源；"
                             "key.env 优先于环境变量，--prompt-key 则使用本次手工输入。") from None
        if exc.response.status_code == 429:
            raise SystemExit("模型服务返回 HTTP 429：本次请求受限，需检查具体模型通道与账户限额。") from None
        raise SystemExit(f"模型服务返回 HTTP {exc.response.status_code}；请检查模型、凭证和服务状态。") from None
    except httpx.RequestError as exc:
        raise SystemExit(f"模型请求失败：{type(exc).__name__}。") from None
    except ModelResponseError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
