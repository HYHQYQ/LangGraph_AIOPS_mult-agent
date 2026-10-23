"""第八课：Send 动态派发工作，并用 reducer 汇总结果。

运行：python -B -X utf8 send_map_reduce_demo.py
对文本按空白分词计数，全部在本地完成，不需要模型服务。
"""

from operator import add
from typing import Annotated, Literal

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class TextResult(TypedDict):
    index: int
    text: str
    word_count: int


class BatchState(TypedDict):
    texts: list[str]
    results: Annotated[list[TextResult], add]
    report: str


class WorkerInput(TypedDict):
    index: int
    text: str


def dispatch(state: BatchState) -> list[Send] | Literal["summarize"]:
    if not state["texts"]:
        print("[dispatch] 输入为空，直接进入汇总")
        return "summarize"

    print(f"[dispatch] 为 {len(state['texts'])} 段文本创建任务")
    return [
        Send("worker", {"index": index, "text": text})
        for index, text in enumerate(state["texts"])
    ]


def worker(state: WorkerInput):
    count = len(state["text"].split())
    print(f"[worker #{state['index']}] 输入字段={sorted(state)}，词数={count}")
    # 一个任务只返回自己的结果；由 BatchState.results 的 reducer 合并。
    return {"results": [{
        "index": state["index"],
        "text": state["text"],
        "word_count": count,
    }]}


def summarize(state: BatchState):
    ordered = sorted(state["results"], key=lambda item: item["index"])
    total = sum(item["word_count"] for item in ordered)
    print(f"[summarize] 收到 {len(ordered)} 条结果，生成一次报告")
    # 只更新 report，不把已累积的完整列表再次提交给 add。
    return {"report": f"已处理 {len(ordered)} 段文本，总词数={total}"}


def build_graph():
    builder = StateGraph(BatchState)
    # worker 只注册一次，运行时可以有多个任务调用同一个节点。
    builder.add_node("worker", worker)
    builder.add_node("summarize", summarize)
    builder.add_conditional_edges(START, dispatch, ["worker", "summarize"])
    builder.add_edge("worker", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


def run_demo(graph, name: str, texts: list[str]):
    print(f"\n=== {name} ===")
    result = graph.invoke({"texts": texts, "results": [], "report": ""})
    for item in sorted(result["results"], key=lambda item: item["index"]):
        print(f"结果 #{item['index']}：{item['text']!r} → {item['word_count']} 词")
    print("报告：", result["report"])
    return result


if __name__ == "__main__":
    # 复用同一张图；没有 Checkpointer，每次输入均从新的批次状态开始。
    graph = build_graph()
    run_demo(graph, "三段文本", [
        "Learn LangGraph",
        "State nodes edges",
        "Send dispatches work",
    ])
    run_demo(graph, "两段相同文本仍是两个任务", ["same text", "same text"])
    run_demo(graph, "空批次", [])
