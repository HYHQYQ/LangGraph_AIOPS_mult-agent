"""第三课：LangGraph 中断、恢复，以及节点重执行。

运行：python -X utf8 interrupt_demo.py
所有操作均为本地模拟，程序分别传入 True/False 演示两条路径。
"""

from typing import Literal

from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ReviewState(TypedDict):
    topic: str
    draft: str
    approved: bool
    status: str
    result: str


def prepare_node(state: ReviewState):
    print("[prepare] 生成草稿")
    return {"draft": f"关于{state['topic']}的学习摘要", "status": "draft_ready"}


def review_node(state: ReviewState):
    # 观察：初次运行与恢复时，这行日志都会打印。
    print("[review] 从节点开头执行，即将到达 interrupt")
    approved = interrupt({
        "question": "是否采纳这份草稿？请传入布尔值。",
        "draft": state["draft"],
    })
    # 恢复值成为 interrupt() 的返回值，而不是直接更新 State。
    if type(approved) is not bool:
        raise ValueError("恢复值必须是布尔值 True 或 False")
    print(f"[review] interrupt 已返回：{approved}")
    return {"approved": approved, "status": "reviewed"}


def route_after_review(state: ReviewState) -> Literal["accept", "reject"]:
    return "accept" if state["approved"] else "reject"


def accept_node(state: ReviewState):
    print("[accept] 采纳草稿（模拟）")
    return {"status": "accepted", "result": state["draft"]}


def reject_node(state: ReviewState):
    print("[reject] 不采纳草稿（模拟）")
    return {"status": "rejected", "result": "草稿未采纳，需要修改。"}


def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("review", review_node)
    builder.add_node("accept", accept_node)
    builder.add_node("reject", reject_node)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "review")
    builder.add_conditional_edges("review", route_after_review)
    builder.add_edge("accept", END)
    builder.add_edge("reject", END)
    return builder.compile(checkpointer=InMemorySaver())


def run_demo(approved: bool):
    print(f"\n=== 恢复值为 {approved} 的场景 ===")
    graph = build_graph()
    config = {"configurable": {"thread_id": f"review-{approved}"}}

    print("\n1. 开始执行，运行到中断处")
    paused = graph.invoke({
        "topic": "LangGraph 中断与恢复",
        "draft": "",
        "approved": False,
        "status": "new",
        "result": "",
    }, config)
    print("已返回调用方，中断载荷：", paused["__interrupt__"][0].value)
    snapshot = graph.get_state(config)
    print("保存的状态：", snapshot.values)
    print("待继续的节点：", snapshot.next)

    print("\n2. 使用同一 thread_id 传入恢复值")
    result = graph.invoke(Command(resume=approved), config)
    print("最终状态：", result["status"])
    print("最终结果：", result["result"])
    print("待执行节点：", graph.get_state(config).next)
    return result


if __name__ == "__main__":
    run_demo(True)
    run_demo(False)
