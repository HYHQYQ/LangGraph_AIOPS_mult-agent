"""第九课：Command(update, goto) 与返回字典加条件边的对照。

运行：python -B -X utf8 command_routing_demo.py
使用本地模拟评分，包含一组静态边仍会执行的对照实验。
"""

from operator import add
from typing import Annotated, Literal

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class ReviewState(TypedDict):
    score: int
    approved: bool
    status: str
    events: Annotated[list[str], add]


def review_update(state: ReviewState):
    return {
        "approved": state["score"] >= 80,
        "status": "reviewed",
        "events": ["review"],
    }


def review_with_dict(state: ReviewState):
    print("[review:dict] 返回状态更新，下一步由条件边决定")
    return review_update(state)


def route_after_review(state: ReviewState) -> Literal["accept", "revise"]:
    return "accept" if state["approved"] else "revise"


def review_with_command(state: ReviewState) -> Command[Literal["accept", "revise"]]:
    update = review_update(state)
    destination = "accept" if update["approved"] else "revise"
    print(f"[review:Command] 同时提交状态更新与 goto={destination}")
    return Command(update=update, goto=destination)


def accept(state: ReviewState):
    print(f"[accept] 读到 approved={state['approved']}，status={state['status']}")
    return {"status": "accepted", "events": ["accept"]}


def revise(state: ReviewState):
    print(f"[revise] 读到 approved={state['approved']}，status={state['status']}")
    return {"status": "needs_revision", "events": ["revise"]}


def static_notice(state: ReviewState):
    print("[static_notice] review 的普通边仍然调度了这个节点")
    # 只更新可合并的事件字段，避免与另一分支同时覆盖 status。
    return {"events": ["static_notice"]}


def build_graph(use_command: bool, extra_static_edge: bool = False):
    builder = StateGraph(ReviewState)
    builder.add_node("review", review_with_command if use_command else review_with_dict)
    builder.add_node("accept", accept)
    builder.add_node("revise", revise)
    builder.add_edge(START, "review")
    if not use_command:
        builder.add_conditional_edges("review", route_after_review)
    # Command 版本由 review 返回的 goto 决定下一步，无须再添加此处的条件边。
    builder.add_edge("accept", END)
    builder.add_edge("revise", END)

    if extra_static_edge:
        builder.add_node("static_notice", static_notice)
        builder.add_edge("review", "static_notice")
        builder.add_edge("static_notice", END)
    return builder.compile()


def run_demo(score: int, use_command: bool, extra_static_edge: bool = False):
    style = "Command" if use_command else "字典 + 条件边"
    suffix = "，另有 review → static_notice 普通边" if extra_static_edge else ""
    print(f"\n=== score={score}，{style}{suffix} ===")
    result = build_graph(use_command, extra_static_edge).invoke({
        "score": score,
        "approved": False,
        "status": "new",
        "events": [],
    })
    print("最终状态：", result["status"])
    print("事件记录：", result["events"])
    return result


if __name__ == "__main__":
    for score in (90, 50):
        run_demo(score, use_command=False)
        run_demo(score, use_command=True)
    run_demo(90, use_command=True, extra_static_edge=True)
