"""第一课：State 部分更新、reducer、条件路由与业务退出条件。

运行：python -X utf8 state_routing_demo.py
搜索和评分均为模拟，不需要 API Key。
"""

from operator import add
from typing import Annotated, Literal

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph


class ResearchState(TypedDict):
    question: str
    search_count: int
    max_searches: int
    mock_scores: list[int]
    search_result: str  # 默认替换：只保留最近一次搜索结果。
    search_history: Annotated[list[str], add]  # 列表相加：累积各轮结果。
    evidence_score: int
    stage: str
    status: str
    report: str


def search_node(state: ResearchState):
    count = state["search_count"] + 1
    result = f"第 {count} 轮搜索结果"
    # 返回本轮增量，不修改输入，也不返回完整的历史。
    return {
        "search_count": count,
        "search_result": result,
        "search_history": [result],
        "stage": "search",
    }


def review_node(state: ResearchState):
    # 每轮读取一个预设评分；预设用完后继续使用最后一个评分。
    index = min(state["search_count"] - 1, len(state["mock_scores"]) - 1)
    return {
        "evidence_score": state["mock_scores"][index],
        "stage": "review",
    }


def route_after_review(state: ResearchState) -> Literal["enough", "retry", "exhausted"]:
    # 最后一轮恰好合格时仍应成功，所以先判断评分，再判断次数。
    if state["evidence_score"] >= 80:
        return "enough"
    if state["search_count"] >= state["max_searches"]:
        return "exhausted"
    return "retry"


def report_node(state: ResearchState):
    return {
        "report": f"问题：{state['question']}；模拟证据合格，可以生成报告。",
        "status": "completed",
        "stage": "report",
    }


def fallback_node(state: ResearchState):
    return {
        "report": f"已搜索 {state['search_count']} 次，证据仍不足，需要补充信息。",
        "status": "insufficient_evidence",
        "stage": "fallback",
    }


def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("search", search_node)
    builder.add_node("review", review_node)
    builder.add_node("report", report_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "search")
    builder.add_edge("search", "review")
    # 路由返回的是标签，通过映射决定真正的目标节点。
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"enough": "report", "retry": "search", "exhausted": "fallback"},
    )
    builder.add_edge("report", END)
    builder.add_edge("fallback", END)
    return builder.compile()


def initial_state(scores: list[int], max_searches: int) -> ResearchState:
    # TypedDict 是类型提示；这里显式检查演示数据的业务约束。
    if max_searches < 1:
        raise ValueError("max_searches 必须至少为 1")
    if not scores or any(score < 0 or score > 100 for score in scores):
        raise ValueError("scores 必须包含至少一个 0 至 100 的评分")
    return {
        "question": "为什么销售额下降？",
        "search_count": 0,
        "max_searches": max_searches,
        "mock_scores": list(scores),
        "search_result": "",
        "search_history": [],
        "evidence_score": 0,
        "stage": "input",
        "status": "running",
        "report": "",
    }


def run_demo(name: str, scores: list[int], max_searches: int):
    print(f"\n=== {name} ===")
    graph = build_graph()
    state = initial_state(scores, max_searches)
    # 本图每轮 search、review 各占一步，结束节点再占一步。
    # 业务次数上限负责正常退出；引擎步数限制作为额外保护。
    config = {"recursion_limit": 2 * max_searches + 2}
    for snapshot in graph.stream(state, config, stream_mode="values"):
        print(
            f"{snapshot['stage']:8} | 次数={snapshot['search_count']}"
            f" | 评分={snapshot['evidence_score']}"
            f" | 历史条数={len(snapshot['search_history'])}"
            f" | 状态={snapshot['status']}"
        )
        state = snapshot
    print("最近一次结果：", state["search_result"])
    print("累积搜索历史：", state["search_history"])
    print("最终报告：", state["report"])
    return state


if __name__ == "__main__":
    run_demo("场景一：第二轮合格", [50, 90], max_searches=3)
    run_demo("场景二：达到上限仍不合格", [40, 60, 70], max_searches=3)
    run_demo("场景三：最后一轮恰好合格", [50, 60, 80], max_searches=3)
