from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    question: str
    search_result: str
    evidence_score: int
    report: str
    search_count: int


def search_node(state: AgentState):

    new_count = state["search_count"] + 1

    print(f"正在进行第 {new_count} 轮搜索")

    return {
        "search_result": f"第 {new_count} 轮搜索结果",
        "search_count": new_count
    }


def review_node(state: AgentState):

    print("正在审核证据")

    if state["search_count"] >= 2:
        score = 90
    else:
        score = 50

    return {
        "evidence_score": score
    }


def route_after_review(state: AgentState):

    if state["evidence_score"] >= 80:
        print("证据充足，生成报告")
        return "report"

    print("证据不足，重新搜索")
    return "search"


def report_node(state: AgentState):

    return {
        "report": f"""
问题：{state['question']}
搜索次数：{state['search_count']}
证据评分：{state['evidence_score']}
最终分析完成
"""
    }


builder = StateGraph(AgentState)

builder.add_node("search", search_node)
builder.add_node("review", review_node)
builder.add_node("report", report_node)

builder.add_edge(START, "search")

builder.add_edge(
    "search",
    "review"
)

builder.add_conditional_edges(
    "review",
    route_after_review
)

builder.add_edge(
    "report",
    END
)

graph = builder.compile()


result = graph.invoke({
    "question": "为什么销售额下降？",
    "search_result": "",
    "evidence_score": 0,
    "report": "",
    "search_count": 0
})

print(result["report"])



