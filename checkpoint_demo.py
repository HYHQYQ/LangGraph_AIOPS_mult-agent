"""LangGraph 第二课：用计数器观察 Checkpointer 与 thread_id。

运行：python -X utf8 checkpoint_demo.py
"""

from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    increment: int
    total: int


def accumulate(state: CounterState):
    return {"total": state.get("total", 0) + state["increment"]}


def build_graph(checkpointer=None):
    builder = StateGraph(CounterState)
    builder.add_node("accumulate", accumulate)
    builder.add_edge(START, "accumulate")
    builder.add_edge("accumulate", END)
    return builder.compile(checkpointer=checkpointer)


def main():
    config_a = {"configurable": {"thread_id": "A"}}
    config_b = {"configurable": {"thread_id": "B"}}

    print("=== 1. 没有 Checkpointer：不同调用不自动延续状态 ===")
    plain = build_graph()
    print("第一次 +2：", plain.invoke({"increment": 2}, config_a)["total"])
    print("第二次 +3：", plain.invoke({"increment": 3}, config_a)["total"])

    print("\n=== 2. 同一 Checkpointer、同一 thread：延续状态 ===")
    saver = InMemorySaver()
    graph = build_graph(saver)
    print("A 第一次 +2：", graph.invoke({"increment": 2}, config_a)["total"])
    print("A 第二次 +3：", graph.invoke({"increment": 3}, config_a)["total"])

    print("\n=== 3. 不同 thread：使用各自的状态 ===")
    print("B 第一次 +3：", graph.invoke({"increment": 3}, config_b)["total"])
    snapshot = graph.get_state(config_a)
    print("读取 A 的最新状态：", snapshot.values)
    print("A 下一步待执行节点：", snapshot.next)

    print("\n=== 4. 新建内存 Checkpointer：旧状态不在其中 ===")
    fresh_graph = build_graph(InMemorySaver())
    print("新存储中的 A +1：", fresh_graph.invoke({"increment": 1}, config_a)["total"])


if __name__ == "__main__":
    main()
