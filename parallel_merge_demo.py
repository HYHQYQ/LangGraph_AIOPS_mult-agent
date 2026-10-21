"""第五课：并发分支、等待汇总，以及同一字段的并发更新。

运行：python -X utf8 parallel_merge_demo.py
time.sleep 仅模拟 I/O 等待；统计计算真实执行，不调用外部服务。
"""

import time
from operator import add
from typing import Annotated

from typing_extensions import TypedDict
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph


class Finding(TypedDict):
    metric: str
    value: int


class BaseState(TypedDict):
    numbers: list[int]
    report: str


class ParallelState(BaseState):
    findings: Annotated[list[Finding], add]


class ConflictState(BaseState):
    findings: list[Finding]  # 故意不配置 reducer，演示并发写入冲突。


def sum_node(state):
    print(f"[sum] 开始，已有结果条数={len(state['findings'])}")
    time.sleep(0.15)
    print(f"[sum] 完成，节点读到的已有结果条数仍为={len(state['findings'])}")
    return {"findings": [{"metric": "sum", "value": sum(state["numbers"])}]}


def maximum_node(state):
    print(f"[maximum] 开始，已有结果条数={len(state['findings'])}")
    time.sleep(0.30)
    print(f"[maximum] 完成，节点读到的已有结果条数仍为={len(state['findings'])}")
    return {"findings": [{"metric": "maximum", "value": max(state["numbers"])}]}


def summarize_node(state):
    print(f"[summarize] 开始，已收到 {len(state['findings'])} 条结果")
    # 按 metric 取值，避免依赖并行分支结果在列表中的位置。
    values = {item["metric"]: item["value"] for item in state["findings"]}
    return {"report": f"总和={values['sum']}；最大值={values['maximum']}"}


def build_graph(state_schema=ParallelState):
    builder = StateGraph(state_schema)
    builder.add_node("sum", sum_node)
    builder.add_node("maximum", maximum_node)
    builder.add_node("summarize", summarize_node)

    # fan-out：两个分支都执行，而不是二选一。
    builder.add_edge(START, "sum")
    builder.add_edge(START, "maximum")
    # fan-in：明确要求两个来源都完成，才调度汇总节点。
    builder.add_edge(["sum", "maximum"], "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


def main():
    initial = {"numbers": [2, 3, 5], "findings": [], "report": ""}

    print("=== 1. 有 reducer：并行计算并汇总 ===")
    result = build_graph().invoke(initial)
    print("合并后的结果：", result["findings"])
    print("最终报告：", result["report"])

    print("\n=== 2. 无 reducer：观察同一字段的并发更新冲突 ===")
    try:
        build_graph(ConflictState).invoke(initial)
    except InvalidUpdateError as exc:
        # 本实验预期出现的错误。这里只捕获该类型，不掩盖其他运行错误。
        print("捕获预期错误：", type(exc).__name__)
        print(str(exc).splitlines()[0])


if __name__ == "__main__":
    main()
