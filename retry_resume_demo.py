"""第七课：节点自动重试，以及失败后的 checkpoint 恢复。

运行：python -B -X utf8 retry_resume_demo.py
全部使用本地模拟，不依赖模型、网络或数据库。
"""

from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class TemporaryReadError(Exception):
    """本实验中允许重试的临时读取异常。"""


class JobState(TypedDict):
    job_id: str
    payload: str
    status: str


def build_graph(failures_before_success: int):
    # 演示用的进程内计数：模拟外部数据源前几次失败，并观察节点调用次数。
    # 这些计数不属于图 State，不会被 checkpointer 保存。
    counts = {"prepare": 0, "fetch": 0, "finish": 0}

    def prepare(state: JobState):
        counts["prepare"] += 1
        print(f"[prepare] 累计调用 {counts['prepare']} 次")
        return {"status": "prepared"}

    def fetch(state: JobState):
        counts["fetch"] += 1
        call_number = counts["fetch"]
        print(f"[fetch] 累计第 {call_number} 次调用")
        if call_number <= failures_before_success:
            print("[fetch] 模拟数据源暂时不可用，抛出异常")
            raise TemporaryReadError("模拟临时读取失败")
        return {"payload": f"任务 {state['job_id']} 的模拟数据", "status": "fetched"}

    def finish(state: JobState):
        counts["finish"] += 1
        print(f"[finish] 收到数据，累计调用 {counts['finish']} 次")
        return {"status": "completed"}

    builder = StateGraph(JobState)
    builder.add_node("prepare", prepare)
    builder.add_node(
        "fetch",
        fetch,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.01,
            jitter=False,
            retry_on=TemporaryReadError,
        ),
    )
    builder.add_node("finish", finish)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "fetch")
    builder.add_edge("fetch", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=InMemorySaver()), counts


def automatic_retry_demo():
    print("=== 1. 临时失败后自动重试成功 ===")
    graph, counts = build_graph(failures_before_success=1)
    config = {"configurable": {"thread_id": "auto-retry"}}
    result = graph.invoke({
        "job_id": "job-A", "payload": "", "status": "new",
    }, config)
    print("最终状态：", result)
    print("各节点调用次数：", counts)


def checkpoint_resume_demo():
    print("\n=== 2. 自动重试耗尽，退出调用后再恢复 ===")
    # 前三次读取失败，第四次恢复正常，模拟数据源在稍后可用。
    graph, counts = build_graph(failures_before_success=3)
    config = {"configurable": {"thread_id": "recover-job"}}
    try:
        graph.invoke({"job_id": "job-B", "payload": "", "status": "new"}, config)
    except TemporaryReadError as exc:
        print("调用方收到异常：", str(exc))

    snapshot = graph.get_state(config)
    print("失败后保存的状态：", snapshot.values)
    print("下一步待执行节点：", snapshot.next)
    print("恢复前的调用次数：", counts)

    print("\n数据源此后可用，用同一 thread_id 和 None 继续未完成执行")
    result = graph.invoke(None, config)
    print("最终状态：", result)
    print("恢复后的调用次数：", counts)
    print("下一步待执行节点：", graph.get_state(config).next)


if __name__ == "__main__":
    automatic_retry_demo()
    checkpoint_resume_demo()
