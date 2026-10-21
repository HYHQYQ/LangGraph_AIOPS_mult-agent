# LangGraph 入门第二课：Checkpointer 和 thread_id 如何让状态跨调用延续

> 本文配套课程示例：`checkpoint_demo.py`。只用一个计数器，就能看清图状态、Checkpoint、Checkpointer 和 `thread_id` 的关系。

第一课里，State 只在一次图执行中流转。图执行结束后，如果再调用一次 `invoke()`，LangGraph 会不会自动记得上次的状态？

答案取决于图是否配置了 Checkpointer，以及这次调用使用了哪个 `thread_id`。

## 先区分四个概念

| 概念 | 含义 | 例子 |
| --- | --- | --- |
| State | 图当前处理的数据 | `{"increment": 3, "total": 5}` |
| Checkpoint | 某个执行时刻保存的 State 和进度快照 | 当前值、后续待执行节点 |
| Checkpointer | 负责保存和读取 Checkpoint 的组件 | `InMemorySaver()` |
| `thread_id` | Checkpointer 用来定位一条持续执行上下文的标识 | `"conversation-001"` |

一次 `invoke()` 是一次 run；一个 thread 可以包含多次 run。这里的 thread 是 LangGraph 的逻辑执行上下文，不是操作系统线程。

## 用计数器看状态是否延续

下面是完整的最小图：

```python
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
```

节点每次读取当前累计值，再加上本次输入的增量。这里的加法写在节点函数中；`total` 并没有配置列表式的 Reducer。

## 没有 Checkpointer：每次调用各算各的

```python
config_a = {"configurable": {"thread_id": "A"}}

plain_graph = build_graph()
print(plain_graph.invoke({"increment": 2}, config_a)["total"])  # 2
print(plain_graph.invoke({"increment": 3}, config_a)["total"])  # 3
```

虽然两次调用都传了 `thread_id="A"`，但图没有配置 Checkpointer，因此没有地方保存和读取上次执行的 checkpoint。第二次运行从新的输入开始，结果是 3。

## 同一 Checkpointer、同一 thread：从已有 State 继续

```python
saver = InMemorySaver()
graph = build_graph(saver)

print(graph.invoke({"increment": 2}, config_a)["total"])  # 2
print(graph.invoke({"increment": 3}, config_a)["total"])  # 5
```

这里的关键组合是：**图编译时提供同一个 Checkpointer，调用时使用同一个 `thread_id`**。第二次输入给出新的 `increment=3`，而已保存的 `total=2` 从该 thread 的 checkpoint 中恢复，因此节点算出 5。

`thread_id` 不是“打开记忆”的开关。它只是 Checkpointer 定位状态时使用的标识；如果没有 Checkpointer，传入同一个字符串并不会凭空产生历史状态。

## 换一个 thread：读取另一份状态

```python
config_b = {"configurable": {"thread_id": "B"}}

print(graph.invoke({"increment": 3}, config_b)["total"])  # 3
print(graph.get_state(config_a).values)                    # A 的 total 仍为 5
```

同一个 Checkpointer 可以按不同 `thread_id` 保存多条执行上下文。生产应用通常会把会话 ID 映射为 `thread_id`，但应用仍要自行完成身份校验和访问控制：知道或猜到一个 thread ID，本身不等于拥有读取该会话的权限。

## 用 `get_state()` 查看快照

```python
snapshot = graph.get_state(config_a)

print(snapshot.values)  # 当前保存的 State，例如 {'increment': 3, 'total': 5}
print(snapshot.next)    # 下一步待执行节点；流程完成时通常是 ()
```

`values` 用于查看该 thread 保存的状态，`next` 表示此快照之后待执行的节点。也可以通过 `get_state_history(config)` 检查同一 thread 的历史快照。

在这张只有一个节点的图里，执行完后 `snapshot.next` 为空元组 `()`，因为没有后续节点。若要体验人工暂停与恢复，应单独配置中断点或在节点中使用 `interrupt()`，并在相同的 Checkpointer 和 thread 下恢复；普通执行报错不意味着任意代码都能无条件从函数中间接着跑。

## 新建内存 Checkpointer：旧状态不在新存储里

```python
fresh_graph = build_graph(InMemorySaver())
print(fresh_graph.invoke({"increment": 1}, config_a)["total"])  # 1
```

虽然 thread 名称还是 `A`，但 `fresh_graph` 使用了新的 `InMemorySaver` 实例。新存储没有旧 checkpoint，所以从 1 开始。`thread_id` 和 Checkpointer 需要配合起来理解：一个用于定位，一份实际存储负责保留数据。

## InMemorySaver 的适用范围

`InMemorySaver` 适合本地练习和短生命周期实验。数据只在当前进程内存中；进程重启或换成另一个进程，就不能指望它继续提供之前的状态。

服务中通常在应用启动时创建图和 Checkpointer，再让请求选择对应 thread。需要跨进程或重启保存时，应使用与部署环境匹配的持久化 Checkpointer，并考虑数据保留、删除、备份和访问权限。多进程部署时，进程内存里的 Saver 也不会自动在进程之间共享。

Checkpoint 是执行状态的保存机制，不等于模型自动“记住一切”。模型下一次看到什么，仍由应用如何把 State 整理成模型输入决定。

## 本课小结

- **State** 是图正在使用的数据；**Checkpoint** 是保存下来的状态快照。
- **Checkpointer** 提供读写 checkpoint 的能力；**`thread_id`** 选择一条持续上下文。
- 想跨调用延续状态，需要复用相同 Checkpointer，并为同一上下文传入相同 `thread_id`。
- 进程内存 Saver 适合教学和本地实验；持久化、隔离和权限需要结合实际部署设计。

示例完整代码：`checkpoint_demo.py`。后续课程会用中断与恢复来观察执行进度如何和状态一起保存。

## 参考资料

- [LangGraph Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [LangGraph Add Memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph Graph API：Re-execution and idempotency](https://docs.langchain.com/oss/python/langgraph/graph-api#re-execution-and-idempotency)

**建议标签：** LangGraph、Python、AI Agent、后端开发
