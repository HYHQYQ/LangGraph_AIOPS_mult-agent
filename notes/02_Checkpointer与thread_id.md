# 第二课：Checkpointer 与 thread_id

这节沿着 State 继续学习：一次图执行结束后，下一次执行怎样取得已有状态？配套代码：[checkpoint_demo.py](../checkpoint_demo.py)。

## 1. 先分清四个概念

| 概念 | 含义 | 例子 |
| --- | --- | --- |
| State | 当前计算使用的数据 | `{"increment": 3, "total": 5}` |
| Checkpoint | 某个执行时刻的状态快照，包含执行进度信息 | 已有的数据、下一步待执行节点等 |
| Checkpointer | 负责保存和读取 checkpoint 的组件 | `InMemorySaver()` |
| thread_id | 标识一条持续的执行上下文 | `"conversation-001"` 或本例的 `"A"` |

一次调用是一次 run；同一个 thread 可以承载多次 run。这里的 thread 是 LangGraph 的逻辑执行上下文，与操作系统线程不同。[依据：官方 Checkpointers 文档](https://docs.langchain.com/oss/python/langgraph/checkpointers)

## 2. 最小实验：计数器

图只有一个节点：`START → accumulate → END`。

```python
def accumulate(state):
    return {"total": state.get("total", 0) + state["increment"]}
```

输入 `increment`，节点读取已有 `total` 并计算新值。`total` 未配置累加 reducer，框架直接保存节点返回的新总数；本例的加法发生在节点函数内部。

```powershell
python -X utf8 checkpoint_demo.py
```

| 实验 | 两次调用的输入 | 观察结果 |
| --- | --- | --- |
| 没有 checkpointer，即使传同一 thread_id | +2，再 +3 | 2，再 3 |
| 同一 checkpointer、同一 thread A | +2，再 +3 | 2，再 5 |
| 同一 checkpointer、换成 thread B | +3 | B 为 3，A 仍为 5 |
| 新建一个内存 checkpointer，再使用 A | +1 | 1 |

验证记录：2026-10-19，上述四组实验已在当前 Python 3.13.7、LangGraph 1.2.4 环境运行，结果与表中一致。

先前的 State 不会仅凭相同字符串凭空出现；必须从相应 checkpoint 存储中读取。

## 3. 接入需要两处配置

编译时提供存储组件：

```python
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

调用时选择 thread：

```python
config = {"configurable": {"thread_id": "A"}}
graph.invoke({"increment": 2}, config)
graph.invoke({"increment": 3}, config)
```

第二次输入只更新 `increment`，没有重新提交 `total`，因此已有 `total=2` 得以保留，再由节点算成 5。输入也会应用字段合并规则；如果第二次主动输入 `total=0`，就会将本例已有总数覆盖为 0。[依据：官方记忆配置示例](https://docs.langchain.com/oss/python/langgraph/add-memory)

## 4. 怎样检查保存的内容？

```python
snapshot = graph.get_state(config)
print(snapshot.values)  # {'increment': 3, 'total': 5}
print(snapshot.next)    # ()
```

`values` 是已保存的状态；`next` 是下一步待执行节点。本例 `next=()` 表示本次流程已经结束。以后学习暂停与恢复时，执行进度信息就会发挥作用。也可以用 `graph.get_state_history(config)` 查看历史快照。[依据：官方状态读取 API](https://docs.langchain.com/oss/python/langgraph/checkpointers)

本例展示的是完成一次执行后，在相同 thread 发起新输入并延续状态。异常后继续未完成流程、人工暂停后恢复，是接下来的独立实验。

## 5. 先记住三个边界

- `InMemorySaver` 保存于进程内存，退出进程后数据丢失；跨重启保存需要持久化后端。
- 更换 thread_id 会选择另一条上下文；一个用户可以拥有多个会话 thread。
- 保存状态不会让模型自动理解全部内容；真正调用模型时，仍要决定将哪些状态组织进模型输入。

前两点属于存储与执行上下文，第三点属于模型输入的组织，两者需要分别设计。[依据：官方 Persistence 与 Memory 文档](https://docs.langchain.com/oss/python/langgraph/persistence)、[Memory](https://docs.langchain.com/oss/python/langgraph/add-memory)

和 LangChain 对照时，只需记住：经典聊天历史主要帮助重新构造对话输入；LangGraph 的 checkpoint 还服务于图状态与执行进度管理。当前 LangChain `create_agent` 同样可以配置 checkpointer，使用这些底层能力。[依据：LangChain 短期记忆](https://docs.langchain.com/oss/python/langchain/short-term-memory)

## 6. 下一步

先完成两个小实验：把第二次输入改为 `{"increment": 3, "total": 0}`；再将第二次调用改用 thread B。分别预测结果，再运行检查。

接着阅读[第三课：中断与恢复](./03_中断与恢复.md)，通过 `interrupt()` 与 `Command(resume=...)` 观察图如何保存“做到哪里”，以及恢复时哪些代码会再次执行。
