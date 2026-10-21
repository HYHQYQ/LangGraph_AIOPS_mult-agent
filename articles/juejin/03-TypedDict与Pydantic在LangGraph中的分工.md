# TypedDict 还是 Pydantic？在 LangGraph 中按边界选择，而不是按“开发/生产”选择

刚接触 LangGraph 时，很容易把 `TypedDict` 和 Pydantic 想成一条升级路线：先用 `TypedDict` 快速试验，等项目上线再换成 Pydantic。

更实用的理解是：它们解决的问题不同。`TypedDict` 描述图内状态的字段和类型；Pydantic 擅长把外部数据解析、校验成符合约束的对象。两者可以出现在同一个应用里。

## 先看各自的职责

| 需求 | 常见选择 | 说明 |
| --- | --- | --- |
| 描述 LangGraph 图内状态 | `TypedDict` | 轻量的类型提示，节点仍以字典方式读写字段 |
| 为状态字段提供默认值 | `dataclass` 或其他合适方案 | LangGraph 文档将 `dataclass` 列为带默认值的选择 |
| 校验外部输入或结构化业务数据 | Pydantic `BaseModel` | 可以定义约束，并在明确的边界调用 `model_validate()` |
| 希望 StateGraph 使用递归数据校验 | Pydantic State | LangGraph 支持，但官方提示性能低于 `TypedDict` 或 `dataclass` |

官方 Graph API 将 `TypedDict` 列为 State 的主要定义方式；也支持 `BaseModel` 作为图状态以进行递归数据校验，并提示它的性能低于 `TypedDict` 或 `dataclass`。这不是“生产一定选哪一个”的普遍结论，而是需要结合校验需求和开销做取舍。

## `TypedDict` 描述数据，不负责运行时校验

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add


class ResearchState(TypedDict):
    question: str
    search_count: int
    max_searches: int
    search_history: Annotated[list[str], add]
```

`TypedDict` 的好处是直观：节点可以通过 `state["question"]` 读值，并返回普通字典作为更新。像 `Annotated[list[str], add]` 这样的注解还可以让 LangGraph 知道如何合并该字段。

但 `TypedDict` 是类型提示，不会自动检查一份来自 HTTP 请求的字典是否真的包含字符串 `question`、整数 `search_count`。外部数据需要在入口处显式验证。

## Pydantic 适合放在数据边界

例如 API 收到用户的研究请求时，可以先用 Pydantic 验证，再转换为图需要的状态：

```python
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    max_searches: int = Field(ge=1, le=10)


raw_request = {
    "question": "为什么销售额下降？",
    "max_searches": 3,
}

request = ResearchRequest.model_validate(raw_request)

initial_state: ResearchState = {
    "question": request.question,
    "search_count": 0,
    "max_searches": request.max_searches,
    "search_history": [],
}
```

这样职责比较清楚：API 输入先通过 Pydantic，进入图之后以 `TypedDict` 描述的 State 流转。

对模型生成的结构化结果，也可以在进入业务流程之前验证：

```python
class Evaluation(BaseModel):
    score: int = Field(ge=0, le=100)
    reasoning: str = Field(min_length=1)


evaluation = Evaluation.model_validate(
    {"score": 92, "reasoning": "多条独立来源支持该结论"}
)
```

如果输入是 LLM 返回的任意 JSON 或字典，显式执行 `model_validate()` 才会应用这些约束。然后可把经过验证的值转换为图状态更新：

```python
return {
    "evidence_score": evaluation.score,
    "evaluation_reasoning": evaluation.reasoning,
}
```

是否把整个 Pydantic 实例直接放进 State，要结合后续节点的使用方式和 checkpoint 序列化要求决定；跨存储边界时，普通字典或 JSON 结构往往更容易检查和迁移。

## LangGraph 的增量更新不等于 Pydantic 必须有所有默认值

节点返回的通常是 State 更新，例如：

```python
def score_node(state: ResearchState):
    return {"search_count": state["search_count"] + 1}
```

State 中没被更新的字段仍然保留。这个“只返回增量”的工作方式由 LangGraph 的 State 更新机制决定，和你选 `TypedDict` 还是 Pydantic 不是同一个概念。

因此，不要仅仅为了适配节点的部分更新，就把每个 Pydantic 字段都改成可选或塞入空值默认值。先确定数据的完整性约束：哪些字段是输入时必需，哪些字段只会在流程某一步之后出现，再选择图状态的 Schema 和入口验证方式。

如果直接把 `BaseModel` 用作 `StateGraph` 的状态 Schema，LangGraph 支持这种方式；官方指出它可用于递归校验，但性能低于 `TypedDict` 和 `dataclass`。如果“每个节点返回的数据都必须通过某种业务校验”是硬要求，建议在具体的节点或状态转换边界显式调用 Pydantic 校验，并针对项目使用的 LangGraph 版本验证实际行为，不要把静态类型注解当作运行时保证。

## 为什么不是“外层永远 TypedDict，里层永远 Pydantic”？

“边界用 Pydantic，图内 State 用 TypedDict”是一个很实用的默认设计，但不是唯一正确答案：

- 小型流程可能直接用 `TypedDict` 和入口检查就够了。
- 如果 StateGraph 本身需要递归校验，也可以使用 Pydantic State，并接受相应成本。
- 如果主要诉求是给图状态字段提供默认值，可以评估 `dataclass`。
- 一个应用也可以用 Pydantic 校验 HTTP 输入、用 Pydantic 约束模型输出，同时让图内 State 保持 `TypedDict`。

关键是明确“在哪个边界、对哪份数据、以什么方式做校验”，而不是把某个库标签等同于“生产级”。

## 一个简化的混合结构

```python
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel, Field


class Evaluation(BaseModel):
    score: int = Field(ge=0, le=100)
    reasoning: str


class ReviewState(TypedDict):
    question: str
    search_count: int
    latest_evaluation: NotRequired[Evaluation]


def review_node(state: ReviewState):
    # 模拟从模型或规则服务得到的数据；进入状态前显式校验。
    evaluation = Evaluation.model_validate(
        {"score": 86, "reasoning": "找到两条相互印证的证据"}
    )
    return {"latest_evaluation": evaluation}
```

这个例子让 Pydantic 负责审核结果的字段约束，让 `TypedDict` 描述 LangGraph 的整体状态形状。真实项目如果要把 Pydantic 对象存进 checkpoint，应确认所用存储和序列化配置能够正确处理它；也可以在写入状态时转成 `evaluation.model_dump()`。

## 怎么做选择？

可以按以下顺序判断：

1. **先确认数据来自哪里。** HTTP、文件、数据库或模型输出等外部数据，通常值得在进入业务逻辑前校验。
2. **再确认图内是否需要强运行时校验。** 如果需要，可以评估 Pydantic State；如果主要是类型提示和状态流转，`TypedDict` 通常更直接。
3. **校验规则要放在明确边界。** Pydantic 类型声明本身不会替代 `model_validate()`，`TypedDict` 也不会在运行时检查字典内容。
4. **考虑默认值和持久化。** LangGraph 文档建议需要 State 默认值时考虑 `dataclass`；保存到 checkpoint 的对象还要考虑序列化与版本迁移。

## 总结

`TypedDict` 和 Pydantic 没有“玩具版 / 生产版”的关系。前者适合清晰地表达图状态字段，后者适合在需要的地方校验复杂数据。很多项目会把它们配合起来：在系统边界校验输入和结构化结果，在图内按需选择 State Schema。

## 参考资料

- [LangGraph Graph API：State schema 与 Pydantic 支持](https://docs.langchain.com/oss/python/langgraph/graph-api#schema)
- [LangGraph Graph API：Reducers 与部分 State 更新](https://docs.langchain.com/oss/python/langgraph/graph-api#reducers)
- [Pydantic 官方文档](https://docs.pydantic.dev/latest/)

**建议标签：** Python、Pydantic、LangGraph、类型系统
