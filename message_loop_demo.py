"""第四课：MessagesState、工具请求与结果消息、Agent 循环。

运行：python -X utf8 message_loop_demo.py
决策节点使用固定 Python 规则模拟模型输出；加法与乘法工具真实执行。
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class CalculatorState(MessagesState):
    left: int
    right: int
    multiplier: int | None
    model_steps: int


@tool
def add(a: int, b: int) -> int:
    """返回两个整数的和。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """返回两个整数的积。"""
    return a * b


TOOLS = [add, multiply]


def scripted_model_node(state: CalculatorState):
    """演示消息协议，不解析自然语言，不调用真实模型。"""
    last = state["messages"][-1]
    step = state["model_steps"] + 1

    if isinstance(last, HumanMessage):
        reply = AIMessage(content="", tool_calls=[{
            "name": "add",
            "args": {"a": state["left"], "b": state["right"]},
            "id": f"call_{step}",
            "type": "tool_call",
        }])
    elif isinstance(last, ToolMessage):
        if last.name == "add" and state["multiplier"] is not None:
            # 使用上一轮工具返回的结果，构造下一次工具请求。
            reply = AIMessage(content="", tool_calls=[{
                "name": "multiply",
                "args": {"a": int(last.content), "b": state["multiplier"]},
                "id": f"call_{step}",
                "type": "tool_call",
            }])
        else:
            reply = AIMessage(content=f"结果是 {last.content}。")
    else:
        raise ValueError("本演示的决策输入应为用户消息或工具结果")

    # 只返回新消息，MessagesState 负责合并到现有消息列表。
    return {"messages": [reply], "model_steps": step}


def build_graph():
    builder = StateGraph(CalculatorState)
    builder.add_node("model", scripted_model_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", tools_condition, {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "model")
    return builder.compile()


def run_demo(left: int, right: int, multiplier: int | None = None):
    expression = f"{left} + {right}"
    if multiplier is not None:
        expression = f"({expression}) * {multiplier}"
    print(f"\n=== 计算 {expression} ===")
    result = build_graph().invoke({
        "messages": [HumanMessage(content=f"请计算 {expression}。")],
        "left": left,
        "right": right,
        "multiplier": multiplier,
        "model_steps": 0,
    }, {"recursion_limit": 10})

    for index, message in enumerate(result["messages"], 1):
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"{index}. AIMessage：请求 {call['name']}，参数={call['args']}，id={call['id']}")
        elif isinstance(message, ToolMessage):
            print(f"{index}. ToolMessage：{message.name} 返回 {message.content}，tool_call_id={message.tool_call_id}")
        else:
            print(f"{index}. {type(message).__name__}：{message.content}")
    print(f"模拟决策轮数：{result['model_steps']}；最终消息条数：{len(result['messages'])}")
    return result


def show_message_merge():
    print("\n=== add_messages 按消息自身的 ID 合并 ===")
    original = [HumanMessage(content="最初的问题", id="user-1")]
    corrected = add_messages(original, [HumanMessage(content="修改后的问题", id="user-1")])
    extended = add_messages(corrected, [AIMessage(content="收到。", id="assistant-1")])
    print("相同消息 ID 更新：", [(m.id, m.content) for m in corrected])
    print("不同消息 ID 加入：", [(m.id, m.content) for m in extended])


if __name__ == "__main__":
    run_demo(2, 3)
    run_demo(2, 3, multiplier=4)
    show_message_merge()
