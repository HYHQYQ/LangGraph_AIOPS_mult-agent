"""可选案例：把经典 AgentExecutor 的工具循环展开成 LangGraph。

运行：python -X utf8 aiops_agent_loop_demo.py
模型决策和工具数据均为本地模拟；用于观察消息协议，不代表真实排障。
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def search_internal_docs(query: str) -> str:
    """模拟检索内部微服务排障规范。"""
    return json.dumps(
        {"source": "mock-runbook", "query": query,
         "guidance": "结合注册状态、请求链路和数据库证据排障，不能仅凭服务健康判定业务正常。"},
        ensure_ascii=False,
    )


@tool
def get_nacos_service_list() -> str:
    """模拟获取注册服务列表。"""
    return json.dumps({"services": ["item-service"]})


@tool
def check_nacos_service_status(service_name: str) -> str:
    """模拟查询指定服务的注册健康状态。"""
    return json.dumps({"service": service_name, "healthy_instances": 1})


@tool
def query_trace_logs(trace_id: str) -> str:
    """模拟按 TraceId 检索请求链路。"""
    return json.dumps(
        {"trace_id": trace_id, "events": ["gateway: upstream timeout", "item-service: request slow"]}
    )


TOOLS = [search_internal_docs, get_nacos_service_list,
         check_nacos_service_status, query_trace_logs]


def scripted_model_node(state: MessagesState):
    """用固定规则模拟模型产生 AIMessage，真实接入时替换此函数的决策部分。"""
    messages = state["messages"]
    question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
    observations = {
        m.name: json.loads(m.content)
        for m in messages if isinstance(m, ToolMessage)
    }
    trace_match = re.search(r"\bT-\d+\b", question)

    # 演示中显式规定优先级：有 TraceId 先查链路，否则先查规范。
    # 这些是 Python 规则，不是模型自主规划，也不是原项目已保证的工具顺序。
    if trace_match:
        plan = [
            ("query_trace_logs", {"trace_id": trace_match.group()}),
            ("search_internal_docs", {"query": question}),
        ]
    else:
        registered_services = observations.get("get_nacos_service_list", {}).get("services", [])
        plan = [
            ("search_internal_docs", {"query": question}),
            ("get_nacos_service_list", {}),
        ]
        if registered_services:
            plan.append(("check_nacos_service_status", {"service_name": registered_services[0]}))

    for name, args in plan:
        if name not in observations:
            return {"messages": [AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args,
                             "id": f"call_{len(observations) + 1}", "type": "tool_call"}],
            )]}

    if trace_match:
        trace = observations["query_trace_logs"]
        conclusion = f"[模拟分析] {trace['trace_id']} 的链路记录：{'；'.join(trace['events'])}。根因仍需更多证据。"
    else:
        status = observations.get("check_nacos_service_status")
        conclusion = (
            f"[模拟分析] {status['service']} 有 {status['healthy_instances']} 个健康实例，"
            "还需读取日志或数据库证据判断业务异常。"
            if status else "[模拟分析] 没有可检查的注册服务，需要补充信息。"
        )
    # 没有 tool_calls 的 AIMessage 将由条件边路由至 END。
    return {"messages": [AIMessage(content=conclusion)]}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("model", scripted_model_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", tools_condition, {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "model")
    return builder.compile()


def run_demo(question: str):
    print(f"\n=== 用户：{question} ===")
    result = build_graph().invoke(
        {"messages": [HumanMessage(content=question)]},
        {"recursion_limit": 12},
    )
    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"AIMessage 请求工具：{call['name']}，参数={call['args']}，id={call['id']}")
        elif isinstance(message, ToolMessage):
            print(f"ToolMessage 返回结果：{message.name}，tool_call_id={message.tool_call_id}")
            print(f"  {message.content}")
        else:
            print(f"{message.type}: {message.content}")
    return result


if __name__ == "__main__":
    run_demo("item-service 商品查询很慢，帮我排查。")
    run_demo("请求 T-1001 超时了，帮我查调用链路。")
