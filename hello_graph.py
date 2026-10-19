from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义 State (全局共享的备忘录)
# 所有的节点都只能读取和修改这个字典里的内容
class AIOpsState(TypedDict):
    error_query: str  # 初始的报错信息
    probe_result: str  # 探针探测后的结果
    status: str  # 当前排障状态


# 2. 定义 Node (工作车间/处理函数)
def router_node(state: AIOpsState):
    print(f"🕵️ [Router] 收到报警: {state['error_query']}")
    # 模拟 Router 分析后，更新状态
    return {"status": "analyzing"}


def db_probe_node(state: AIOpsState):
    print("🛠️ [Data Expert] 正在启动底层听诊器，排查数据库...")
    # 模拟探针执行逻辑
    simulated_result = "未发现元数据锁 (MDL)，慢查询日志正常。"
    print(f"🛠️ [Data Expert] 探测完毕: {simulated_result}")
    return {"probe_result": simulated_result, "status": "resolved"}


# 3. 构建 Graph (拼装流水线)
workflow = StateGraph(AIOpsState)

# 将工作车间注册到图里
workflow.add_node("router", router_node)
workflow.add_node("db_probe", db_probe_node)

# 4. 定义 Edge (传送带/执行顺序)
workflow.add_edge(START, "router")  # 强制起点是 router
workflow.add_edge("router", "db_probe")  # router 结束后走到 db_probe
workflow.add_edge("db_probe", END)  # db_probe 结束后整个流程结束

# 5. 编译成可执行的应用
app = workflow.compile()

# 6. 运行测试
if __name__ == "__main__":
    initial_state = {
        "error_query": "商品详情页接口超时报 500",
        "probe_result": "",
        "status": "new"
    }

    print("=== 开始执行 AIOps 排障流 ===")
    final_state = app.invoke(initial_state)
    print("\n=== 执行结束，最终的全局状态 (State) ===")
    print(final_state)