"""第六课：子图复用，以及父子图之间的状态接口。

运行：python -X utf8 subgraph_demo.py
使用普通文本处理，不依赖模型或外部服务。
"""

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph


class AnalysisState(TypedDict):
    text: str
    words: list[str]
    word_count: int


def normalize_node(state: AnalysisState):
    words = state["text"].split()
    normalized = " ".join(words)
    print(f"  [子图 normalize] 规范化文本：{normalized!r}")
    return {"text": normalized, "words": words}


def count_node(state: AnalysisState):
    count = len(state["words"])
    print(f"  [子图 count] 内部 words={state['words']}，单词数={count}")
    return {"word_count": count}


def build_analysis_subgraph():
    builder = StateGraph(AnalysisState)
    builder.add_node("normalize", normalize_node)
    builder.add_node("count", count_node)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "count")
    builder.add_edge("count", END)
    return builder.compile()


class SharedParentState(TypedDict):
    text: str
    word_count: int
    report: str


def report_node(state: SharedParentState):
    print("[父图 report] 使用子图返回的共享字段")
    return {"report": f"{state['text']} | 单词数={state['word_count']}"}


def build_shared_parent(analyzer):
    builder = StateGraph(SharedParentState)
    # text、word_count 是共享字段；父图没有定义子图内部的 words 字段。
    builder.add_node("analyze", analyzer)
    builder.add_node("report", report_node)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "report")
    builder.add_edge("report", END)
    return builder.compile()


class MappedParentState(TypedDict):
    request_text: str
    answer: str
    label: str


def build_mapped_parent(analyzer):
    def call_analysis(state: MappedParentState):
        print("[父图 wrapper] request_text → 子图 text")
        child_result = analyzer.invoke({"text": state["request_text"]})
        print("[父图 wrapper] 子图 text、word_count → 父图 answer")
        return {
            "answer": f"{child_result['text']} | 单词数={child_result['word_count']}",
        }

    builder = StateGraph(MappedParentState)
    builder.add_node("analyze", call_analysis)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", END)
    return builder.compile()


def main():
    # 同一个编译后的子图，复用到两种父图结构。
    analyzer = build_analysis_subgraph()

    print("=== 1. 有共享字段：直接把子图注册为节点 ===")
    shared_result = build_shared_parent(analyzer).invoke({
        "text": "  Learn   LangGraph  step   by step  ",
        "word_count": 0,
        "report": "",
    })
    print("父图结果：", shared_result)
    print("父图最终结果是否包含 words：", "words" in shared_result)

    print("\n=== 2. 字段不同：用包装节点转换输入输出 ===")
    mapped_result = build_mapped_parent(analyzer).invoke({
        "request_text": "  graphs   can be  reused  ",
        "answer": "",
        "label": "文本分析练习",
    })
    print("父图结果：", mapped_result)
    print("父图保留的 label：", mapped_result["label"])


if __name__ == "__main__":
    main()
