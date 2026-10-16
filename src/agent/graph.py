"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

# ============================== 导入依赖 ==============================

from __future__ import annotations

from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Annotated, Optional, Literal, List
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from src.model.models import model_list
# 导入子图
from src.agent.generate_test import generate_unit_test_graph
from src.agent.unit_test_save import unit_test_save_graph
from src.agent.run_test import run_test_graph
from src.agent.generate_report import generate_report_graph
from src.agent.report_save import report_save_graph
from src.agent.load_function import load_funcs_graph

# ============================== 准备环境 ==============================


# 加载环境变量
load_dotenv()

SQLITE_PATH = Path(__file__).parent.parent.parent / "data/database/test.db"
os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
# 定义检查点保存器
checkpoint_saver = SqliteSaver(conn)

# 测试函数保存目录
TEST_DIR = Path(__file__).parent.parent.parent / "data/test_functions"
os.makedirs(os.path.dirname(TEST_DIR), exist_ok=True)

# 测试报告保存目录
REPORT_DIR = Path(__file__).parent.parent.parent / "data/reports"
os.makedirs(os.path.dirname(REPORT_DIR), exist_ok=True)


# ============================== 定义状态 ==============================

@dataclass
class Context:
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model: str = model_list[0]  # 模型
    sqlite_path: str = SQLITE_PATH  # sqlite数据库路径
    test_dir: str = TEST_DIR  # 测试函数保存目录
    report_dir: str = REPORT_DIR  # 测试报告保存目录


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    # ================ 流程状态 ================
    file_path: Optional[str] = None  # 被测函数所在路径
    func_metadata_list: Optional[List[Dict[str, Any]]] = None  # 函数元数据列表
    imports: Optional[List[str]] = None  # 被测函数中的import语句列表
    unit_test_response: Optional[Dict[str, Any]] = None  # 单元测试响应字典
    unit_test_save_path: Optional[str] = None  # 单元测试保存路径
    test_output: Optional[Dict[str, Any]] = None  # 测试函数输出
    test_result: Optional[bool] = None  # 测试结果, 是否通过
    test_report: Optional[str] = None  # 测试报告
    report_save_path: Optional[str] = None  # 测试报告保存路径

    # ================ 测试函数状态 ================
    index: Optional[int] = None  # 测试函数索引

    # ================ 模型状态 ================
    test_messages: Annotated[list, add_messages] = field(default_factory=list)  # 生成单元测试消息列表
    dependencies_messages: Annotated[list, add_messages] = field(default_factory=list)  # 生成依赖列表消息列表
    report_messages: Annotated[list, add_messages] = field(default_factory=list)  # 生成测试报告消息列表


# ============================== 定义节点 ==============================


def load_funcs(state: State, runtime: Runtime) -> Dict[str, Any]:
    """
    加载测试函数对象
    :param state: 状态对象
    :param runtime: 运行时对象
    :return: 包含测试函数对象的字典
    """

    # 调用子图
    return load_funcs_graph.invoke(
        input={"file_path": state.file_path},  # 从状态中获取文件路径
    )


def generate_unit_test(state: State, runtime: Runtime) -> Dict[str, Any]:
    """
    生成单元测试字符串
    :param state: 状态对象
    :param runtime: 运行时对象
    :return: 包含单元测试字符串的字典
    """

    # 调用子图
    return generate_unit_test_graph.invoke(
        input={"func_metadata": state.func_metadata_list[state.index]},  # 从状态中获取函数元数据
        context={
            "model": runtime.context.model,
        },
    )


def unit_test_save(state: State, runtime: Runtime) -> Dict[str, Any]:
    """
    保存测试函数单元测试
    :param state: 状态对象
    :param runtime: 运行时对象
    :return: 包含保存路径的字典
    """

    # 调用子图
    return unit_test_save_graph.invoke(
        input={"function_name": state.func_metadata_list[state.index].get("func_name", "unknown"),
               "unit_test_str": state.unit_test_response.get("unit_test").get("unit_test_str", "")},
        # 从状态中获取函数元数据和单元测试字符串
        context={
            "test_dir": runtime.context.test_dir,
        },
    )


def run_test(state: State, runtime: Runtime) -> Dict[str, Any]:
    """
    运行单元测试
    :param state: 状态对象
    :param runtime: 运行时对象
    :return: 包含测试结果的字典
    """

    # 调用子图
    return run_test_graph.invoke(
        input={"func_metadata": state.func_metadata_list[state.index], "imports": state.imports,
               "unit_test_response": state.unit_test_response, "dependencies_messages": state.dependencies_messages},
        # 从状态中获取函数元数据、import语句列表、单元测试响应字典和依赖列表消息列表
        context={
            "model": runtime.context.model,
        }
    )


def generate_report(state: State, runtime: Runtime) -> Dict[str, Any]:
    """
    生成测试报告
    :param state: 状态对象
    :param runtime: 运行时对象
    :return: 包含测试报告的字典
    """

    # 调用子图
    return generate_report_graph.invoke(
        input={"func_metadata": state.func_metadata_list[state.index],
               "unit_test_str": state.unit_test_response.get("unit_test").get("unit_test_str", ""),
               "test_output": state.test_output, "test_result": state.test_result,
               "report_messages": state.report_messages,
               "index": state.index},  # 从状态中获取函数元数据、测试报告消息列表和测试函数索引
        context={
            "model": runtime.context.model,
        },
    )


def report_save(state: State, runtime: Runtime) -> Dict[str, Any]:
    """
    保存测试报告
    :param state: 状态对象
    :param runtime: 运行时对象
    :return: 包含测试报告的字典
    """

    # 调用子图
    return report_save_graph.invoke(
        input={"func_metadata": state.func_metadata_list[state.index - 1], "test_report": state.test_report},
        # 从状态中获取函数元数据和测试报告
        context={
            "report_dir": runtime.context.report_dir,
        },
    )


def route_by_index(state: State, runtime: Runtime) -> Literal["end", "generate_unit_test"]:
    """
    根据索引路由到对应的节点
    :param state: 状态对象
    :param runtime: 运行时对象
    :return: 包含路由索引的字典
    """

    index = state.index
    if index >= len(state.func_metadata_list):
        return "end"
    else:
        return "generate_unit_test"


# ================================== 编译图 ==================================

# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("generate_unit_test", generate_unit_test)
    .add_node("unit_test_save", unit_test_save)
    .add_node("run_test", run_test)
    .add_node("generate_report", generate_report)
    .add_node("report_save", report_save)
    .add_node("load_funcs", load_funcs)
    .add_edge("__start__", "load_funcs")
    .add_edge("load_funcs", "generate_unit_test")
    .add_edge("generate_unit_test", "unit_test_save")
    .add_edge("unit_test_save", "__end__")
    .add_edge("generate_unit_test", "run_test")
    .add_edge("run_test", "generate_report")
    .add_conditional_edges(
        "generate_report",
        route_by_index,
        {
            "end": "__end__",
            "generate_unit_test": "generate_unit_test",
        }
    )
    .add_edge("generate_report", "report_save")
    .add_edge("report_save", "__end__")
    .compile(
        name="Main Graph",
        # checkpointer=checkpoint_saver,
    )
)

# ================================== 调用图 ==================================
