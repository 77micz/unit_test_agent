"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

# ============================= 导入依赖 =============================
from __future__ import annotations

import datetime
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
# 导入inspect模块
from loguru import logger


# ================================ 定义函数元数据结构 ================================


# ================================ 定义上下文参数 ================================
@dataclass
class Context:
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    test_dir: Optional[str] = None  # 测试函数保存目录


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    # ================ 流程状态 ================
    function_name: Optional[str] = None  # 函数名称
    unit_test_str: Optional[str] = None  # 单元测试字符串
    unit_test_save_path: Optional[str] = None  # 单元测试保存路径


# ============================ 定义节点 ============================
def save_unit_test(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    保存函数单元测试
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 包含测试函数保存路径的字典
    """

    # 获取函数元数据
    function = state.unit_test_str
    if function is None:
        raise ValueError("函数字符串不存在")

    # 保存函数
    test_dir = runtime.context.test_dir
    if test_dir is None:
        raise ValueError("测试函数保存目录不存在")

    logger.info(f"开始保存：{state.function_name} 的单元测试函数")

    parent_dir = f"{test_dir}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    # 创建父目录
    os.makedirs(parent_dir, exist_ok=True)
    save_path = f"{parent_dir}/{state.function_name}_{uuid.uuid4().hex}.py"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(function)

    logger.info(f"被测试函数：{state.function_name} 的单元测试函数已保存到: {save_path}")

    return {"unit_test_save_path": save_path}






def transport_save_path(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    将测试函数保存路径传输给主图
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 测试函数保存路径
    """
    return {"unit_test_save_path": state.unit_test_save_path}





# ============================ 定义图 ============================

# Define the graph
unit_test_save_graph = (
    StateGraph(State, context_schema=Context)
    .add_node("save_unit_test", save_unit_test)
    .add_node("transport_save_path", transport_save_path)
    .add_edge("__start__", "save_unit_test")
    .add_edge("save_unit_test", "transport_save_path")
    .add_edge("transport_save_path", "__end__")
    .compile(name="Unit Test Save Graph")
)
