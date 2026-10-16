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



    report_dir: Optional[str] = None  # 测试报告保存目录


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    # ================ 流程状态 ================
    func_metadata: Optional[Dict[str, Any]] = None  # 函数元数据
    test_report: Optional[str] = None # 测试报告
    report_save_path: Optional[str] = None  # 测试报告保存路径


# ============================ 定义节点 ============================
def save_report(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    保存测试报告
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 包含测试函数保存路径的字典
    """

    # 获取函数元数据
    report = state.test_report
    if report is None:
        raise ValueError("测试报告不存在")

    # 保存测试报告
    report_dir = runtime.context.report_dir
    if report_dir is None:
        raise ValueError("测试报告保存目录不存在")

    logger.info(f"开始保存：{state.func_metadata.get('func_name','unknown')} 的测试报告")

    parent_dir = f"{report_dir}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    # 创建父目录
    os.makedirs(parent_dir, exist_ok=True)
    save_path = f"{parent_dir}/{state.func_metadata.get('func_name','unknown')}_report_{uuid.uuid4().hex}.md"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"被测试函数：{state.func_metadata.get('func_name','unknown')} 的测试报告已保存到: {save_path}")

    return {"report_save_path": save_path}






def transport_save_path(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    将测试报告保存路径传输给主图
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 测试报告保存路径
    """
    return {"report_save_path": state.report_save_path}





# ============================ 定义图 ============================

# Define the graph
report_save_graph = (
    StateGraph(State, context_schema=Context)
    .add_node("save_report", save_report)
    .add_node("transport_save_path", transport_save_path)
    .add_edge("__start__", "save_report")
    .add_edge("save_report", "transport_save_path")
    .add_edge("transport_save_path", "__end__")
    .compile(name="Report Save Graph")
)
