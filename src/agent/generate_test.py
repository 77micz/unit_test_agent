"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

# ============================= 导入依赖 =============================
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from loguru import logger
from typing_extensions import Annotated

from src.model.models import model_dict


# ================================ 定义llm响应结构 ================================

response_schema = {
    "unit_test": {
        "unit_test_str": "单元测试函数字符串",
        "func_name": "单元测试函数名",
    },
    "imports": ["测试需要导入的需要安装的第三方依赖1","测试需要导入的需要安装的第三方依赖2"],
}



# ================================ 定义上下文参数 ================================
@dataclass
class Context:
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """



    model: Optional[str] = None  # 模型


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    # ================ 流程状态 ================
    func_metadata: Optional[Dict[str, Any]] = None  # json格式函数元数据
    unit_test_response: Optional[Dict[str, Any]] = None # 单元测试响应字典

    # ================ 模型状态 ================
    test_messages: Annotated[list, add_messages] = field(default_factory=list)  # 生成单元测试消息列表


# ============================ 定义节点 ============================
def generate_unit_test(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    生成单元测试字符串
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 单元测试字符串
    """

    # 获取函数元数据
    metadata = state.func_metadata
    if metadata is None:
        raise ValueError("函数元数据不存在")

    # 调用大模型生成单元测试
    model = runtime.context.model
    if model is None or model not in model_dict:
        raise ValueError("模型不存在")
    llm = model_dict.get(model)

    logger.info(f"开始生成函数: {metadata.get('func_name','unknown')} 单元测试")

    # 系统提示词
    system_prompt = f"""
你是一个专业的代码测试智能体，负责根据函数元数据生成单元测试函数字符串，关键的地方加上注释说明，文档中说明是哪个函数的单元测试。
以json格式返回单元测试函数字符串。
格式：{response_schema}
python自带软件包直接导入，不需要安装，例如：import sys,from typing import List
"""

    # 用户提示词
    user_prompt = f"""
函数元数据：{metadata}
    """

    # 只获取第一条消息
    messages = state.test_messages[:1] if state.test_messages else []

    # 检测列表是否为空
    if len(messages) == 0:
        messages.append({"role": "system", "content": system_prompt})

    # 添加用户提示词
    messages.append({"role": "user", "content": user_prompt})

    # 调用大模型生成单元测试
    response = llm.invoke(messages)
    content = response.content

    try:
        # json反序列化为字典
        unit_test_response = json.loads(content)

        logger.info(f"成功为函数：{metadata.get('func_name','unknown')} 生成单元测试")
        return {"unit_test_response": unit_test_response, "test_messages": [response]}
    except Exception as e:
        raise ValueError(f"生成单元测试失败：{e}")


def transport_function_str(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    将函数字符串传输给主图
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 函数字符串
    """
    return {"unit_test_response": state.unit_test_response}


# ============================ 定义图 ============================

# Define the graph
generate_unit_test_graph = (
    StateGraph(State, context_schema=Context)
    .add_node("generate_unit_test", generate_unit_test)
    .add_node("transport_function_str", transport_function_str)
    .add_edge("__start__", "generate_unit_test")
    .add_edge("generate_unit_test", "transport_function_str")
    .add_edge("transport_function_str", "__end__")
    .compile(name="Generate Unit Test Graph")
)
