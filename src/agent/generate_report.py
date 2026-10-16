"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""


# ============================= 导入依赖 =============================
from __future__ import annotations

from dataclasses import dataclass,field
from typing import Any, Dict,Optional
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from loguru import logger
import json
from src.model.models import model_dict







# ================================ 定义函数元数据结构 ================================



# ================================ 定义上下文参数 ================================
@dataclass
class Context:
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """



    model: Optional[str] = None # 模型


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    # ================ 流程状态 ================
    func_metadata: Optional[Dict[str, Any]] = None # json格式函数元数据
    unit_test_str: Optional[str] = None # 单元测试字符串
    test_output: Optional[Dict[str, Any]] = None # 测试函数输出
    test_result: Optional[bool] = None # 测试结果, 是否通过
    test_report: Optional[str] = None # 测试报告

    # ================ 测试函数状态 ================
    index: Optional[int] = None  # 测试函数索引

    # ================ 模型状态 ================
    report_messages: Annotated[list, add_messages] = field(default_factory=list) # 生成测试报告消息列表








# ============================ 定义节点 ============================
def generate_report(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    生成测试报告
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 测试报告
    """

    # 1.组装函数测试json
    metadata = state.func_metadata
    if metadata is None:
        raise ValueError("函数元数据不存在")
    test_json = {
        "tested_function": metadata,
        "unit_test": state.unit_test_str,
        "test_output": state.test_output,
        "test_result": state.test_result,
    }


       # 调用大模型生成测试报告
    model = runtime.context.model
    if model is None or model not in model_dict:
        raise ValueError("模型不存在")
    llm = model_dict.get(model)

    logger.info(f"开始生成测试报告。被测函数：: {state.func_metadata.get('func_name','unknown')}")

    # 系统提示词
    system_prompt = f"""
你是一个专业的代码测试报告智能体，负责根据被测试函数元数据、单元测试、测试输出、测试结果生成测试报告，以MarkDown格式生成测试报告。
包含以下内容：
1. 测试结果
2. 关键问题列表（逻辑错误等导致无法通过测试的原因）
3. 改进建议（提示性能等建议）
测试通过时默认关键问题为空。
以json格式返回。
格式：{{"test_report": "测试报告"}}
"""

    # 用户提示词
    user_prompt = f"""
以下是包含被测函数、单元测试、测试输出、测试结果的json字符串：
{json.dumps(test_json, ensure_ascii=False)}
请根据以上信息生成测试报告。
    """

    # 只获取第一条消息
    messages = state.report_messages[:1] if state.report_messages else []

    # 检测列表是否为空
    if len(messages) == 0:
        messages.append({"role": "system", "content": system_prompt})

    # 添加用户提示词
    messages.append({"role": "user", "content": user_prompt})

    # 调用大模型生成测试报告
    response = llm.invoke(messages)
    content = response.content

    try:
        # json反序列化为字典
        test_report = json.loads(content).get("test_report", "")

        logger.info(f"成功为函数：{state.func_metadata.get('func_name','unknown')} 生成测试报告")
        return {"test_report": test_report, "report_messages": [response]}
    except Exception as e:
        raise ValueError(f"生成测试报告失败：{e}")





def transport_test_report(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    将测试报告传输给主图
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 测试报告
    """
    return {"test_report": state.test_report,"report_messages": state.report_messages,"index": state.index+1}





# ============================ 定义图 ============================

# Define the graph
generate_report_graph = (
    StateGraph(State, context_schema=Context)
    .add_node("generate_report", generate_report)
    .add_node("transport_test_report", transport_test_report)
    .add_edge("__start__", "generate_report")
    .add_edge("generate_report", "transport_test_report")
    .add_edge("transport_test_report", "__end__")
    .compile(name="Generate Report Graph")
)
