"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

# ============================= 导入依赖 =============================
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from langgraph.graph.message import add_messages
from pathlib import Path
from typing import Any, Dict, Optional, List, Annotated

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
# 导入inspect模块
from loguru import logger
import shutil
import re
from model.models import model_dict
import json
import uuid

# ================================ 全局变量 ================================


dependencies_schema = {
    "dependencies": ["requests", "pandas", "numpy", "pytest"],
}

PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
IMAGE = "python:3.12-slim"


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

    # ================ 输入 ================
    func_metadata: Optional[Dict[str, Any]] = None  # json格式函数元数据
    imports: Optional[List[str]] = None  # 被测函数中的import语句列表
    unit_test_response: Optional[Dict[str, Any]] = None  # 单元测试响应字典
    # ================ 中间结果 ================
    dependencies: Optional[List[str]] = None  # 第三方依赖列表

    # ================== 输出 ==================
    test_output: Optional[Dict[str, Any]] = None  # 测试函数输出
    test_result: Optional[bool] = None  # 测试结果, 是否通过

    # ================ 模型状态 ================
    dependencies_messages: Annotated[list, add_messages] = field(default_factory=list)  # 生成依赖列表消息列表


# ============================ 定义节点 ============================


def extract_dependencies(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    从单元测试响应和被测函数中提取第三方依赖
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 包含第三方依赖的字符串列表
    """

    dependencies = []

    # 1. 从单元测试响应中提取第三方依赖
    unit_test_dependencies = state.unit_test_response.get("unit_test", {}).get("imports", [])

    # 2. 从被测函数中提取第三方依赖
    func_dependencies = state.imports or []

    # 合并
    dependencies.extend(unit_test_dependencies)
    dependencies.extend(func_dependencies)

    # 3. 去重
    dependencies = list(set(dependencies))
    # 检查是否有内容
    if not dependencies:
        return {"dependencies": [], "dependencies_messages": []}

    # 4.调用大模型获取依赖名称
    llm = model_dict.get(runtime.context.model, None)
    if llm is None:
        raise ValueError(f"模型 {runtime.context.model} 不存在")

    logger.info(f"开始生成函数: {state.func_metadata.get('func_name', 'unknown')} 依赖列表")

    system_prompt = f"""
你是一个专业的Python依赖提取助手，你的任务是根据给定列表中包含的import和from语句，获取第三方依赖的包名。
请返回一个列表，每个元素是一个第三方依赖的包名(可以pip install的包/库名)，列表中不能包含重复的包名。
以json格式响应。
响应格式：{dependencies_schema}
"""

    user_prompt = f"""
请根据以下import和from语句，提取第三方依赖的包名：
{dependencies}
"""

    # 只获取第一条消息
    messages = state.dependencies_messages[:1] if state.dependencies_messages else []

    # 检测列表是否为空
    if len(messages) == 0:
        messages.append({"role": "system", "content": system_prompt})

    # 添加用户提示词
    messages.append({"role": "user", "content": user_prompt})

    # 调用大模型生成依赖列表消息
    response = llm.invoke(messages)
    content = response.content

    try:
        # json反序列化为字典
        dependencies = json.loads(content)

        logger.info(f"成功为函数: {state.func_metadata.get('func_name', 'unknown')} 获取依赖列表： {dependencies}")
        return {"dependencies": dependencies, "dependencies_messages": [response]}
    except Exception as e:
        raise ValueError(f"提取第三方依赖失败：{e}")


def run_test_in_sandbox(state: State, runtime: Runtime[Context], timeout: int = 120) -> Dict[str, Any]:
    """
    在 Docker 沙箱中运行测试函数。
    返回 {"test_output": {"stdout": ..., "stderr": ..., "return_code": ...}}
    :param state: 图状态
    :param runtime: 运行时环境
    :param timeout: 超时时间，单位秒
    :return: 包含测试结果的字典
    """

    # 1. 获取测试代码、被测函数源码和函数名
    test_code = state.unit_test_response.get("unit_test", {}).get("unit_test_str", "")
    func_metadata = state.func_metadata or {}
    func_name = func_metadata.get("func_name", "unknown_func")
    func_code = func_metadata.get("func_code", "")
    # 单元测试名称
    unit_test_name = state.unit_test_response.get("unit_test").get("func_name", "")

    if not test_code or not func_code:
        return {"test_output": {"stdout": "", "stderr": "缺少测试代码或被测函数源码", "return_code": -1},
                "test_result": False}

    # 2. 创建临时目录，用于挂载进容器
    temp_dir = tempfile.mkdtemp()

    volume_name = None
    try:
        # 3. 写入被测函数文件（例如 removeKdigits.py）
        func_file = Path(temp_dir) / f"{func_name}.py"
        func_file.write_text(func_code, encoding="utf-8")

        # 4. 构建完整的测试文件内容
        # 如果 LLM 没有生成 import，我们动态补上
        import_stmt = f"from {func_name} import {func_name}\n"
        test_code = import_stmt + test_code

        # 强制添加执行入口，确保函数被调用
        run_stmt = f'\n\nif __name__ == "__main__":\n    {unit_test_name}()\n'
        if "if __name__" not in test_code:
            test_code += run_stmt

        # 写入测试文件（例如 test_removeKdigits.py）
        test_file = Path(temp_dir) / f"{unit_test_name}.py"
        test_file.write_text(test_code, encoding="utf-8")

        logger.info(f"开始在测试沙箱中运行测试，被测试函数：{func_name}")

        # 5. 构建依赖列表
        dependencies = state.dependencies or []
        volume_name = f"sandbox_deps_{uuid.uuid4().hex[:12]}"

        # 创建临时 volume
        subprocess.run(
            ["docker", "volume", "create", volume_name],
            capture_output=True, check=True,
        )

        # ---- 阶段 1：装依赖（有网络）----
        if dependencies:
            install_cmd = [
                "docker", "run", "--rm",
                "--network", "bridge",
                "--memory", "512m", "--cpus", "1",
                "--security-opt", "no-new-privileges",
                "-v", f"{volume_name}:/deps",
                IMAGE,
                "pip", "install", "--no-cache-dir", "-q",
                "--target=/deps", "-i", PIP_INDEX,
                *dependencies,
            ]
            r = subprocess.run(install_cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=90)
            if r.returncode != 0:
                return {
                    "test_output": {
                        "stdout": r.stdout,
                        "stderr": r.stderr,
                        "return_code": r.returncode
                    },
                    "test_result": False
                }

        # ---- 阶段 2：执行测试（无网络，只读）----
        run_cmd = [
            "docker", "run", "--rm",
            "--network", "none",  # ★ 无网络
            "--read-only",  # ★ 只读
            "--tmpfs", "/tmp:size=64m,noexec",
            "--user", "65534:65534",
            "--memory", "512m", "--cpus", "1",
            "--pids-limit", "128",
            "--security-opt", "no-new-privileges",
            "-e", "PYTHONPATH=/deps",
            "-v", f"{volume_name}:/deps:ro",  # 依赖只读挂载
            "-v", f"{temp_dir}:/workspace:ro",
            "-w", "/workspace",
            IMAGE,
            "python", test_file.name,
        ]
        result = subprocess.run(
            run_cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return {
            "test_output": {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            },
            "test_result": result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        return {"test_output": {"stdout": "", "stderr": "Execution timed out", "return_code": -1}, "test_result": False}
    finally:
        # 6. 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        # 清理临时 volume
        subprocess.run(
            ["docker", "volume", "rm", "-f", volume_name],
            capture_output=True,
        )


def transport_test_result(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    将测试结果传输给主图
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 测试结果
    """
    return {"test_result": state.test_result, "test_output": state.test_output,
            "dependencies_messages": state.dependencies_messages}


# ============================ 定义图 ============================

# Define the graph
run_test_graph = (
    StateGraph(State, context_schema=Context)
    .add_node("extract_dependencies", extract_dependencies)
    .add_node("run_test_in_sandbox", run_test_in_sandbox)
    .add_node("transport_test_result", transport_test_result)
    .add_edge("__start__", "extract_dependencies")
    .add_edge("extract_dependencies", "run_test_in_sandbox")
    .add_edge("run_test_in_sandbox", "transport_test_result")
    .add_edge("transport_test_result", "__end__")
    .compile(name="Run Test Graph")
)
