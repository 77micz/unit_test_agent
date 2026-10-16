"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

# ============================= 导入依赖 =============================
from __future__ import annotations

import ast
import importlib.util
# 导入inspect模块
import inspect
import sys
import uuid
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any, Dict, Callable, Optional, List

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from loguru import logger

# ================================ 定义函数元数据结构 ================================

# 函数元数据结构
mt = """
{
    "func_name": "函数名",
    "func_params": [
        {
            "arg_name": "参数1",
            "type": "参数1类型",
        },
        {
            "arg_name": "参数2",
            "type": "参数2类型",
        }
    ],
    "func_return_type": "返回值类型",
    "func_doc": "函数文档",
    "func_code": "函数内容",
}
    """


# ================================ 定义上下文参数 ================================
@dataclass
class Context:
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    # ================ 流程状态 ================
    file_path: Optional[str] = None  # 被测函数所在路径
    func_list: Optional[List[Callable]] = None  # 函数列表
    imports: Optional[List[str]] = field(default_factory=list)  # 被测函数中的import语句列表
    index: Optional[int] = None  # 测试函数索引
    func_metadata_list: Optional[List[Dict[str, Any]]] = None  # 函数元数据列表


# ============================ 定义节点 ============================
def load_funcs(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    加载被测函数
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 函数元数据字典
    """
    logger.info(f"开始加载函数文件：{state.file_path}")

    path = Path(state.file_path).resolve()
    source = path.read_text(encoding="utf-8")

    # 1. 用 AST 解析出顶层函数名，避免依赖函数名已知
    tree = ast.parse(source, filename=str(path))
    func_names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    # 获取所有顶层import语句
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    if len(func_names) == 0:
        raise ValueError(
            f"期望文件中至少有一个顶层函数，实际找到 {len(func_names)} 个"
        )

    # 2. 动态加载模块
    module_name = f"_generated_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 创建模块 spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise ValueError(f"加载函数文件失败：{e}")

    # 3. 获取函数列表
    fn_list = [getattr(module, name) for name in func_names]

    return {"func_list": fn_list}


def extract_imports(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    从 Python 文件中提取所有 import 语句，返回字符串列表。
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 包含 import 语句的字符串列表
    """
    with open(state.file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    imports = []
    # 只遍历顶层节点（模块级别），忽略函数/类内部的 import
    for node in tree.body:
        if isinstance(node, ast.Import):
            # 处理 import xxx, import xxx as yyy
            for alias in node.names:
                if alias.asname:
                    imports.append(f"import {alias.name} as {alias.asname}")
                else:
                    imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # 处理 from xxx import a, b as c
            module = node.module or ""  # 相对导入时 module 可能为 None
            names = []
            for alias in node.names:
                if alias.asname:
                    names.append(f"{alias.name} as {alias.asname}")
                else:
                    names.append(alias.name)
            imports.append(f"from {module} import {', '.join(names)}")
    return {"imports": imports}


def extract_metadata(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    提取函数元数据
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 函数元数据字典
    """
    logger.info(f"开始提取文件：{state.file_path} 中的函数元数据")

    # 从state中获取import语句
    imports = state.imports
    # 将其内容拼接为一个字符串
    imports_str = "\n".join(imports)+"\n\n"

    # 初始化函数元数据列表
    func_metadata_list = []

    # 从state中获取函数对象
    funcs = state.func_list

    for func in funcs:

        logger.info(f"开始提取函数: {func.__name__} 元数据")

        # 从函数对象中提取元数据
        metadata = {}
        if func is None:
            return metadata

        # 获取函数名
        func_name = func.__name__
        metadata["func_name"] = func_name
        # 获取函数参数
        func_params = inspect.signature(func).parameters
        params_info = []
        for param_name, param in func_params.items():
            params_info.append({
                "arg_name": param_name,
                "type": str(param.annotation),
            })
        metadata["func_params"] = params_info
        # 获取函数返回值
        func_return = inspect.signature(func).return_annotation
        metadata["func_return_type"] = str(func_return)
        # 获取函数文档
        func_doc = func.__doc__
        metadata["func_doc"] = func_doc
        # 获取函数内容加入导入语句
        func_code = inspect.getsource(func)
        metadata["func_code"] = imports_str + func_code
        # 添加到元数据列表
        func_metadata_list.append(metadata)

        logger.info(f"成功提取函数: {func_name} 元数据")

    return {"func_metadata_list": func_metadata_list}


def transport_metadata(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    将函数元数据传输给主图
    :param state: 图状态
    :param runtime: 运行时环境
    :return: 函数元数据字典
    """
    return {"func_metadata_list": state.func_metadata_list, "index": 0}


# ============================ 定义图 ============================

# Define the graph
load_funcs_graph = (
    StateGraph(State, context_schema=Context)
    .add_node("load_funcs", load_funcs)
    .add_node("extract_metadata", extract_metadata)
    .add_node("transport_metadata", transport_metadata)
    .add_edge("__start__", "load_funcs")
    .add_edge("load_funcs", "extract_metadata")
    .add_edge("extract_metadata", "transport_metadata")
    .add_edge("transport_metadata", "__end__")
    .compile(name="Load Functions Graph")
)
