# unit_test_agent

基于 **LangGraph + 大语言模型** 的智能单元测试生成 Agent：输入一个 Python 源文件，Agent 自动完成 **函数解析 → 用例生成 → 沙箱执行 → 报告产出** 的全流程闭环。

## ✨ 核心特性

- **端到端自动化**：从一个 Python 文件出发，自动解析全部顶层函数、生成单元测试、在隔离沙箱中执行、输出结构化测试报告
- **LangGraph 工作流编排**：主图 + 6 个子图的层级化设计，条件路由支持单文件多函数批量处理
- **结构化 Prompt 工程**：通过 JSON Schema 约束 LLM 稳定输出测试代码与第三方依赖列表
- **安全沙箱执行**：两阶段 Docker 隔离方案——联网装依赖 + 断网执行（`--network none`、只读文件系统、非 root、资源限额）
- **自动评测闭环**：以沙箱执行结果（退出码 / stdout / stderr）作为 Agent 输出的验证信号，形成"生成 → 执行 → 反馈"质量闭环
- **可观测与可调试**：loguru 结构化日志，兼容 LangGraph Studio 可视化调试与 LangSmith Tracing

## 🏗️ 架构设计

```javascript
                        ┌─────────────────────────────────────────────┐
                        │                 主图 (Main Graph)             │
                        └─────────────────────────────────────────────┘

  ┌─────────────┐   ┌──────────────────┐   ┌────────────────┐
  │ load_funcs  │──▶│ generate_unit_test│──▶│ unit_test_save │──▶ __end__
  │ 函数加载子图 │   │   用例生成子图     │   │  用例保存子图   │
  └─────────────┘   └──────────────────┘   └────────────────┘
                             │
                             ▼
                      ┌─────────────┐   ┌──────────────────┐
                      │  run_test   │──▶│ generate_report  │──┬──▶ report_save ──▶ __end__
                      │ 沙箱执行子图 │   │   报告生成子图     │  │
                      └─────────────┘   └──────────────────┘  └──▶ (按 index 条件路由，循环处理下一个函数)
```

### 子图职责

| 子图 | 职责 |
| --- | --- |
| `load_function` | AST 解析源码提取顶层函数与 import 语句，`importlib` 动态加载模块，`inspect` 提取函数签名 / 返回值 / 文档 / 源码，组装结构化元数据 |
| `generate_test` | 基于函数元数据，通过结构化 System Prompt + JSON Schema 调用 LLM 生成单元测试代码与依赖列表 |
| `unit_test_save` | 将生成的测试代码按函数名落盘到 `data/test_functions/` |
| `run_test` | LLM 推断第三方依赖包名；两阶段 Docker 沙箱执行：阶段一联网 `pip install` 到独立 volume，阶段二断网 + 只读 + 资源限额运行测试 |
| `generate_report` | 汇总函数元数据、测试代码、执行输出与通过状态，生成测试报告 |
| `report_save` | 将报告落盘到 `data/reports/` |

### 状态管理

- 主图 `State` 维护流程状态（文件路径、函数元数据列表、测试输出、报告等）与模型消息状态
- 消息字段使用 `add_messages` 归约器自动合并
- 预留 `SqliteSaver` 检查点，支持状态持久化与断点恢复

## 🚀 快速开始

### 环境要求

- Python ≥ 3.11
- Docker（用于沙箱执行）
- 一个大模型 API Key（如 OpenAI 兼容接口）

### 安装

```bash
git clone https://github.com/77micz/unit_test_agent.git
cd unit_test_agent

# 推荐使用 uv
uv sync

# 或 pip
pip install -e . "langgraph-cli[inmem]"
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填入模型 API Key；如需 LangSmith Tracing，填入 LANGSMITH_API_KEY
```

可选模型配置见 `src/model/models.py` 中的 `model_list` / `model_dict`。

### 运行

```bash
# 启动 LangGraph Server（推荐，可用 LangGraph Studio 可视化调试）
langgraph dev
```

在 Studio 中新建 Thread，输入被测文件路径即可运行：

```json
{"file_path": "/path/to/your/functions.py"}
```

也可以在代码中直接调用：

```python
from src.agent.graph import graph

result = graph.invoke({"file_path": "tests/test_funcs/example.py"})
```

运行产物：

- 生成的单元测试：`data/test_functions/`
- 测试报告：`data/reports/`

## 📁 项目结构

```javascript
unit_test_agent/
├── src/
│   ├── agent/
│   │   ├── graph.py            # 主图：工作流编排与条件路由
│   │   ├── load_function.py    # 函数加载子图（AST + inspect 元数据提取）
│   │   ├── generate_test.py    # 用例生成子图（LLM 结构化输出）
│   │   ├── run_test.py         # 沙箱执行子图（依赖推断 + Docker 隔离执行）
│   │   ├── generate_report.py  # 报告生成子图
│   │   ├── unit_test_save.py   # 用例保存子图
│   │   └── report_save.py      # 报告保存子图
│   └── model/
│       └── models.py           # 模型注册与配置
├── tests/                      # 单元测试与集成测试
├── langgraph.json              # LangGraph 部署配置
├── pyproject.toml
└── Makefile
```

## 🔒 沙箱安全设计

`run_test` 子图采用两阶段隔离执行：

| 阶段 | 网络 | 文件系统 | 资源限制 |
| --- | --- | --- | --- |
| ① 依赖安装 | 联网（bridge） | 独立 Docker volume | 512m 内存 / 1 CPU |
| ② 测试执行 | **断网**（`--network none`） | **只读**挂载 + tmpfs | 512m 内存 / 1 CPU / PID 128 / 非 root |

- 测试代码与依赖均以只读方式挂载，执行环境不可写
- 执行超时控制（默认 120s），stdout / stderr / 退出码结构化回收

## 🧪 测试

```bash
make test          # 运行全部测试
make unit_tests    # 单元测试
make integration_tests  # 集成测试（真实调用 LLM）
```

## 🗺️ Roadmap

- [ ] 失败用例反馈注入：将执行失败信息回传给生成节点，实现自我修正（Reflection）
- [ ] 多 Agent 协作：生成 Agent + 评审 Agent 分工
- [ ] 基于覆盖率（coverage）的用例质量评估
- [ ] 启用 SqliteSaver 检查点，支持断点续跑

## 📄 License

MIT
