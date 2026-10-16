# ================================== 模型包 ==================================


# =============================== 导入依赖 ===============================
from langchain_openai import ChatOpenAI
import os




# 需要先加载环境变量





# ============================== 模型实例 ==============================

llm = ChatOpenAI(
    model="deepseek-v4.1-flash", # 模型名称
    api_key=os.getenv("DASHSCOPE_API_KEY"), # 从环境变量中获取 API 密钥
    base_url=os.getenv("BASE_URL"), # 从环境变量中获取 API 基础 URL
    max_retries= 3, # 最大重试次数
    timeout= 120, # 请求超时时间, 单位秒
    model_kwargs={"response_format": {"type": "json_object"}}, # 结构化输出, 要求模型返回 JSON 对象，配合"要求输出json格式"提示词
)



model_dict = {
    "deepseek-v4.1-flash": llm,
}



model_list = [
    "deepseek-v4.1-flash",
]






















