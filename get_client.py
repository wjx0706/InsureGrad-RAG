import os
from dotenv import find_dotenv, load_dotenv
from typing import Optional, Any
from langchain_community.chat_models import ChatSparkLLM
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv(find_dotenv())

class ChunkPrintHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, *, chunk: Optional[Any] = None, **kwargs: Any):
        print(token, end="", flush=True)
        debug = False
        if debug:
            print("\n", kwargs)

def get_client():
    app_id = os.getenv("IFLYTEK_SPARK_APP_ID")
    api_key = os.getenv("IFLYTEK_SPARK_API_KEY")
    api_secret = os.getenv("IFLYTEK_SPARK_API_SECRET")

    if not all([app_id, api_key, api_secret]):
        raise ValueError("请确保环境变量已设置。")

    llm = ChatSparkLLM(
            model='Spark4.0 Ultra',
            app_id=app_id,
            api_key=api_key,
            api_secret=api_secret,
            spark_api_url="wss://spark-api.xf-yun.com/v4.0/chat",
            spark_llm_domain="4.0Ultra",
            streaming=True,
            callbacks=[ChunkPrintHandler()]
        )
    
    return llm

if __name__ == "__main__":
    client = get_client()
    print("Client initialized successfully.")
    # 测试模型
    response = client.invoke("你好，介绍一下你自己。")
    print("\nResponse:", response)