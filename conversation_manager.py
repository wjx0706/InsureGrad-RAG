import os
import contextlib
from typing import Optional, Dict, List
from langchain_core.runnables.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.retrievers import MultiQueryRetriever
from rag_chain import build_user_info, build_qa_chain, rerank
from get_client import get_client
from langchain.callbacks.base import BaseCallbackHandler

class SilentMultiQueryRetriever(MultiQueryRetriever):
    def invoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs):
        # 静默执行多查询生成，不输出任何内容
        with disable_stdout():
            return super().invoke(input, config, **kwargs)
@contextlib.contextmanager
def disable_stdout():
    with open(os.devnull, "w") as f:
        old_stdout = os.dup(1)
        os.dup2(f.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            
class StreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)

    def get_response(self):
        return "".join(self.tokens)

class PlannerAgent:
    def __init__(self, retriever):
        self.llm = get_client()
        self.retriever = retriever
        self.multi_retriever = SilentMultiQueryRetriever.from_llm(
                            retriever=self.retriever,
                            llm=self.llm)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
        # self._should_stop = True # 新增：中断标志
        # print("PlannerAgent initialized.")

    # def stop_generation(self):
    #     """设置中断标志，用于停止 stream_reply 的生成"""
    #     self._should_stop = True
    #     print("PlannerAgent: 设置中断标志为 True")

    # def reset_stop_flag(self):
    #     """重置中断标志，以便下次生成能够正常开始"""
    #     self._should_stop = False
    #     print("PlannerAgent: 重置中断标志为 False")

    def get_history(self) -> List[Dict[str, str]]:
        result = []
        for msg in self.memory.chat_memory.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            result.append({"role": role, "content": msg.content})
        return result

    def reply(self, user_question: str, user_info_dict: Dict) -> str:
        qa_chain, _ = build_qa_chain(self.llm, 
                self.multi_retriever, 
                self.tokenizer,
                self.model)

        user_info_text = build_user_info(user_info_dict)
        docs = self.retriever.get_relevant_documents(user_question)
        reranked_docs = rerank(self.tokenizer, self.model, user_question, docs, 5)
        context = "\n".join([doc.page_content for doc in reranked_docs])

        self.memory.chat_memory.add_user_message(user_question)
        response = qa_chain.invoke({
            "user_info": user_info_text,
            "context": context
        })
        self.memory.chat_memory.add_ai_message(response)
        return response

    def stream_reply(self, user_question: str, user_info_dict: Dict) -> str:
        # print(f"\n--- Entering stream_reply ---")
        # print(f"Message received by agent: {user_question}")
        # print(f"User info received by agent: {user_info_dict}")
        # self.reset_stop_flag()

        qa_chain, _ = build_qa_chain(llm=self.llm,
                                    multi_retriever=self.multi_retriever, 
                                    model=self.model, 
                                    tokenizer=self.tokenizer)

        user_info_text = build_user_info(user_info_dict)
        current_chat_history = self.memory.buffer_as_messages

        full_response_content = "" # 用于累积完整响应以便存入 memory

        # 使用 qa_chain.stream() 进行流式处理
        try:
            for chunk in qa_chain.stream(
                {
                    "chat_history": current_chat_history,
                    "user_info": user_info_text,
                    "question": user_question
                }
            ):
                if isinstance(chunk, str):
                    content_to_yield = chunk
                elif hasattr(chunk, 'content') and isinstance(chunk.content, str):
                    content_to_yield = chunk.content
                else:
                    continue

                if content_to_yield:
                    full_response_content += content_to_yield
                    yield content_to_yield # 每次 yield 实时生成的令牌或小块

                # 只有在没有被中断的情况下才将完整响应添加到内存
            self.memory.save_context({"question": user_question}, {"answer": full_response_content})
            print(f"--- 响应已保存到内存: {full_response_content[:50]}... ---")

        except KeyboardInterrupt:
            print("PlannerAgent: 捕获到 KeyboardInterrupt，停止生成且不保存到内存。")
            pass 
        except Exception as e:
            print(f"错误：流式处理过程中发生异常：{e}")
            yield f"发生错误：{e}"
            print("--- 发生错误，不保存到内存。 ---")

        self.memory.chat_memory.add_user_message(user_question)
        self.memory.chat_memory.add_ai_message(full_response_content)


if __name__ == "__main__":
    from get_client import *
    from retriever_builder import process_pdfs_to_chunks, save_embeddings

    pdf_paths = "/teamspace/studios/this_studio/pdf_files"
    all_chunks = process_pdfs_to_chunks(pdf_paths)
    vectordb = save_embeddings(all_chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    
    rag_chain, memory = build_qa_chain(
        llm=get_client(),
        multi_retriever=retriever,
        model=model,
        tokenizer=tokenizer
    )

    test_user_info = {
        "job_info": "软件工程师",
        "situation": "已入职试用期",
        "city": "上海",
        "age": 26,
    }
    test_question = "试用期公司会给我交社保和公积金吗？"
    agent = PlannerAgent(retriever=retriever)

    # 流式输出测试（如果启用流式 Spark 回调）
    print("\n流式输出测试：")
    for chunk in agent.stream_reply(test_question, test_user_info):
        # print(chunk, end="", flush=True)
        pass