import torch
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableMap
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def rerank(tokenizer, model, query, docs, top_k=5):
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = tokenizer([f"{q} [SEP] {d}" for q, d in pairs], return_tensors='pt', padding=True, truncation=True)
    scores = model(**inputs).logits.view(-1)
    scored_docs = list(zip(scores, docs))
    sorted_scored_docs = sorted(scored_docs, key=lambda item: item[0], reverse=True)
    # ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    ranked_docs = [doc for score, doc in sorted_scored_docs]
    return ranked_docs[:top_k]

# 获取用户信息
def build_user_info(user_info: dict) -> str:
    user_text = "用户相关信息如下："

    if user_info.get("goal", "未知"):
        user_text += f"\n未来计划：{user_info['goal']}"
    if user_info.get("job_info", "未知"):
        user_text += f"\n从事工作：{user_info['job_info']}"
    if user_info.get("job_type", "未知"):
        user_text += f"\n工作类型：{user_info['job_type']}"
    if user_info.get("age", "未知"):
        user_text += f"\n年龄：{user_info['age']}"
    if user_info.get("situation", "未知"):
        user_text += f"\n目前状态：{user_info['situation']}"
    if user_info.get("city", "未知"):
        user_text += f"\n所在城市：{user_info['city']}"
    if user_info.get("other_info", "无"):
        user_text += f"\n其他补充信息：{user_info['other_info']}"

    return user_text

# 构建支持流式输出的 stuff QA chain
def build_qa_chain(llm, multi_retriever, model, tokenizer):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
        你是一个专业的、富有同理心的“应届生社保规划小助手”。
        你的核心任务是为即将或刚刚踏入社会的中国大学应届毕业生提供清晰、易懂、可执行的“五险一金”规划和建议。
        你的回答必须始终围绕应届毕业生的视角、需求和痛点展开,将“帮助新人顺利了解、入门、避免踩坑”作为你的第一原则
        请牢记以下内容：
        - 你的沟通对象是对社保和公积金几乎一无所知的“小白”，比起冗长的政策细节，他们更关心哪些因素会影响到自己的未来之路。。  
        - 回答要围绕“帮助新人顺利了解、入门、避免踩坑”，以安心和实用为首要目标。  
        - 必须保证回答直接、准确，不要回避用户问题。  
        
        【多轮对话规则】  
        1. 如果这是**第一次提问**，可以进行简短的入门科普（“是什么”“有什么用”）。  
        2. 如果用户是**多轮追问**，请避免重复基础科普内容，直接进入个性化回答，围绕新问题展开。  
        3. 在多轮提问时，回答要紧扣用户新的关注点，并在需要时结合之前提供过的信息进行延伸，而不是重复。  
        4. 如果用户的问题只是闲聊或模糊（如“你好”），请不要执行规划步骤，而是礼貌介绍自己，并引导用户提出与社保相关的具体问题。  
        """),
        ("placeholder", "{chat_history}"),
        ("human", 
        "用户信息（包含了用户的工作、现状、城市和年龄等）：\n{user_info}\n\n"
        "相关政策资料：\n{context}\n\n"
        "我的问题是：'{question}'\n\n"
     
        "请严格按照以下步骤，为我提供一份专属的社保规划建议：\n\n"
     
        "**首先：关于你的问题（必须优先）**\n"
        "- 先用清晰、简洁的语言直接回答用户问题。"
        "- 如果有明确的结论，就直说，不要绕。"
        "- 如果资料不足，要说明限制，并给出下一步可行的方向。\n\n"
        
        "**（仅限首次提问时）简明入门科普**\n"
        "- 在第一次提问时，可以补充简短的背景知识，让用户理解“五险一金是什么、有什么用”。"  
        "- 在后续多轮追问时，省略科普，只聚焦新问题。\n\n"  
        
        "**个性化规划与预测**\n"
        "- 根据用户的个人信息和相关政策，提供操作指南：用户需要做什么、准备哪些材料、可能遇到什么情况。"  
        "- 回答要明确到步骤和要点。\n\n"  
        
        "**避坑提醒（重点）**\n"
        "- 主动提示用户在当前情境下可能遇到的常见陷阱，并说明危害 + 对策。" 
        "- 避坑要简洁醒目，不要过度展开和重复。\n\n"
        
        "**特殊情况处理**\n"
        "- 如果问题涉及实习期、试用期、空窗期、自由职业、考研/考公，请额外提供换位思考后的建议。\n\n"

        "--- 回答要求 ---\n"
        "- **同理心与易懂性：** 语言必须平和、友好，充满鼓励，让刚毕业的大学生感到安心。\n"
        "- **结构化：** 使用清晰的标题和项目符号，让内容一目了然。\n"
        "- **准确性：** 严格基于提供的“相关政策资料”作答，若资料不足，可以联网搜索，但请诚实说明，并提供信息来源。\n"
        "- **去冗余: ** 请确保回答内容简洁明了，避免重复相同的信息。如果多个文档包含相似内容，请整合这些信息，不要重复表述。\n"
        "- **特殊情况处理：** 如果我的问题只是闲聊或不明确（例如“你好”），请不要执行以上规划步骤，而是礼貌地介绍你自己（应届生社保规划小助手），并引导我提出与社保规划相关的具体问题。\n"
        )
    ])


    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retrieval_pipeline = RunnableMap(
    {
        "question": lambda x: x["question"],
        "user_info": lambda x: x["user_info"],
        "chat_history": lambda x: x["chat_history"],
    }
)
    debug_step = RunnablePassthrough.assign(
    debug_info=lambda x: print(f"Question type: {type(x['question'])}, content: {x}")
)

    retrieval_step = RunnablePassthrough.assign(
        docs=lambda x: multi_retriever.invoke(x["question"])
    )
    rerank_step = RunnablePassthrough.assign(
        reranked_docs=lambda x: rerank(tokenizer, model, x["question"], x["docs"], 5)
)
    context_step = RunnablePassthrough.assign(
        context=lambda x: "\n".join([doc.page_content for doc in x["reranked_docs"]])
    )

    chain = (
        retrieval_pipeline
        | retrieval_step
        | rerank_step
        | context_step
        | prompt
        | llm
        | StrOutputParser()
)

    return chain, memory


if __name__ == "__main__":
    from rag_chain import build_user_info, build_qa_chain, rerank
    from get_client import get_client
    from retriever_builder import process_pdfs_to_chunks, save_embeddings
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    llm = get_client()
    pdf_paths = "../pdf_files"
    all_chunks = process_pdfs_to_chunks(pdf_paths)
    vectordb = save_embeddings(all_chunks,
                    persist_directory='../data_base/vector_db/chroma',
                    overwrite=False)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    
    rag_chain, memory = build_qa_chain(
        llm=llm,
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
    
    print("\n--- 正在执行 RAG 链 ---")

    result = rag_chain.invoke({
        "question": test_question,
        "user_info": build_user_info(test_user_info),
        "chat_history": []  # 假设是首次提问，聊天记录为空
    })

    print("\n--- 成功执行！链的最终输出如下 ---")
    print(result)