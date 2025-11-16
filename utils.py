import os
import sys
from retriever_builder import process_pdfs_to_chunks, save_embeddings
from conversation_manager import PlannerAgent

# 部署时解压pdf文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.system(f"tar -xzvf /home/user/app/pdf_files.tar.gz")

# 初始化向量数据库
pdf_paths = "../pdf_files"
all_chunks = process_pdfs_to_chunks(pdf_paths)
vectordb = save_embeddings(all_chunks,
                persist_directory='../data_base/vector_db/chroma',
                overwrite=True)  # 是否需要复写（是否有新增）
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 用于缓存 agent 实例（支持多轮）
agent = PlannerAgent(retriever=retriever)

def classify_job_type(job_name: str) -> str:
    job_name = job_name.lower()
    job_name = job_name.split('(')[0]

    # 新业态关键词
    new_economy_jobs = ["外卖", "快递", "网约车", "主播", "骑手", "平台", "直播", "自媒体"]
    # 灵活就业关键词
    flexible_jobs = ["自由", "个体户", "兼职", "临时工", "接单", "顾问", "自由职业者"]
    # 城镇职工关键词（白领/技术类等）
    urban_jobs = ["公司", "企业", "工程师", "职员", "护士", "程序员"]
    # 城乡居民关键词
    rural_jobs = ["农民", "养殖户", "渔民", "果农", "农业工人", "林业工人", "乡村医生"]

    for kw in new_economy_jobs:
        if kw in job_name:
            return "新业态就业"
    for kw in flexible_jobs:
        if kw in job_name:
            return "灵活就业"
    for kw in urban_jobs:
        if kw in job_name:
            return "城镇职工"
    for kw in rural_jobs:
        if kw in job_name:
            return "城乡居民"
    if job_name not in new_economy_jobs and job_name not in flexible_jobs and job_name not in urban_jobs:
        return "其他"
    # 默认值
    return "其他"

def user_asks(message, history, *args):

    user_goal   = args[0] if len(args) > 0 else "未知"
    job_input   = args[1] if len(args) > 1 else "未知"
    situation   = args[2] if len(args) > 2 else "未知"
    city        = args[3] if len(args) > 3 else "未知"
    age         = args[4] if len(args) > 4 else "未知"
    other_info  = args[5] if len(args) > 5 else "未知"

    # 自动分类
    job_type = classify_job_type(job_input)

    user_info = {
        "goal": user_goal,
        "job_info": job_input,
        "job_type": job_type,
        "city": city,
        "age": age,
        "situation": situation,
        "other_info": other_info,
    }

    current_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]

    # 显示用户消息，同时清空输入框
    yield current_history, ""

    full_response = ""
    
    try:
        for chunk in agent.stream_reply(message, user_info):

            print(f"--- Received chunk from agent: '{chunk}' ---")
            full_response += chunk
            current_history[-1]["content"] = full_response
            yield current_history, ""
        print("--- Agent streaming loop finished ---")
    except Exception as e:
        print(f"!!! ERROR: An exception occurred during agent.stream_reply or its iteration: {e}")
        import traceback
        traceback.print_exc()
        current_history[-1]["content"] = f"抱歉，系统内部发生错误，无法生成回复。错误详情：{e}"
        yield current_history, ""
    print("--- Exiting user_asks ---")

