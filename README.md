# InsureGrad-RAG
本项目构建了⼀个⾯向应届⼤学⽣的五险⼀⾦社会保障智能规划助⼿。针对应届⼤学⽣⼊职前后社保知识空⽩、缺乏规划经验的痛点，使⽤RAG（检索增强⽣成）技术，实现了结合⽤⼾画像、政策数据⽣成智能问答、政策科普和⻛险规避等结构化规划建议的功能。

This project builds an AI assistant for *Social Insurances and Housing Fund* planning aimed at fresh university graduates. The RAG (Retrieval-Augmented Generation) system is utilized to address the critical pain points (knowledge gaps and lack of planning experience) faced by fresh university graduates during their initial employment. By combining user profiles with policy data, it generates structured recommendations encompassing intelligent Q&A, policy education, and risk avoidance guidance.


## Project Structure

```text
├─InsureGrad-RAG
│  ├─get_client.py            # Encapsulates the client connection to the Spark API
│  ├─retriever_builder.py     # Builds multi-source retrievers
│  ├─rag_chain.py             # Defines the RAG pipeline (retrieval, re-ranking, context concatenation)
│  ├─conversation_manager.py  # Manages multi-turn conversation history and context memory
│  ├─utils.py                 # Utility functions (used in app.py)
│  ├─app.py                   # Main entry for the Gradio application (chat interface + inference calls)
│  ├─requirements.txt         # Python project dependencies
│  └─run.sh                   # Startup script (runs the app)
```

## Environment Requirements

### Operating System
- Ubuntu 20.04.6 LTS

### Python Environment
- Python 3.10.10

### Required Python Packages
```
gradio==5.38.0
langchain==0.3.26
langchain-community==0.3.27
langchain-core==0.3.68
transformers==4.53.2
torch==2.7.0+cu128
pypdf==5.7.0
```

## Demo 

🔗 Online Demo: https://huggingface.co/spaces/toro0706/my-rag-app
