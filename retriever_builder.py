import os
import re
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def build_or_load_vectorstore(paths: list[str], embedding_model, persist_directory):
    all_docs = []
    pdf_paths = [
        os.path.join(paths, fname)
        for fname in os.listdir(paths)
        if fname.lower().endswith(".pdf")
    ]

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, 
                                              chunk_overlap=50,
                                              separators=["\n\n", "(?<=\。 )", "\n", " ", ""],
                                              strip_whitespace=True
                                              )
    split_docs = splitter.split_documents(all_docs)

    vectordb = Chroma.from_documents(documents=split_docs, embedding=embedding_model, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


def process_pdfs_to_chunks(paths: List[str],
                           chunk_size: int = 200,
                           chunk_overlap: int = 50) -> List[Document]:
    # 初始化分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "(?<=\。 )", "\n", " ", ""],
        strip_whitespace=True
    )

    all_chunks = []
    pdf_paths = [
        os.path.join(paths, fname)
        for fname in os.listdir(paths)
        if fname.lower().endswith(".pdf")
    ]

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for page in pages:
            chunks = splitter.split_text(page.page_content)

            for chunk in chunks:
                # 清洗：按行 strip，然后再合并为一段文本
                cleaned_chunk_text = re.sub(r'\n\s+', '\n', chunk)
                lines = cleaned_chunk_text.split('\n')
                cleaned_chunk = [line.strip() for line in lines]
                final_page_content = ' '.join(cleaned_chunk)

                if final_page_content.strip():  # 非空才添加
                    all_chunks.append(Document(
                        page_content=final_page_content,
                        metadata={
                            "source": pdf_path,
                            "page": page.metadata.get("page", None)
                        }
                    ))

    return all_chunks


def save_embeddings(all_chunks: List[Document] | None = None,
                    persist_directory: str = './data_base/vector_db/chroma',
                    overwrite=False):
 
    embedding_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-zh")

    if os.path.exists(persist_directory) and not overwrite:
        print("===已存在向量库，跳过保存===")
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    vectorstore.persist()
    print(f"===向量数据库已保存到：{persist_directory}===")

    return vectorstore


def get_hybrid_retriever(all_chunks: List[Document],
                         persist_directory: str = './data_base/vector_db/chroma',
                         k: int = 4) -> EnsembleRetriever:
    # 稠密 embedding
    embedding_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-zh")
    
    # 构建/加载 Chroma
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectorstore.persist()

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 构建稀疏 BM25
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = k

    # 融合
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.4, 0.6]
    )
    return hybrid_retriever


if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh")
    VECTORDB_DIR = "./data_base/vector_db/chroma/"

    pdf_paths = "./pdf_files/"
    all_chunks = process_pdfs_to_chunks(pdf_paths)
    vectorstore = save_embeddings(all_chunks)
    
    # 获取混合检索器
    hybrid_retriever = get_hybrid_retriever(all_chunks)
    print("===混合检索器已准备就绪===")
