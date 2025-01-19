import os
import json
from typing import List, Dict
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_callback_manager
from colpali.colpali import ColPali

def load_and_preprocess_documents(
  file_paths: List[str], text_splitter: RecursiveCharacterTextSplitter,
) -> List[Dict]:
  """
  Загружает документы и разбивает текст на куски.
  """
  documents = []
  for file_path in file_paths:
    loader = UnstructuredFileLoader(file_path)
    loaded_docs = loader.load()
    for doc in loaded_docs:
      chunks = text_splitter.split_text(doc.page_content)
      for chunk in chunks:
        documents.append({"text": chunk, "metadata": {"source": file_path}})
  return documents

def create_vector_store(documents: List[Dict], embedding_function) -> FAISS:
  """
  Создает векторный магазин с использованием FAISS.
  """
  return FAISS.from_documents(documents, embedding_function)

def create_retrieval_qa_chain(vectorstore: FAISS, llm: OpenAI) -> RetrievalQA:
  """
  Создает цепочку поиска ответов с использованием RAG.
  """
  retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
  return RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
  )

def query_with_colpali(
  query: str,
  qa_chain: RetrievalQA,
  colpali_model: ColPali,
  top_k: int = 5,
) -> str:
  """
  Использует ColPali для уточнения запроса, затем использует RAG.
  """
  refined_query = colpali_model.refine_query(query, top_k=top_k)
  return qa_chain.run(refined_query)

def main():
  """
  Основной код для конвейера.
  """
  # Загрузка документов
  file_paths = [
    "document1.pdf",
    "document2.txt",
    "document3.docx",
  ]
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  documents = load_and_preprocess_documents(file_paths, text_splitter)

  # Создание векторного магазина
  embeddings = OpenAIEmbeddings()
  vectorstore = create_vector_store(documents, embeddings)

  # Создание цепочки поиска ответов
  llm = OpenAI(temperature=0.7)
  qa_chain = create_retrieval_qa_chain(vectorstore, llm)

  # Загрузка модели ColPali
  colpali_model = ColPali()

  # Запрос
  query = "Какая выручка компании в 2022 году?"
  answer = query_with_colpali(query, qa_chain, colpali_model)
  print(f"Ответ: {answer}")

if __name__ == "__main__":
  main()
