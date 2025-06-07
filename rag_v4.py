from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.llms import Ollama

def load_medical_docs(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = content.strip().split('\n\n')
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 2:
            department = lines[0].replace("科別 : ", "")
            description = lines[1].replace("科別病症敘述 : ", "")
            docs.append(Document(page_content=description, metadata={"科別": department}))
    return docs

# 載入與處理資料
raw_docs = load_medical_docs("rag_v3/cleaned_output_for_RAG.txt")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(raw_docs)

# 向量資料庫與檢索器
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="rag_v4_dir")
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# LLM 與 Prompt
llm = Ollama(model="mistral")
system_prompt = """
你是一位醫院導診助理，根據病人的敘述協助他們選擇合適的科別。
請使用繁體中文作答，避免醫療建議，僅根據提供資料回答。
若無法判斷，請回答：「根據目前資料無法判斷，建議聯絡醫師進一步諮詢。」
"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "問題: {input}")])
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# 測試用輸入


context = []
input_text = input("請問您有什麼明顯症狀？\n>>> ")

while input_text.lower() != "bye":
    res = chain.invoke({"input": input_text}, return_only_outputs=False)
    print("\n=== 檢索到的 Context ===")
    print(res["context"])
    print("\n=== 模型回答 ===")
    print(res["answer"])
    input_text = input(">>> ")