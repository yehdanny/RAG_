from langchain_community.llms import Ollama
#from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document

llm = Ollama(model="mistral")

file = "rag_v3/output.txt"

def load_by_department(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 每個科別段落以空行區隔
    entries = content.strip().split('\n\n')
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 2:
            department = lines[0].replace("科別 : ", "")
            description = lines[1].replace("科別病症敘述 : ", "")
            full_text = f"這是醫院的 : {department} ，其部門醫生或常見病症有 : {description}"
            docs.append(Document(page_content=full_text))
    #for doc in docs:
    #    print(f'c = {doc.page_content}')
    return docs

# 用法替代原本 loader.load()
split_docs = load_by_department("rag_v3/output.txt")


embeddings = OllamaEmbeddings(model="nomic-embed-text")#nomic-embed-text#mistral
vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="rag_v3_dir",
    collection_name="rag_v3",
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

system_prompt = """
你是一位醫院導診助理，負責根據提供的科別與病症對應資料，協助病人判斷該前往哪一個科別就診。

請根據以下提供的參考內容回答問題：
- 所有回答必須使用繁體中文。
- 不可以出現簡體中文字或非正體中文用語。
- 不可以出現任何醫療建議或診斷。
- 回答中不得使用資料中未提及的醫療術語或專業詞彙。
- 只能根據我提供的情境資料（{context}）回答問題，不可以加入你自己的臆測或知識。
- 如果你無法從提供的內容中找出明確答案，請回應「根據目前資料無法判斷，建議聯絡醫師進一步諮詢。」
- 若有多個科別符合病症，請列出科別。

以下是參考資料：
{context}
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "問題: {input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input("您想問什麼問題？\n>>> ")

while input_text.lower() != "bye":
    #response = retrieval_chain.invoke({"input": input_text, "context": context})
    #response = retrieval_chain.invoke({"input": input_text})
    #context = response["context"]
    res = retrieval_chain.invoke({"input": input_text}, return_only_outputs=False)
    print("\n=== 檢索到的 Context ===")
    print(res["context"])
    print("\n=== 模型回答 ===")
    print(res["answer"])
    input_text = input(">>> ")