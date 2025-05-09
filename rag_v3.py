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

file = "hospital/output.txt"

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
            full_text = f"科別 : {department}\n科別病症敘述 : {description}"
            docs.append(Document(page_content=full_text))
    
    return docs

# 用法替代原本 loader.load()
split_docs = load_by_department("hospital/output.txt")

embeddings = OllamaEmbeddings(model="nomic-embed-text")#nomic-embed-text#mistral
vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="rag_v3_dir",
    collection_name="rag_v3",
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

system_prompt = "現在開始使用我提供的情境來回答，只能使用繁體中文，不要有簡體中文字。如果你不確定答案，就說不知道。情境如下:\n\n{context}"
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
    response = retrieval_chain.invoke({"input": input_text})
    context = response["context"]

    print(response["answer"])

    input_text = input(">>> ")