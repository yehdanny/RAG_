from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain



llm = Ollama(model="mistral")
loader = TextLoader("rag_v2/output.txt", encoding="utf-8")
'''
這裡使用了 Langchain 社群提供的第三方套件來建立 LLM 物件，
同時例用 TextLoader 來讀取 output.txt 的內容，
就不用像我們前面自己手動呼叫 open() 函數讀取檔案。
接著，我們來把這些文字分割成小段落：
'''

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

'''
這裡借用了 RecursiveCharacterTextSplitter 來把文字分割成小段落
'''

splited_docs = text_splitter.split_documents(loader.load())

'''把這些段落轉換成向量：'''
#embeddings = OllamaEmbeddings(model="nomic-embed-text")
embeddings = OllamaEmbeddings(model="mistral")
'''nomic-embed-text 模型是因為用這個 Model 來計算速度比較快'''

vector_db = Chroma.from_documents(
    documents=splited_docs,
    embedding=embeddings,
    persist_directory="rag_v2_dir",
    collection_name="embeddings",
)
'''
我們用 Chroma 來幫我們存放這些向量。
persist_directory 是用來指定存放向量的目錄，
collection_name 則是用來指定這些向量的集合名稱。
'''




retriever = vector_db.as_retriever(search_kwargs={"k": 3})

system_prompt = "現在開始使用我提供的情境來回答，只能使用繁體中文，不要有簡體中文字。如果你不確定答案，就說不知道。情境如下:\n\n{context}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "問題: {input}"),
    ]
)


#建立聊天的 prompt
document_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input("您想問什麼問題？\n>>> ")

while input_text.lower() != "bye":
    response = retrieval_chain.invoke({"input": input_text, "context": context})
    context = response["context"]

    print(response["answer"])

    input_text = input(">>> ")