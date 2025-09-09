from fastapi import FastAPI, File, UploadFile
import uvicorn
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from fastapi import FastAPI, File, UploadFile
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

@app.get("/")
def home():
    return {"message": "Hello, World!"} 


@app.get("/health")
def health_check():
    return {"message": "Health check successful"}


@app.post("/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    file_content = await file.read()
    file_name = file.filename
    file_type = file.content_type
    file_size = file.size
    file_path = f"uploads/{"_".join(file_name.split())}.pdf"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    vector_store.add_documents(docs)
    vector_store.save_local(f"faiss_{"_".join(file_name.split())}")
    return {"message": "File uploaded successfully", "file_name": file_name, "file_type": file_type, "file_size": file_size, "file_path": file_path}

@app.get("/documents")
def list_documents():
    documents = []
    for file in os.listdir("uploads"):
        documents.append({"name": file, "size": os.path.getsize(f"uploads/{file}"), "type": file.split(".")[-1]})
    return {"message": "Documents listed successfully", "documents": documents} 

@app.post("/query")
def query(query: str):
    docs = vector_store.similarity_search(query)
    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)
    print(query)
    print("\n\n\n")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer questions about the documents uploaded."),
        ("human", "Here are the documents uploaded: {documents}\nHere is the query: {query}")
    ])

    response = llm.invoke(prompt.invoke({
        "documents": "\n".join([doc.page_content for doc in docs]),
        "query": query
    }))
    return {"message": "Query successful", "query": query, "response": response.content}

if __name__ == "__main__":
    uvicorn.run(app='main:app', host="0.0.0.0", port=8000, reload=True)