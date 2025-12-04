# from fastapi import FastAPI,UploadFile,File
# import tempfile,re
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import BaseModel
# from pinecone import Pinecone,ServerlessSpec
# import os
# import time
# from dotenv import load_dotenv
# load_dotenv()

# app=FastAPI()
# embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "https://rag-chatbot-hazel.vercel.app", 'http://localhost:5173'  
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#   query: str
# @app.post("/chat")
# async def chat(request: ChatRequest):
#   # qa = RetrievalQA.from_chain_type(llm, retriever=PineconeVectorStore.as_retriever())
#     return "hello"

# @app.get("/health")
# def health():
#     return {'status':'backend connected'}


# @app.post("/upload-pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#   # contents = await file.read()
#   # return {"filename": file.filename, "size": len(contents), "status": "uploaded"}
#       with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             content = await file.read()
#             temp_file.write(content)
#             temp_file_path = temp_file.name
#             # return {"filename": file.filename, "size": len(content), "status": "uploaded"}
#       loader = PyPDFLoader(temp_file_path)
#       docs = loader.load()
#       # return (f"Loaded {len(docs)} pages")
#       # return {
#       #   "content": docs[0].page_content,
#       #   "metadata": docs[0].metadata
#       #   }
#       for doc in docs:
#             doc.page_content = re.sub(r"\[\d+,\s*\d+\]:", "", doc.page_content)
#             doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()

#       text_splitter=RecursiveCharacterTextSplitter(
#           chunk_size=1000,
#           chunk_overlap=200,
#           length_function=len,
#           separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
#       )
#       # all_text = "\n\n".join([doc.page_content for doc in docs])
#       # clean = re.sub(r"\[\d+,\s*\d+\]:", "", all_text)

#       # chunks=text_splitter.split_text(clean)
#       # return chunks
#       chunks = text_splitter.split_documents(docs)
#       # return (f"Created {len(chunks)} chunks")

#       index_name='rag-index'
#       if index_name in pc.list_indexes().names():
#             print("Deleting old index...")
#             pc.delete_index(index_name)
#             time.sleep(5)
      
#             pc.create_index(
#             name=index_name,
#             dimension=768,
#             metric="cosine",
#             spec=ServerlessSpec(
#                 cloud="aws",
#                 region="us-east-1"
#             )
#         )
#       index = pc.Index(index_name)
#       vectorstore = PineconeVectorStore(
#            text_key="text",
#           embedding=embeddings,
#           index_name=index_name,
#           pinecone_api_key=os.getenv("PINECONE_API_KEY")
#         )
#       chunk_texts = [chunk.page_content for chunk in chunks]
#       metadatas = [chunk.metadata for chunk in chunks]
#       ids = vectorstore.add_texts(texts=chunk_texts,
#                                   metadatas=metadatas)
#       stats = index.describe_index_stats()
#       os.unlink(temp_file_path)
#       return {
#             "status": "success",
#             "filename": file.filename,
#             "pages": len(docs),
#             "chunks": len(chunks),
#             "vectors_inserted": stats.total_vector_count,
            
#       }


# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import StreamingResponse
# import tempfile, re, json, asyncio
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import BaseModel
# from pinecone import Pinecone, ServerlessSpec
# import os
# import time
# from dotenv import load_dotenv
# load_dotenv()

# app = FastAPI()
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "https://rag-chatbot-hazel.vercel.app", 'http://localhost:5173'  
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     query: str

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     return "hello"

# @app.get("/health")
# def health():
#     return {'status': 'backend connected'}


# async def process_pdf_stream(file: UploadFile):
#     """Generator function that yields progress updates"""
    
#     try:
#         # Step 1: Loading PDF
#         yield f"data: {json.dumps({'step': 'loading', 'message': '📄 Loading PDF file...'})}\n\n"
        
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             content = await file.read()
#             temp_file.write(content)
#             temp_file_path = temp_file.name
        
#         loader = PyPDFLoader(temp_file_path)
#         docs = loader.load()
        
#         await asyncio.sleep(0.5)  # Small delay for visibility
        
#         # Step 2: Chunking
#         yield f"data: {json.dumps({'step': 'chunking', 'message': '✂️ Chunking document...'})}\n\n"
        
#         for doc in docs:
#             doc.page_content = re.sub(r"\[\d+,\s*\d+\]:", "", doc.page_content)
#             doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
#         )
        
#         chunks = text_splitter.split_documents(docs)
        
#         await asyncio.sleep(0.5)
        
#         # Step 3: Deleting old index
#         yield f"data: {json.dumps({'step': 'deleting', 'message': '🗑️ Clearing previous knowledge base...'})}\n\n"
        
#         index_name = 'rag-index'
#         if index_name in pc.list_indexes().names():
#             pc.delete_index(index_name)
#             time.sleep(5)
        
#         pc.create_index(
#             name=index_name,
#             dimension=768,
#             metric="cosine",
#             spec=ServerlessSpec(
#                 cloud="aws",
#                 region="us-east-1"
#             )
#         )
        
#         await asyncio.sleep(0.5)
        
#         # Step 4: Adding to vector store
#         yield f"data: {json.dumps({'step': 'vectorizing', 'message': '🚀 Adding to vector store...'})}\n\n"
        
#         index = pc.Index(index_name)
#         vectorstore = PineconeVectorStore(
#             text_key="text",
#             embedding=embeddings,
#             index_name=index_name,
#             pinecone_api_key=os.getenv("PINECONE_API_KEY")
#         )
        
#         chunk_texts = [chunk.page_content for chunk in chunks]
#         metadatas = [chunk.metadata for chunk in chunks]
#         ids = vectorstore.add_texts(texts=chunk_texts, metadatas=metadatas)
        
#         stats = index.describe_index_stats()
#         os.unlink(temp_file_path)
        
#         await asyncio.sleep(0.5)
        
#         # Step 5: Complete
#         result = {
#             "step": "complete",
#             "message": "✅ Upload completed successfully!",
#             "data": {
#                 "status": "success",
#                 "filename": file.filename,
#                 "pages": len(docs),
#                 "chunks": len(chunks),
#                 "vectors_inserted": stats.total_vector_count,
#             }
#         }
#         yield f"data: {json.dumps(result)}\n\n"
        
#     except Exception as e:
#         error_result = {
#             "step": "error",
#             "message": "❌ Upload failed!",
#             "error": str(e)
#         }
#         yield f"data: {json.dumps(error_result)}\n\n"


# @app.post("/upload-pdf-stream")
# async def upload_pdf_stream(file: UploadFile = File(...)):
#     """Streaming endpoint that sends progress updates"""
#     return StreamingResponse(
#         process_pdf_stream(file),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#         }
#     )


# Keep the original endpoint for backwards compatibility
# @app.post("/upload-pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#         content = await file.read()
#         temp_file.write(content)
#         temp_file_path = temp_file.name
    
#     loader = PyPDFLoader(temp_file_path)
#     docs = loader.load()
    
#     for doc in docs:
#         doc.page_content = re.sub(r"\[\d+,\s*\d+\]:", "", doc.page_content)
#         doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
#     )
    
#     chunks = text_splitter.split_documents(docs)
    
#     index_name = 'rag-index'
#     if index_name in pc.list_indexes().names():
#         pc.delete_index(index_name)
#         time.sleep(5)
    
#     pc.create_index(
#         name=index_name,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
    
#     index = pc.Index(index_name)
#     vectorstore = PineconeVectorStore(
#         text_key="text",
#         embedding=embeddings,
#         index_name=index_name,
#         pinecone_api_key=os.getenv("PINECONE_API_KEY")
#     )
    
#     chunk_texts = [chunk.page_content for chunk in chunks]
#     metadatas = [chunk.metadata for chunk in chunks]
#     ids = vectorstore.add_texts(texts=chunk_texts, metadatas=metadatas)
    
#     stats = index.describe_index_stats()
#     os.unlink(temp_file_path)
    
#     return {
#         "status": "success",
#         "filename": file.filename,
#         "pages": len(docs),
#         "chunks": len(chunks),
#         "vectors_inserted": stats.total_vector_count,
#     }



from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import tempfile, re, json, asyncio, traceback
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-chatbot-hazel.vercel.app", 'http://localhost:5173'  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    return "hello"

@app.get("/health")
def health():
    return {'status': 'backend connected'}


async def process_pdf_stream(file: UploadFile):
    """Generator function that yields progress updates"""
    temp_file_path = None
    
    try:
        # Step 1: Loading PDF
        yield f"data: {json.dumps({'step': 'loading', 'message': '📄 Loading PDF file...'})}\n\n"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content found in PDF")
        
        await asyncio.sleep(0.5)
        
        # Step 2: Chunking
        yield f"data: {json.dumps({'step': 'chunking', 'message': '✂️ Chunking document...'})}\n\n"
        
        for doc in docs:
            doc.page_content = re.sub(r"\[\d+,\s*\d+\]:", "", doc.page_content)
            doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(docs)
        
        if not chunks:
            raise ValueError("Failed to create chunks from document")
        
        await asyncio.sleep(0.5)
        
        # Step 3: Deleting old index
        yield f"data: {json.dumps({'step': 'deleting', 'message': '🗑️ Clearing previous knowledge base...'})}\n\n"
        
        index_name = 'rag-index'
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            time.sleep(5)
        
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        await asyncio.sleep(0.5)
        
        # Step 4: Adding to vector store
        yield f"data: {json.dumps({'step': 'vectorizing', 'message': '🚀 Adding to vector store...'})}\n\n"
        
        index = pc.Index(index_name)
        vectorstore = PineconeVectorStore(
            text_key="text",
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        chunk_texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = vectorstore.add_texts(texts=chunk_texts, metadatas=metadatas)
        
        stats = index.describe_index_stats()
        
        await asyncio.sleep(0.5)
        
        # Step 5: Complete
        result = {
            "step": "complete",
            "message": "✅ Upload completed successfully!",
            "data": {
                "status": "success",
                "filename": file.filename,
                "pages": len(docs),
                "chunks": len(chunks),
                "vectors_inserted": stats.total_vector_count,
            }
        }
        yield f"data: {json.dumps(result)}\n\n"
        
    except Exception as e:
        # Send detailed error information
        error_message = str(e)
        error_trace = traceback.format_exc()
        
        print(f"Error during PDF processing: {error_trace}")
        
        error_result = {
            "step": "error",
            "message": "Upload failed",
            "error": error_message
        }
        yield f"data: {json.dumps(error_result)}\n\n"
    
    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Failed to delete temp file: {e}")


@app.post("/upload-pdf-stream")
async def upload_pdf_stream(file: UploadFile = File(...)):
    """Streaming endpoint that sends progress updates"""
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        async def error_stream():
            yield f"data: {json.dumps({'step': 'error', 'message': 'Invalid file type', 'error': 'Please upload a PDF file'})}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    return StreamingResponse(
        process_pdf_stream(file),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )