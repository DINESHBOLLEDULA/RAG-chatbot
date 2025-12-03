from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-chatbot-hazel.vercel.app",   # ⬅ NO trailing slash
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173'])

@app.get("/health")
def health():
    return {'status':'backend connected'}

@app.get("/upload-pdf")
async def upload_pdf():
    return {'status':'pdf uploaded'}