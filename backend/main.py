from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=['https://rag-chatbot-hazel.vercel.app'])

@app.get("/health")
def health():
    return {'status':'backend connected'}