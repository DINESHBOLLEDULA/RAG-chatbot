from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173",'https://rag-chatbot-1-urn7.onrender.com'])

@app.get("/health")
def health():
    return {'status':'backend connected'}