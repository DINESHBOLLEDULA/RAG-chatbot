from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173",'https://rag-chatbot-git-main-dinesh-kumars-projects-8e3489ae.vercel.app/'])

@app.get("/health")
def health():
    return {'status':'backend connected'}