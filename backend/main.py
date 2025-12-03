from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=['https://rag-chatbot-hfrrudhll-dinesh-kumars-projects-8e3489ae.vercel.app'])

@app.get("/health")
def health():
    return {'status':'backend connected'}