from fastapi import FastAPI, Request
import faiss
import json
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index and chunks
index = faiss.read_index("data/faiss_enfado.index")
with open("data/chunks_enfado.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def get_query_embedding(query):
    query = query.replace("\n", " ")
    response = client.embeddings.create(input=[query], model="text-embedding-ada-002")
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

def retrieve_chunks(query, k=5):
    embedding = get_query_embedding(query)
    distances, indices = index.search(embedding, k)
    return [chunks[i]["text"] for i in indices[0]]

def answer_question(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are an assistant for course content. Use the context to answer the question.

Context:
{context}

Question: {query}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("question", "")
    chunks = retrieve_chunks(query)
    answer = answer_question(query, chunks)
    return {"answer": answer}

