from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import faiss
import json
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

# ==== CONFIG ====
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index and chunks
index = faiss.read_index("data/faiss_enfado.index")
with open("data/chunks_enfado.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ==== FUNCIONES ====

SIMILARITY_THRESHOLD = 0.80  # cuanto más bajo, más estricto

def get_query_embedding(query):
    query = query.replace("\n", " ")
    response = client.embeddings.create(input=[query], model="text-embedding-ada-002")
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

def retrieve_chunks(query, k=5, threshold=SIMILARITY_THRESHOLD):
    embedding = get_query_embedding(query)
    distances, indices = index.search(embedding, k)
    print(f"Distancias: {distances[0]}")
    relevant = [i for i, dist in zip(indices[0], distances[0]) if dist <= threshold]
    if not relevant:
        return [], True
    return [chunks[i]["text"] for i in relevant], False

def answer_question(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
Eres una asistente que solo responde preguntas relacionadas con el curso "Gestión del enfado en la infancia".

Usa únicamente este contexto para responder:

{context}

Pregunta: {query}

Respuesta en español:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ==== FASTAPI ====

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
    query = data.get("question", "").strip()

    context_chunks, irrelevant = retrieve_chunks(query)

    if irrelevant or not context_chunks or len(context_chunks) < 2:
        return {
            "answer": "Lo siento, esta pregunta no está relacionada con el curso de gestión del enfado. No puedo ayudarte con eso."
        }

    answer = answer_question(query, context_chunks)
    return {"answer": answer}
