from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from typing import Dict
import faiss
import json
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage

# ==== CONFIG ====
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index and chunks
index = faiss.read_index("data/faiss_enfado.index")
with open("data/chunks_enfado.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Memoria temporal: IP → pregunta pendiente
pending_questions: Dict[str, str] = defaultdict(str)

# ==== FUNCIONES ====

def get_query_embedding(query):
    query = query.replace("\n", " ")
    response = client.embeddings.create(input=[query], model="text-embedding-ada-002")
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

def retrieve_chunks(query, k=5, threshold=0.85):
    embedding = get_query_embedding(query)
    distances, indices = index.search(embedding, k)
    if all(dist > threshold for dist in distances[0]):
        return [], True
    return [chunks[i]["text"] for i in indices[0]], False

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

def enviar_email_profesor(pregunta: str, nombre: str):
    remitente = os.getenv("EMAIL_FROM")
    destinatario = os.getenv("EMAIL_TO")
    password = os.getenv("EMAIL_PASSWORD")

    mensaje = EmailMessage()
    mensaje["Subject"] = "📩 Pregunta fuera del curso"
    mensaje["From"] = remitente
    mensaje["To"] = destinatario
    mensaje.set_content(f"""
Un usuario ha hecho una pregunta fuera del curso de gestión del enfado.

Nombre: {nombre}
Pregunta: {pregunta}
""")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(remitente, password)
            smtp.send_message(mensaje)
    except Exception as e:
        print("❌ Error al enviar el email:", e)

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
    client_ip = req.client.host

    # Si el usuario está completando una pregunta pendiente con su nombre
    if pending_questions[client_ip]:
        if len(query.split()) >= 2:  # Detecta nombre completo por número de palabras
            pregunta = pending_questions.pop(client_ip)
            nombre = query
            enviar_email_profesor(pregunta, nombre)
            return {
                "answer": "Gracias, hemos enviado tu pregunta a un profesional. Te responderán lo antes posible."
            }
        else:
            return {
                "answer": "Por favor, indícame tu nombre completo para poder enviar tu pregunta a un profesional."
            }

    # Pregunta normal → validamos
    context_chunks, irrelevant = retrieve_chunks(query)

    if irrelevant or not context_chunks:
        pending_questions[client_ip] = query
        return {
            "answer": "Lo siento, solo puedo responder preguntas relacionadas con el curso de gestión del enfado. "
                      "Por favor, dime tu nombre completo para enviar tu duda a un profesional."
        }

    # Pregunta válida → responder
    answer = answer_question(query, context_chunks)
    return {"answer": answer}
