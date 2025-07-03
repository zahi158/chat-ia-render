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

SIMILARITY_THRESHOLD = 0.80

def get_query_embedding(query):
    query = query.replace("\n", " ")
    response = client.embeddings.create(input=[query], model="text-embedding-ada-002")
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

def retrieve_chunks(query, k=5, threshold=SIMILARITY_THRESHOLD):
    embedding = get_query_embedding(query)
    distances, indices = index.search(embedding, k)
    relevant = [i for i, dist in zip(indices[0], distances[0]) if dist <= threshold]
    if not relevant:
        return [], True
    return [chunks[i]["text"] for i in relevant], False

def answer_question(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant that only answers questions strictly related to the course "Gestión del enfado en la infancia".

You have the following context extracted from the course materials:

{context}

If the question is not clearly related to emotional regulation, anger management, or parenting, say: 
"Lo siento, esta pregunta no está relacionada con el curso de gestión del enfado. ¿Quieres que la envíe a un profesional?"

Question: {query}

Answer in Spanish:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
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

# Mapeo: IP -> {"fase": 1, "pregunta": str}
user_states: Dict[str, Dict] = defaultdict(dict)

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("question", "").strip().lower()
    client_ip = req.client.host
    state = user_states.get(client_ip, {})

    # FASE 2: Esperando confirmación de envío al profesional
    if state.get("fase") == 1:
        if query in {"sí", "si"}:
            user_states[client_ip]["fase"] = 2
            return {"answer": "Perfecto. Por favor, dime tu nombre completo para enviar tu pregunta al profesional."}
        else:
            user_states.pop(client_ip, None)
            return {"answer": "De acuerdo 😊 Si tienes otra duda relacionada con el curso, estaré encantado de ayudarte."}

    # FASE 3: Esperando nombre
    if state.get("fase") == 2:
        if len(query.split()) >= 2:
            nombre = query
            pregunta = state["pregunta"]
            enviar_email_profesor(pregunta, nombre)
            user_states.pop(client_ip, None)
            return {
                "answer": f"Gracias {nombre}, hemos enviado tu pregunta a un profesional. "
                          "¿Tienes alguna otra duda relacionada con el curso?"
            }
        else:
            return {"answer": "Por favor, dime tu nombre completo para poder enviar tu pregunta al profesional."}

    # FASE 0: Validar pregunta nueva
    context_chunks, irrelevant = retrieve_chunks(query)
    if irrelevant or not context_chunks or len(context_chunks) < 2:
        user_states[client_ip] = {"fase": 1, "pregunta": query}
        return {
            "answer": "Lo siento, esta pregunta no está relacionada con el curso de gestión del enfado. "
                      "¿Quieres que la envíe a un profesional?"
        }

    # Pregunta válida
    answer = answer_question(query, context_chunks)
    return {"answer": answer}





