import os
from flask import Flask, request
import requests
from openai import OpenAI

app = Flask(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def enviar_mensagem(chat_id, texto):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": texto})

@app.route("/", methods=["GET"])
def inicio():
    return "Bot da Atlas rodando!"

@app.route("/webhook", methods=["POST"])
def webhook():
    dados = request.json

    mensagem = dados.get("message", {})
    chat_id = mensagem.get("chat", {}).get("id")
    texto = mensagem.get("text", "")

    if chat_id and texto:
        resposta = client.responses.create(
            model="gpt-4.1-mini",
            input=f"""
Você é um atendente comercial da Atlas/Virtude Digital.
Você vende tráfego pago e marketing para restaurantes, hamburguerias e delivery.
Responda em português brasileiro, de forma curta, simpática e vendedora.
Sempre tente descobrir o segmento, cidade e objetivo do cliente.

Mensagem do cliente: {texto}
"""
        )

        enviar_mensagem(chat_id, resposta.output_text)

    return "ok"