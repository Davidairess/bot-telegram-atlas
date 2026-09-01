import logging
import os
import sqlite3
import tempfile
import threading
from typing import Dict, List

import requests
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("telegram_personal_assistant")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DB_PATH = os.getenv(
    "CONVERSATION_DB_PATH",
    os.path.join(tempfile.gettempdir(), "bot_telegram_atlas_memory.sqlite3"),
)

MAX_STORED_MESSAGES_PER_CHAT = 60
MAX_CONTEXT_MESSAGES_PER_REQUEST = 40
MAX_CONTEXT_CHARACTERS = 12000
TELEGRAM_MESSAGE_LIMIT = 4000
OPENAI_TIMEOUT_SECONDS = 45
TELEGRAM_TIMEOUT_SECONDS = 15

FALLBACK_ERROR_MESSAGE = (
    "Tive um problema pra processar sua mensagem agora. "
    "Tenta de novo em alguns instantes."
)

SYSTEM_PROMPT = (
    "Voce e meu assistente pessoal no Telegram.\n"
    "Responda sempre em portugues brasileiro, com tom natural, direto e informal.\n"
    "Nao tente vender servicos, nao fale como atendente comercial e nao trate o usuario como lead ou cliente.\n"
    "Ajude com trafego pago, Meta Ads, copies, anuncios, atendimento de clientes, prospeccao, "
    "estrategias para restaurantes/delivery, desenvolvimento de sites, sistemas, programacao, debugging, "
    "organizacao de projetos e duvidas gerais.\n"
    "Quando o usuario pedir uma mensagem para cliente, entregue o texto pronto para copiar e colar.\n"
    "Quando o usuario pedir ajuda tecnica, explique passo a passo de forma simples, assumindo que ele nao e programador.\n"
    "Se faltar contexto, peca so o minimo necessario.\n"
)

DB_LOCK = threading.Lock()
client = (
    OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS)
    if OPENAI_API_KEY
    else None
)


def init_db() -> None:
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_messages_chat_id_id "
            "ON conversation_messages(chat_id, id)"
        )


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def store_message(chat_id: str, role: str, content: str) -> None:
    with DB_LOCK:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversation_messages (chat_id, role, content, created_at)
                VALUES (?, ?, ?, strftime('%s', 'now'))
                """,
                (chat_id, role, content),
            )
            conn.execute(
                """
                DELETE FROM conversation_messages
                WHERE id IN (
                    SELECT id
                    FROM conversation_messages
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (chat_id, MAX_STORED_MESSAGES_PER_CHAT),
            )


def trim_messages_to_char_budget(
    messages: List[Dict[str, str]], max_characters: int
) -> List[Dict[str, str]]:
    if not messages:
        return []

    selected: List[Dict[str, str]] = []
    total_characters = 0

    for message in reversed(messages):
        content = message.get("content", "")
        content_size = len(content)
        if selected and total_characters + content_size > max_characters:
            break
        selected.append(message)
        total_characters += content_size

    return list(reversed(selected))


def load_recent_messages(chat_id: str) -> List[Dict[str, str]]:
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT role, content
            FROM conversation_messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, MAX_CONTEXT_MESSAGES_PER_REQUEST),
        )
        rows = cursor.fetchall()

    messages = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    return trim_messages_to_char_budget(messages, MAX_CONTEXT_CHARACTERS)


def build_openai_messages(chat_id: str, user_text: str) -> List[Dict[str, str]]:
    recent_messages = load_recent_messages(chat_id)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent_messages)
    messages.append({"role": "user", "content": user_text})
    return messages


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> List[str]:
    cleaned_text = text.strip()
    if len(cleaned_text) <= limit:
        return [cleaned_text]

    chunks: List[str] = []
    remaining = cleaned_text

    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def send_telegram_message(chat_id: int, text: str) -> bool:
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN nao configurado; nao foi possivel enviar mensagem.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payloads = split_message(text)

    try:
        for payload in payloads:
            response = requests.post(
                url,
                json={"chat_id": chat_id, "text": payload},
                timeout=TELEGRAM_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        return True
    except requests.RequestException:
        logger.exception("Falha ao enviar mensagem para o Telegram no chat_id=%s", chat_id)
        return False


def generate_openai_reply(chat_id: str, user_text: str) -> str:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY nao configurada.")

    messages = build_openai_messages(chat_id, user_text)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )

    choice = response.choices[0] if response.choices else None
    assistant_text = ""
    if choice and getattr(choice, "message", None):
        assistant_text = (choice.message.content or "").strip()

    if not assistant_text:
        raise RuntimeError("A OpenAI retornou uma resposta vazia.")

    return assistant_text


def handle_message(chat_id: int, user_text: str) -> None:
    chat_key = str(chat_id)

    logger.info(
        "Mensagem recebida do Telegram: chat_id=%s, tamanho=%s",
        chat_id,
        len(user_text),
    )

    store_message(chat_key, "user", user_text)

    try:
        assistant_text = generate_openai_reply(chat_key, user_text)
        store_message(chat_key, "assistant", assistant_text)
        send_telegram_message(chat_id, assistant_text)
    except Exception:
        logger.exception("Erro ao processar mensagem no chat_id=%s", chat_id)
        send_telegram_message(chat_id, FALLBACK_ERROR_MESSAGE)


@app.route("/", methods=["GET"])
def inicio():
    return "Bot pessoal rodando!"


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        dados = request.get_json(silent=True) or {}

        mensagem = dados.get("message") or dados.get("edited_message") or {}
        chat_id = mensagem.get("chat", {}).get("id")
        texto = mensagem.get("text") or ""

        if chat_id is not None and texto:
            handle_message(chat_id, texto)
        else:
            logger.info(
                "Update ignorado: tipo=%s",
                next(iter(dados.keys()), "desconhecido"),
            )

        return "ok", 200
    except Exception:
        logger.exception("Erro inesperado no webhook")
        return "ok", 200


init_db()
