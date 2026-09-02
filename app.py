import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, request
from openai import OpenAI

from atlas_ads_client import ATLAS_OFFLINE_FRIENDLY_MESSAGE, AtlasAdsClient

app = Flask(__name__)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("telegram_personal_assistant")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_ALLOWED_USER_ID = os.getenv("TELEGRAM_ALLOWED_USER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DB_PATH = os.getenv(
    "CONVERSATION_DB_PATH",
    os.path.join(tempfile.gettempdir(), "bot_telegram_atlas_memory.sqlite3"),
)
ATLAS_ADS_API_URL = os.getenv("ATLAS_ADS_API_URL")
ATLAS_ADS_API_KEY = os.getenv("ATLAS_ADS_API_KEY")

MAX_STORED_MESSAGES_PER_CHAT = 60
MAX_CONTEXT_MESSAGES_PER_REQUEST = 40
MAX_CONTEXT_CHARACTERS = 12000
TELEGRAM_MESSAGE_LIMIT = 4000
OPENAI_TIMEOUT_SECONDS = 45
TELEGRAM_TIMEOUT_SECONDS = 15
ATLAS_CONTEXT_TTL_SECONDS = int(os.getenv("ATLAS_CONTEXT_TTL_SECONDS", "1800"))

FALLBACK_ERROR_MESSAGE = (
    "Tive um problema pra processar sua mensagem agora. "
    "Tenta de novo em alguns instantes."
)
UNAUTHORIZED_MESSAGE = "Esse bot esta liberado apenas para o usuario autorizado."
ATLAS_OFFLINE_MESSAGE = ATLAS_OFFLINE_FRIENDLY_MESSAGE
ATLAS_NOT_CONFIGURED_MESSAGE = (
    "O Atlas Ads AI ainda nao esta configurado neste bot."
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
atlas_client = AtlasAdsClient(
    base_url=ATLAS_ADS_API_URL,
    api_key=ATLAS_ADS_API_KEY,
    connect_timeout=5,
    read_timeout=90,
)

ADS_ACTION_COMMANDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("pause_campaign", ("pausa a campanha", "pausar campanha", "pausar os anuncios", "pausa os anuncios")),
    ("activate_campaign", ("ativa a campanha", "ativar campanha", "ativar os anuncios", "ativa os anuncios")),
    ("update_campaign_budget", ("muda o orcamento", "altera o orcamento", "ajusta o orcamento", "mudar o budget", "aumenta o orcamento", "reduz o orcamento")),
    ("create_ad", ("cria um anuncio", "criar um anuncio", "cria anuncio", "novo anuncio", "anuncio para o cliente")),
    ("upload_creative", ("subir esse criativo", "subir criativo", "enviar criativo", "publicar criativo")),
]

ADS_QUERY_COMMANDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("show_campaigns", ("mostra minhas campanhas", "minhas campanhas", "listar campanhas", "ver campanhas")),
    ("today_status", ("como estao meus anuncios hoje", "como estao meus anuncios", "como estao as campanhas hoje", "como estao meus ads hoje", "status dos anuncios hoje")),
    ("top_spend_campaign", ("qual campanha esta gastando mais", "qual campanha gasta mais", "campanha gastando mais", "maior gasto", "qual anuncio esta gastando mais")),
    ("performance_7d", ("me mostra o desempenho dos ultimos 7 dias", "desempenho dos ultimos 7 dias", "ultimos 7 dias", "performance dos ultimos 7 dias")),
    ("meta_accounts", ("contas meta", "minhas contas meta", "status da conta meta", "status meta")),
]

ADS_KEYWORDS = (
    "campanha",
    "campanhas",
    "anuncio",
    "anuncios",
    "meta ads",
    "facebook ads",
    "instagram ads",
    "orcamento",
    "budget",
    "criativo",
    "pixel",
    "cpa",
    "cpl",
    "cpc",
    "roas",
    "conversao",
    "leads",
    "remarketing",
    "meta business",
)

CLEAR_LOCAL_CONVERSATION_PATTERNS = (
    "oi",
    "ola",
    "oi tudo bem",
    "ola tudo bem",
    "bom dia",
    "boa tarde",
    "boa noite",
    "como voce esta",
    "como vai voce",
    "me ajuda a organizar um projeto",
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_actions (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                confirmation_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS atlas_conversation_context (
                chat_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
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


def load_pending_action(chat_id: str) -> Optional[Dict[str, str]]:
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT chat_id, user_id, action_id, confirmation_id, created_at, updated_at
            FROM pending_actions
            WHERE chat_id = ?
            """,
            (chat_id,),
        )
        row = cursor.fetchone()

    if not row:
        return None

    return {
        "chat_id": row[0],
        "user_id": row[1],
        "action_id": row[2],
        "confirmation_id": row[3],
        "created_at": str(row[4]),
        "updated_at": str(row[5]),
    }


def store_pending_action(
    chat_id: str,
    user_id: str,
    action_id: str,
    confirmation_id: str,
) -> None:
    with DB_LOCK:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO pending_actions (
                    chat_id, user_id, action_id, confirmation_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, strftime('%s', 'now'), strftime('%s', 'now'))
                ON CONFLICT(chat_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    action_id = excluded.action_id,
                    confirmation_id = excluded.confirmation_id,
                    updated_at = excluded.updated_at
                """,
                (chat_id, user_id, action_id, confirmation_id),
            )


def clear_pending_action(chat_id: str) -> None:
    with DB_LOCK:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM pending_actions WHERE chat_id = ?", (chat_id,))


def load_atlas_context(chat_id: str) -> Optional[Dict[str, str]]:
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT conversation_id, updated_at
            FROM atlas_conversation_context
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()

    if not row:
        return None
    if int(time.time()) - int(row[1]) > ATLAS_CONTEXT_TTL_SECONDS:
        clear_atlas_context(chat_id)
        return None
    return {"conversation_id": row[0], "updated_at": str(row[1])}


def activate_atlas_context(chat_id: str) -> str:
    conversation_id = f"telegram:{chat_id}"
    with DB_LOCK:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO atlas_conversation_context (chat_id, conversation_id, updated_at)
                VALUES (?, ?, strftime('%s', 'now'))
                ON CONFLICT(chat_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (chat_id, conversation_id),
            )
    return conversation_id


def clear_atlas_context(chat_id: str) -> None:
    with DB_LOCK:
        with get_db_connection() as conn:
            conn.execute(
                "DELETE FROM atlas_conversation_context WHERE chat_id = ?", (chat_id,)
            )


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower().strip()


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


def build_openai_messages(chat_id: str, user_text: str) -> List[Dict[str, str]]:
    recent_messages = load_recent_messages(chat_id)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent_messages)
    messages.append({"role": "user", "content": user_text})
    return messages


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


def is_allowed_user(user_id: Optional[int]) -> bool:
    if not TELEGRAM_ALLOWED_USER_ID:
        return True
    if user_id is None:
        return False
    return str(user_id) == TELEGRAM_ALLOWED_USER_ID.strip()


def match_command(normalized_text: str, patterns: List[Tuple[str, Tuple[str, ...]]]) -> Optional[str]:
    for command_name, command_patterns in patterns:
        for pattern in command_patterns:
            if pattern in normalized_text:
                return command_name
    return None


def detect_ads_intent(text: str) -> Optional[Dict[str, str]]:
    normalized_text = normalize_text(text)

    command_name = match_command(normalized_text, ADS_ACTION_COMMANDS)
    if command_name:
        return {"kind": "action", "command_name": command_name}

    command_name = match_command(normalized_text, ADS_QUERY_COMMANDS)
    if command_name:
        return {"kind": "query", "command_name": command_name}

    if any(keyword in normalized_text for keyword in ADS_KEYWORDS):
        return {"kind": "query", "command_name": "general_ads_query"}

    return None


def is_clearly_local_conversation(text: str) -> bool:
    normalized = normalize_text(text).strip(" !?.")
    return normalized in CLEAR_LOCAL_CONVERSATION_PATTERNS


def is_confirmation_text(text: str) -> bool:
    normalized = normalize_text(text)
    confirmation_patterns = (
        "confirmo",
        "confirmar",
        "pode executar",
        "sim, pode",
        "sim pode",
        "pode seguir",
    )
    return any(pattern == normalized or normalized.startswith(pattern) for pattern in confirmation_patterns)


def format_atlas_data(data: object, default_text: str) -> str:
    if not data:
        return default_text

    if isinstance(data, str):
        cleaned = data.strip()
        return cleaned or default_text

    if isinstance(data, dict):
        for key in ("message", "reply", "answer", "summary", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        nested = data.get("data")
        if isinstance(nested, (dict, list)):
            nested_text = json.dumps(nested, ensure_ascii=False, indent=2)
            if nested_text.strip():
                return nested_text

        return json.dumps(data, ensure_ascii=False, indent=2)

    if isinstance(data, list):
        return json.dumps(data, ensure_ascii=False, indent=2)

    return str(data)


def atlas_request(
    *,
    text: str,
    chat_id: str,
    user_id: int,
    action_id: Optional[str] = None,
    confirmation_id: Optional[str] = None,
) -> Any:
    if not atlas_client.is_configured():
        return None
    context = load_atlas_context(chat_id)
    conversation_id = (
        context["conversation_id"] if context else activate_atlas_context(chat_id)
    )
    result = atlas_client.send_command(
        message=text,
        external_user_id=str(chat_id),
        conversation_id=conversation_id,
        action_id=action_id,
        confirmation_id=confirmation_id,
    )
    activate_atlas_context(chat_id)
    return result


def atlas_response_text(result: Any, default_text: str) -> str:
    payload = result.data if getattr(result, "ok", False) else None
    if isinstance(payload, dict):
        status = payload.get("status")
        message = payload.get("message")
        data = payload.get("data")

        if status == "confirmation_required":
            if isinstance(message, str) and message.strip():
                return message.strip()
            return "Preciso da confirmação do Atlas Ads AI para continuar."

        if status == "completed":
            base_text = message.strip() if isinstance(message, str) and message.strip() else default_text
            if isinstance(data, (dict, list)) and data:
                data_text = json.dumps(data, ensure_ascii=False, indent=2)
                return f"{base_text}\n\n{data_text}".strip()
            return base_text

        if status == "failed":
            if isinstance(message, str) and message.strip():
                return message.strip()
            if isinstance(data, (dict, list)) and data:
                return json.dumps(data, ensure_ascii=False, indent=2)
            return default_text

        if isinstance(message, str) and message.strip():
            return message.strip()

        if isinstance(data, (dict, list)) and data:
            return json.dumps(data, ensure_ascii=False, indent=2)

    return default_text


def atlas_friendly_failure(result: Any) -> str:
    status_code = getattr(result, "status_code", None)
    friendly_message = getattr(result, "friendly_message", "") or ""
    if status_code is None or (isinstance(status_code, int) and status_code >= 500):
        return ATLAS_OFFLINE_MESSAGE
    return friendly_message or ATLAS_OFFLINE_MESSAGE


def handle_atlas_query(chat_id: str, user_id: int, text: str) -> str:
    result = atlas_request(text=text, chat_id=chat_id, user_id=user_id)
    if result is None:
        return ATLAS_NOT_CONFIGURED_MESSAGE
    if not result.ok:
        return atlas_friendly_failure(result)
    return atlas_response_text(result, "Nao consegui extrair um retorno util do Atlas Ads AI.")


def handle_atlas_action(chat_id: str, user_id: int, text: str) -> str:
    result = atlas_request(text=text, chat_id=chat_id, user_id=user_id)
    if result is None:
        return ATLAS_NOT_CONFIGURED_MESSAGE
    if not result.ok:
        return atlas_friendly_failure(result)

    payload = result.data if isinstance(result.data, dict) else {}
    status = payload.get("status") if isinstance(payload, dict) else None
    action_id = payload.get("action_id") if isinstance(payload, dict) else None
    confirmation_id = payload.get("confirmation_id") if isinstance(payload, dict) else None

    if status == "confirmation_required":
        if isinstance(action_id, str) and action_id and isinstance(confirmation_id, str) and confirmation_id:
            store_pending_action(
                chat_id=chat_id,
                user_id=str(user_id),
                action_id=action_id,
                confirmation_id=confirmation_id,
            )
        return atlas_response_text(result, "Recebi a solicitacao. Responda confirmo para seguir.")

    return atlas_response_text(result, "Recebi a solicitacao, mas nao consegui preparar a acao.")


def handle_pending_confirmation(chat_id: str, user_id: int, text: str) -> Optional[str]:
    pending = load_pending_action(chat_id)
    if not pending:
        return None

    if str(user_id) != pending["user_id"]:
        return UNAUTHORIZED_MESSAGE

    if not is_confirmation_text(text):
        return None

    result = atlas_request(
        text="confirmo",
        chat_id=chat_id,
        user_id=user_id,
        action_id=pending["action_id"],
        confirmation_id=pending["confirmation_id"],
    )

    if result is None:
        return ATLAS_NOT_CONFIGURED_MESSAGE

    if getattr(result, "status_code", None) is None:
        return ATLAS_OFFLINE_MESSAGE

    clear_pending_action(chat_id)
    if not result.ok:
        return atlas_friendly_failure(result)

    return atlas_response_text(result, "Confirmacao recebida. A acao foi enviada para execucao no Atlas Ads AI.")


def handle_message(chat_id: int, user_id: Optional[int], user_text: str) -> None:
    chat_key = str(chat_id)
    user_key = str(user_id) if user_id is not None else ""

    logger.info(
        "Mensagem recebida do Telegram: chat_id=%s, user_id=%s, tamanho=%s",
        chat_id,
        user_id,
        len(user_text),
    )

    if not is_allowed_user(user_id):
        logger.warning(
            "Mensagem bloqueada por restricao de usuario: chat_id=%s user_id=%s",
            chat_id,
            user_id,
        )
        send_telegram_message(chat_id, UNAUTHORIZED_MESSAGE)
        return

    store_message(chat_key, "user", user_text)

    try:
        pending_reply = handle_pending_confirmation(chat_key, user_id or 0, user_text)
        if pending_reply is not None:
            logger.info(
                "route=atlas chat_id=%s message_length=%s reason=pending_confirmation",
                chat_id,
                len(user_text),
            )
            reply_text = pending_reply
        else:
            intent = detect_ads_intent(user_text)
            if intent:
                logger.info(
                    "route=atlas chat_id=%s message_length=%s reason=ads_intent",
                    chat_id,
                    len(user_text),
                )
                reply_text = (
                    handle_atlas_action(chat_key, user_id or 0, user_text)
                    if intent["kind"] == "action"
                    else handle_atlas_query(chat_key, user_id or 0, user_text)
                )
            elif load_atlas_context(chat_key) and not is_clearly_local_conversation(user_text):
                logger.info(
                    "route=atlas chat_id=%s message_length=%s reason=atlas_context",
                    chat_id,
                    len(user_text),
                )
                reply_text = handle_atlas_query(chat_key, user_id or 0, user_text)
            else:
                reason = "explicit_local_exit" if load_atlas_context(chat_key) else "no_ads_context"
                clear_atlas_context(chat_key)
                logger.info(
                    "route=local_openai chat_id=%s message_length=%s reason=%s",
                    chat_id,
                    len(user_text),
                    reason,
                )
                reply_text = generate_openai_reply(chat_key, user_text)

        store_message(chat_key, "assistant", reply_text)
        send_telegram_message(chat_id, reply_text)
    except Exception:
        logger.exception(
            "Erro ao processar mensagem no chat_id=%s user_id=%s",
            chat_id,
            user_key,
        )
        store_message(chat_key, "assistant", FALLBACK_ERROR_MESSAGE)
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
        user_id = mensagem.get("from", {}).get("id")

        if chat_id is not None and texto:
            handle_message(chat_id, user_id, texto)
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
