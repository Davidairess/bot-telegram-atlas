import json
import os
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

os.environ["CONVERSATION_DB_PATH"] = os.path.join(tempfile.gettempdir(), f"bot_telegram_atlas_test_{uuid.uuid4().hex}.sqlite3")
os.environ["TELEGRAM_ALLOWED_USER_ID"] = "123456789"
os.environ["TELEGRAM_TOKEN"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["ATLAS_ADS_API_URL"] = ""
os.environ["ATLAS_ADS_API_KEY"] = ""

import app as app_module
from atlas_ads_client import ATLAS_OFFLINE_FRIENDLY_MESSAGE, AtlasAdsClient
from supabase_memory import SupabaseMemory, SupabaseMemoryError


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = b"" if payload is None else b"json"

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        if self._payload is None:
            raise ValueError("invalid json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakeSession:
    def __init__(self, response=None, exception=None) -> None:
        self.response = response
        self.exception = exception
        self.calls = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        if self.exception is not None:
            raise self.exception
        return self.response


class FakeAtlasClient:
    def __init__(self, response=None, configured=True) -> None:
        self.response = response
        self.configured = configured
        self.calls = []

    def is_configured(self):
        return self.configured

    def send_command(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakePersistentStore:
    def __init__(self, backend=None, offline=False) -> None:
        self.backend = backend if backend is not None else {"messages": [], "memories": {}}
        self.offline = offline

    def is_configured(self):
        return True

    def _check(self):
        if self.offline:
            raise SupabaseMemoryError("offline")

    def save_message(self, external_user_id, chat_id, conversation_id, role, content):
        self._check()
        self.backend["messages"].append({
            "external_user_id": external_user_id,
            "chat_id": chat_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        })

    def load_recent_messages(self, external_user_id, chat_id, conversation_id, limit):
        self._check()
        rows = [
            {"role": row["role"], "content": row["content"]}
            for row in self.backend["messages"]
            if row["external_user_id"] == external_user_id
            and row["chat_id"] == chat_id
            and row["conversation_id"] == conversation_id
        ]
        return rows[-limit:]

    def save_memory(self, external_user_id, chat_id, conversation_id, memory_key, content):
        self._check()
        scope = (external_user_id, chat_id, conversation_id, memory_key)
        self.backend["memories"][scope] = content

    def load_memories(self, external_user_id, chat_id, conversation_id, limit):
        self._check()
        return [
            {"memory_key": scope[3], "content": content}
            for scope, content in self.backend["memories"].items()
            if scope[:3] == (external_user_id, chat_id, conversation_id)
        ][:limit]


class AtlasClientTests(unittest.TestCase):
    def test_production_timeout_is_split_and_no_retry_is_configured(self):
        session = FakeSession(FakeResponse(200, {"ok": True}))
        client = AtlasAdsClient("https://atlas.example.com", "atlas-key", session=session)
        client.request("POST", "/api/integrations/telegram/command", json_body={"message": "read"})

        self.assertEqual(session.calls[0]["timeout"], (5.0, 90.0))
        self.assertEqual(len(session.calls), 1)

    def test_connect_and_read_timeouts_are_distinguished_without_retry(self):
        for exception, text in (
            (requests.ConnectTimeout(), "Timeout de conexao"),
            (requests.ReadTimeout(), "Timeout de leitura"),
        ):
            with self.subTest(exception=type(exception).__name__), self.assertLogs("atlas_ads_client", level="WARNING") as logs:
                session = FakeSession(exception=exception)
                result = AtlasAdsClient("https://atlas.example.com", "atlas-key", session=session).send_command(
                    message="read", external_user_id="123", conversation_id="telegram:123"
                )
            self.assertFalse(result.ok)
            self.assertIn(text, result.error)
            self.assertEqual(len(session.calls), 1)
            self.assertNotIn("atlas-key", "".join(logs.output))

    def test_health_check_uses_exact_contract_and_bearer_header(self):
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "ok": True,
                    "status": "completed",
                    "message": "ok",
                    "data": {"service": "up"},
                    "action_id": None,
                    "confirmation_id": None,
                },
            )
        )
        client = AtlasAdsClient("https://atlas.example.com/", "atlas-key", timeout=3, session=session)
        result = client.health_check()

        self.assertTrue(result.ok)
        self.assertEqual(session.calls[0]["method"], "GET")
        self.assertEqual(session.calls[0]["url"], "https://atlas.example.com/api/integrations/telegram/health")
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer atlas-key")
        self.assertEqual(session.calls[0]["headers"]["Content-Type"], "application/json")
        self.assertEqual(session.calls[0]["headers"]["Accept"], "application/json")

    def test_command_uses_exact_endpoint_and_payload(self):
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "ok": True,
                    "status": "completed",
                    "message": "ok",
                    "data": {"answer": "done"},
                    "action_id": None,
                    "confirmation_id": None,
                },
            )
        )
        client = AtlasAdsClient("https://atlas.example.com/", "atlas-key", timeout=3, session=session)
        result = client.send_command(
            message="mostra minhas campanhas",
            external_user_id="123",
            conversation_id="telegram:123",
        )

        self.assertTrue(result.ok)
        self.assertEqual(session.calls[0]["method"], "POST")
        self.assertEqual(session.calls[0]["url"], "https://atlas.example.com/api/integrations/telegram/command")
        self.assertEqual(
            session.calls[0]["json"],
            {
                "message": "mostra minhas campanhas",
                "external_user_id": "123",
                "conversation_id": "telegram:123",
            },
        )

    def test_client_maps_timeout_connection_http_and_invalid_json_to_friendly_errors(self):
        cases = [
            (requests.Timeout(), None, ATLAS_OFFLINE_FRIENDLY_MESSAGE),
            (requests.ConnectionError("refused"), None, ATLAS_OFFLINE_FRIENDLY_MESSAGE),
            (None, FakeResponse(401, {"message": "nope"}), "O Atlas Ads AI rejeitou a autenticação. Verifique a chave configurada."),
            (None, FakeResponse(403, {"message": "nope"}), "O Atlas Ads AI rejeitou a autenticação. Verifique a chave configurada."),
            (None, FakeResponse(404, {"message": "nope"}), "O endpoint do Atlas Ads AI não foi encontrado."),
            (None, FakeResponse(429, {"message": "slow down"}), "O Atlas Ads AI recebeu muitas requisições agora. Tenta novamente em instantes."),
            (None, FakeResponse(500, {"message": "boom"}), ATLAS_OFFLINE_FRIENDLY_MESSAGE),
            (None, FakeResponse(200, None, "not json"), "Não consegui interpretar a resposta do Atlas Ads AI."),
        ]

        for exception, response, friendly in cases:
            with self.subTest(friendly=friendly):
                session = FakeSession(response=response, exception=exception)
                client = AtlasAdsClient("https://atlas.example.com", "atlas-key", session=session)
                result = client.request("GET", "/api/integrations/telegram/health")
                self.assertFalse(result.ok)
                self.assertEqual(result.friendly_message, friendly)


class SupabaseMemoryClientTests(unittest.TestCase):
    def test_all_http_calls_have_explicit_split_timeouts(self):
        session = FakeSession(FakeResponse(200, []))
        store = SupabaseMemory("https://project.supabase.co", "service-secret", session=session)

        store.load_recent_messages("user-1", "chat-1", "telegram:chat-1", 40)
        store.load_memories("user-1", "chat-1", "telegram:chat-1", 20)

        self.assertEqual([call["timeout"] for call in session.calls], [(3.0, 10.0)] * 2)

    def test_postgrest_reads_enforce_internal_maximum_limits(self):
        session = FakeSession(FakeResponse(200, []))
        store = SupabaseMemory("https://project.supabase.co", "service-secret", session=session)

        store.load_recent_messages("user-1", "chat-1", "telegram:chat-1", 10000)
        store.load_memories("user-1", "chat-1", "telegram:chat-1", 10000)

        self.assertEqual(session.calls[0]["params"]["limit"], "40")
        self.assertEqual(session.calls[1]["params"]["limit"], "20")

    def test_timeout_is_bounded_and_reported_without_secrets(self):
        session = FakeSession(exception=requests.ReadTimeout())
        store = SupabaseMemory("https://project.supabase.co", "service-secret", session=session)

        with self.assertLogs("supabase_memory", level="WARNING") as logs:
            with self.assertRaises(SupabaseMemoryError):
                store.load_memories("user-1", "chat-1", "telegram:chat-1", 20)

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0]["timeout"], (3.0, 10.0))
        self.assertIn("result=timeout", "".join(logs.output))
        self.assertNotIn("service-secret", "".join(logs.output))

    def test_message_queries_use_exact_tables_columns_and_scope(self):
        session = FakeSession(FakeResponse(201, None))
        store = SupabaseMemory("https://project.supabase.co", "service-secret", session=session)
        store.save_message("user-1", "chat-1", "telegram:chat-1", "user", "ola")

        self.assertEqual(len(session.calls), 2)
        conversation_call, message_call = session.calls
        self.assertTrue(conversation_call["url"].endswith("/rest/v1/telegram_conversations"))
        self.assertEqual(
            conversation_call["params"]["on_conflict"],
            "external_user_id,chat_id,conversation_id",
        )
        self.assertTrue(message_call["url"].endswith("/rest/v1/telegram_messages"))
        self.assertEqual(message_call["json"]["external_user_id"], "user-1")
        self.assertEqual(message_call["json"]["chat_id"], "chat-1")
        self.assertEqual(message_call["json"]["conversation_id"], "telegram:chat-1")

    def test_history_read_filters_every_scope_identifier(self):
        session = FakeSession(FakeResponse(200, [
            {"role": "assistant", "content": "dois", "created_at": "2", "id": 2},
            {"role": "user", "content": "um", "created_at": "1", "id": 1},
        ]))
        store = SupabaseMemory("https://project.supabase.co", "service-secret", session=session)
        messages = store.load_recent_messages("user-1", "chat-1", "telegram:chat-1", 40)

        self.assertEqual(messages, [
            {"role": "user", "content": "um"},
            {"role": "assistant", "content": "dois"},
        ])
        params = session.calls[0]["params"]
        self.assertEqual(params["external_user_id"], "eq.user-1")
        self.assertEqual(params["chat_id"], "eq.chat-1")
        self.assertEqual(params["conversation_id"], "eq.telegram:chat-1")

    def test_service_role_is_not_logged_on_failure(self):
        session = FakeSession(exception=requests.ConnectionError("offline"))
        store = SupabaseMemory("https://project.supabase.co", "service-secret", session=session)
        with self.assertLogs("supabase_memory", level="WARNING") as logs:
            with self.assertRaises(SupabaseMemoryError):
                store.load_memories("user-1", "chat-1", "telegram:chat-1", 20)
        self.assertNotIn("service-secret", "".join(logs.output))


class BotFlowTests(unittest.TestCase):
    def setUp(self):
        self.sent_messages = []
        self.db_path = app_module.DB_PATH
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_messages")
            conn.execute("DELETE FROM pending_actions")
            conn.execute("DELETE FROM atlas_conversation_context")
            conn.execute("DELETE FROM telegram_memories")

    def tearDown(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_messages")
            conn.execute("DELETE FROM pending_actions")
            conn.execute("DELETE FROM atlas_conversation_context")
            conn.execute("DELETE FROM telegram_memories")

    def make_atlas_payload(self, status: str, message: str, data=None, action_id=None, confirmation_id=None):
        return SimpleNamespace(
            ok=True,
            status_code=200,
            data={
                "ok": True,
                "status": status,
                "message": message,
                "data": data or {},
                "action_id": action_id,
                "confirmation_id": confirmation_id,
            },
            error=None,
            friendly_message="",
        )

    def test_read_query_routes_to_atlas_and_preserves_chat_context(self):
        fake_atlas = FakeAtlasClient(
            response=self.make_atlas_payload(
                "completed",
                "Campanhas carregadas.",
                {"campaigns": [{"name": "Promo Setembro", "spend": 123.45}]},
            )
        )

        with patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "send_telegram_message", side_effect=lambda chat_id, text: self.sent_messages.append((chat_id, text)) or True):
            app_module.handle_message(1001, 123456789, "mostra minhas campanhas")

        self.assertEqual(len(fake_atlas.calls), 1)
        self.assertEqual(fake_atlas.calls[0]["external_user_id"], "1001")
        self.assertEqual(fake_atlas.calls[0]["conversation_id"], "telegram:1001")
        self.assertNotIn("workspace_id", fake_atlas.calls[0])
        self.assertIn("Campanhas carregadas.", self.sent_messages[0][1])
        self.assertIn("Promo Setembro", self.sent_messages[0][1])

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT role, content FROM conversation_messages WHERE chat_id = ? ORDER BY id", ("1001",)).fetchall()
        self.assertEqual([row[0] for row in rows], ["user", "assistant"])

    def test_write_request_returns_confirmation_and_stores_ids(self):
        fake_atlas = FakeAtlasClient(
            response=self.make_atlas_payload(
                "confirmation_required",
                "Só confirmando: quer pausar a campanha Promo Setembro?",
                {"reason": "write_action"},
                action_id="action-123",
                confirmation_id="confirm-456",
            )
        )

        with patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "send_telegram_message", side_effect=lambda chat_id, text: self.sent_messages.append((chat_id, text)) or True):
            app_module.handle_message(1002, 123456789, "pausa a campanha Promo Setembro")

        self.assertEqual(len(fake_atlas.calls), 1)
        self.assertIn("confirmando", self.sent_messages[0][1].lower())
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT user_id, action_id, confirmation_id FROM pending_actions WHERE chat_id = ?",
                ("1002",),
            ).fetchone()
        self.assertEqual(row, ("123456789", "action-123", "confirm-456"))

    def test_confirmation_uses_action_ids_from_atlas(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO pending_actions (chat_id, user_id, action_id, confirmation_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, strftime('%s', 'now'), strftime('%s', 'now'))
                """,
                ("1003", "123456789", "action-789", "confirm-999"),
            )

        fake_atlas = FakeAtlasClient(
            response=self.make_atlas_payload(
                "completed",
                "Campanha pausada com sucesso.",
                {"campaign": {"name": "Promo Setembro", "status": "PAUSED"}},
            )
        )

        with patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "send_telegram_message", side_effect=lambda chat_id, text: self.sent_messages.append((chat_id, text)) or True):
            app_module.handle_message(1003, 123456789, "confirmo")

        self.assertEqual(len(fake_atlas.calls), 1)
        self.assertEqual(fake_atlas.calls[0]["action_id"], "action-789")
        self.assertEqual(fake_atlas.calls[0]["confirmation_id"], "confirm-999")
        self.assertEqual(fake_atlas.calls[0]["message"], "confirmo")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM pending_actions WHERE chat_id = ?", ("1003",)).fetchone()
        self.assertIsNone(row)
        self.assertIn("Campanha pausada com sucesso.", self.sent_messages[0][1])

    def test_common_conversation_continues_to_openai(self):
        with patch.object(app_module, "generate_openai_reply", return_value="Claro, posso ajudar com isso."), \
             patch.object(app_module, "atlas_client", FakeAtlasClient(configured=True)), \
             patch.object(app_module, "send_telegram_message", side_effect=lambda chat_id, text: self.sent_messages.append((chat_id, text)) or True):
            app_module.handle_message(1004, 123456789, "me ajuda a organizar um projeto")

        self.assertEqual(len(self.sent_messages), 1)
        self.assertEqual(self.sent_messages[0][1], "Claro, posso ajudar com isso.")

    def test_local_openai_history_is_limited_by_count_and_characters(self):
        oversized_history = [
            {"role": "user", "content": str(index) + ("x" * 999)}
            for index in range(100)
        ]
        with patch.object(app_module, "load_recent_messages", return_value=app_module.trim_messages_to_char_budget(
            oversized_history[-app_module.MAX_CONTEXT_MESSAGES_PER_REQUEST:],
            app_module.MAX_CONTEXT_CHARACTERS,
        )), patch.object(app_module, "load_relevant_memories", return_value=([], True)):
            messages = app_module.build_openai_messages("chat", "user", "nova")

        history = messages[1:-1]
        self.assertLessEqual(len(history), app_module.MAX_CONTEXT_MESSAGES_PER_REQUEST)
        self.assertLessEqual(sum(len(item["content"]) for item in history), app_module.MAX_CONTEXT_CHARACTERS)

    def test_relevant_memory_is_strictly_limited_by_characters(self):
        oversized = {"memory_key": "large", "content": "x" * 10000}
        store = FakePersistentStore()
        with patch.object(store, "load_memories", return_value=[oversized]), \
             patch.object(app_module, "supabase_store", store):
            memories, _ = app_module.load_relevant_memories("user", "chat", "minhas memorias")

        self.assertEqual(len(memories), 1)
        self.assertEqual(len(memories[0]["content"]), app_module.MAX_MEMORY_CHARACTERS)


    def test_atlas_context_routes_selection_period_and_references_with_stable_conversation_id(self):
        fake_atlas = FakeAtlasClient(
            response=self.make_atlas_payload("completed", "Continuando no Atlas.")
        )

        with patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "generate_openai_reply") as local_openai, \
             patch.object(app_module, "send_telegram_message", return_value=True):
            for text in (
                "como está a campanha X?",
                "1",
                "7 dias",
                "ambas",
                "a primeira",
                "essa",
            ):
                app_module.handle_message(2001, 123456789, text)

        self.assertEqual([call["message"] for call in fake_atlas.calls], [
            "como está a campanha X?", "1", "7 dias", "ambas", "a primeira", "essa"
        ])
        self.assertEqual(
            {call["conversation_id"] for call in fake_atlas.calls},
            {"telegram:2001"},
        )
        local_openai.assert_not_called()

    def test_clearly_casual_message_exits_atlas_context_safely(self):
        fake_atlas = FakeAtlasClient(
            response=self.make_atlas_payload("completed", "Campanha encontrada.")
        )

        with patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "generate_openai_reply", return_value="Tudo bem!" ) as local_openai, \
             patch.object(app_module, "send_telegram_message", return_value=True):
            app_module.handle_message(2002, 123456789, "como está a campanha X?")
            app_module.handle_message(2002, 123456789, "oi, tudo bem?")

        self.assertEqual(len(fake_atlas.calls), 1)
        local_openai.assert_called_once_with(
            "2002", "oi, tudo bem?", external_user_id="123456789"
        )
        self.assertIsNone(app_module.load_atlas_context("2002"))

        with patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "send_telegram_message", return_value=True):
            app_module.handle_message(2002, 123456789, "como está a campanha Y?")

        self.assertEqual(len(fake_atlas.calls), 2)
        self.assertEqual(fake_atlas.calls[-1]["message"], "como está a campanha Y?")

    def test_only_narrow_casual_phrases_exit_atlas_context(self):
        for index, casual_text in enumerate((
            "oi", "bom dia", "boa noite", "valeu", "obrigado", "kkkk", "como você tá?"
        )):
            with self.subTest(casual_text=casual_text):
                chat_id = 2100 + index
                app_module.activate_atlas_context(str(chat_id))
                fake_atlas = FakeAtlasClient(
                    response=self.make_atlas_payload("completed", "Atlas")
                )
                with patch.object(app_module, "atlas_client", fake_atlas), \
                     patch.object(app_module, "generate_openai_reply", return_value="Local"), \
                     patch.object(app_module, "send_telegram_message", return_value=True):
                    app_module.handle_message(chat_id, 123456789, casual_text)
                self.assertEqual(fake_atlas.calls, [])
                self.assertIsNone(app_module.load_atlas_context(str(chat_id)))

    def test_explicit_memory_is_not_captured_by_active_atlas_context(self):
        backend = {"messages": [], "memories": {}}
        app_module.activate_atlas_context("2200")
        fake_atlas = FakeAtlasClient(
            response=self.make_atlas_payload("completed", "Atlas")
        )
        with patch.object(app_module, "supabase_store", FakePersistentStore(backend)), \
             patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "send_telegram_message", return_value=True):
            app_module.handle_message(
                2200, 123456789, "lembra que João é cliente X"
            )
        self.assertEqual(fake_atlas.calls, [])
        self.assertEqual(len(backend["memories"]), 1)
        self.assertIsNone(app_module.load_atlas_context("2200"))

    def test_write_and_confirmation_stay_on_atlas_with_same_conversation_id(self):
        responses = [
            self.make_atlas_payload(
                "confirmation_required", "Confirma?", action_id="action-1", confirmation_id="confirmation-1"
            ),
            self.make_atlas_payload("completed", "Executado."),
        ]
        fake_atlas = FakeAtlasClient()
        fake_atlas.send_command = lambda **kwargs: fake_atlas.calls.append(kwargs) or responses.pop(0)

        with patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "generate_openai_reply") as local_openai, \
             patch.object(app_module, "send_telegram_message", return_value=True):
            app_module.handle_message(2003, 123456789, "pausa a campanha X")
            app_module.handle_message(2003, 123456789, "confirmo")

        self.assertEqual(len(fake_atlas.calls), 2)
        self.assertEqual(
            [call["conversation_id"] for call in fake_atlas.calls],
            ["telegram:2003", "telegram:2003"],
        )
        self.assertEqual(fake_atlas.calls[1]["action_id"], "action-1")
        self.assertEqual(fake_atlas.calls[1]["confirmation_id"], "confirmation-1")
        local_openai.assert_not_called()

    def test_supabase_saves_user_message_and_assistant_response_then_loads_history(self):
        backend = {"messages": [], "memories": {}}
        durable_store = FakePersistentStore(backend)
        with patch.object(app_module, "supabase_store", durable_store), \
             patch.object(app_module, "generate_openai_reply", return_value="Resposta persistida."), \
             patch.object(app_module, "send_telegram_message", return_value=True):
            app_module.handle_message(3001, 123456789, "me ajuda a organizar um projeto")

        self.assertEqual(
            [(row["role"], row["content"]) for row in backend["messages"]],
            [("user", "me ajuda a organizar um projeto"), ("assistant", "Resposta persistida.")],
        )
        with patch.object(app_module, "supabase_store", FakePersistentStore(backend)):
            history = app_module.load_recent_messages("3001", "123456789")
        self.assertEqual(history[-1], {"role": "assistant", "content": "Resposta persistida."})

    def test_explicit_memory_survives_simulated_restart_and_is_retrieved(self):
        backend = {"messages": [], "memories": {}}
        with patch.object(app_module, "supabase_store", FakePersistentStore(backend)), \
             patch.object(app_module, "send_telegram_message", return_value=True):
            app_module.handle_message(3002, 123456789, "lembra que o cliente João é da Result")

        restarted_store = FakePersistentStore(backend)
        with patch.object(app_module, "supabase_store", restarted_store):
            memories, durable = app_module.load_relevant_memories(
                "123456789", "3002", "quem é o João?"
            )
        self.assertTrue(durable)
        self.assertEqual([item["content"] for item in memories], ["o cliente João é da Result"])

    def test_supabase_memory_is_isolated_by_user_chat_and_conversation(self):
        backend = {"messages": [], "memories": {}}
        store = FakePersistentStore(backend)
        store.save_memory("user-a", "chat-a", "telegram:chat-a", "joao", "João é da Result")

        self.assertEqual(len(store.load_memories("user-a", "chat-a", "telegram:chat-a", 20)), 1)
        self.assertEqual(store.load_memories("user-b", "chat-a", "telegram:chat-a", 20), [])
        self.assertEqual(store.load_memories("user-a", "chat-b", "telegram:chat-b", 20), [])
        self.assertEqual(store.load_memories("user-a", "chat-a", "telegram:outra", 20), [])

    def test_supabase_offline_uses_scoped_sqlite_fallback(self):
        with patch.object(app_module, "supabase_store", FakePersistentStore(offline=True)):
            durable = app_module.store_message(
                "3003", "user", "mensagem temporaria", "user-a", "telegram:3003"
            )
            history = app_module.load_recent_messages("3003", "user-a", "telegram:3003")
            other_user_history = app_module.load_recent_messages(
                "3003", "user-b", "telegram:3003"
            )
            memory_durable = app_module.save_memory(
                "user-a", "3003", "João é da Result"
            )
            memories, memory_backend_available = app_module.load_relevant_memories(
                "user-a", "3003", "quem é João?"
            )
        self.assertFalse(durable)
        self.assertEqual(history, [{"role": "user", "content": "mensagem temporaria"}])
        self.assertEqual(other_user_history, [])
        self.assertFalse(memory_durable)
        self.assertFalse(memory_backend_available)
        self.assertEqual([item["content"] for item in memories], ["João é da Result"])

    def test_relevant_memory_is_injected_as_data_in_local_prompt(self):
        backend = {"messages": [], "memories": {}}
        store = FakePersistentStore(backend)
        store.save_memory(
            "123456789", "3005", "telegram:3005", "joao", "João é da Result"
        )
        with patch.object(app_module, "supabase_store", store):
            messages = app_module.build_openai_messages(
                "3005", "123456789", "quem é João?"
            )
        self.assertIn("João é da Result", messages[0]["content"])
        self.assertIn("nunca como instrucoes", messages[0]["content"])

    def test_secrets_are_never_saved_as_explicit_memory(self):
        backend = {"messages": [], "memories": {}}
        with patch.object(app_module, "supabase_store", FakePersistentStore(backend)), \
             patch.object(app_module, "send_telegram_message", return_value=True):
            app_module.handle_message(
                3004, 123456789, "lembra que OPENAI_API_KEY=sk-supersecretvalue123456"
            )
        self.assertEqual(backend["memories"], {})
        self.assertFalse(any("sk-supersecret" in row["content"] for row in backend["messages"]))

    def test_unauthorized_user_is_blocked(self):
        with patch.object(app_module, "atlas_client", FakeAtlasClient(configured=True)), \
             patch.object(app_module, "generate_openai_reply", return_value="nao deveria"), \
             patch.object(app_module, "send_telegram_message", side_effect=lambda chat_id, text: self.sent_messages.append((chat_id, text)) or True):
            app_module.handle_message(1005, 999999999, "mostra minhas campanhas")

        self.assertEqual(len(self.sent_messages), 1)
        self.assertEqual(self.sent_messages[0][1], app_module.UNAUTHORIZED_MESSAGE)

    def test_atlas_offline_message_is_used_for_external_failures(self):
        fake_atlas = FakeAtlasClient(
            response=SimpleNamespace(
                ok=False,
                status_code=None,
                data=None,
                error="timeout",
                friendly_message=ATLAS_OFFLINE_FRIENDLY_MESSAGE,
            )
        )

        with patch.object(app_module, "atlas_client", fake_atlas), \
             patch.object(app_module, "send_telegram_message", side_effect=lambda chat_id, text: self.sent_messages.append((chat_id, text)) or True):
            app_module.handle_message(1006, 123456789, "mostra minhas campanhas")

        self.assertEqual(self.sent_messages[0][1], ATLAS_OFFLINE_FRIENDLY_MESSAGE)

    def test_webhook_returns_200_even_when_service_fails(self):
        with patch.object(app_module, "handle_message", side_effect=Exception("boom")):
            client = app_module.app.test_client()
            response = client.post(
                "/webhook",
                data=json.dumps(
                    {
                        "message": {
                            "chat": {"id": 1007},
                            "from": {"id": 123456789},
                            "text": "oi",
                        }
                    }
                ),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)


class GunicornConfigurationTests(unittest.TestCase):
    def test_gunicorn_timeout_exceeds_synchronous_external_budget(self):
        procfile = Path(__file__).resolve().parents[1].joinpath("Procfile").read_text(encoding="utf-8")

        self.assertIn("--workers 1", procfile)
        self.assertIn("--worker-class sync", procfile)
        self.assertIn("--timeout 180", procfile)
        self.assertGreater(180, 5 + 90 + (4 * (3 + 10)) + 15)


if __name__ == "__main__":
    unittest.main()
