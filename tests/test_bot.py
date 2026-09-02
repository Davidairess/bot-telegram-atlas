import json
import os
import sqlite3
import tempfile
import unittest
import uuid
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


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

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


class BotFlowTests(unittest.TestCase):
    def setUp(self):
        self.sent_messages = []
        self.db_path = app_module.DB_PATH
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_messages")
            conn.execute("DELETE FROM pending_actions")

    def tearDown(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_messages")
            conn.execute("DELETE FROM pending_actions")

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


if __name__ == "__main__":
    unittest.main()
