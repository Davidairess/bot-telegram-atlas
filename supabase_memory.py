import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger("supabase_memory")

MAX_HISTORY_ROWS = 40
MAX_MEMORY_ROWS = 20


class SupabaseMemoryError(RuntimeError):
    pass


class SupabaseMemory:
    def __init__(
        self,
        url: Optional[str],
        service_role_key: Optional[str],
        *,
        session: Optional[requests.Session] = None,
        connect_timeout: float = 3.0,
        read_timeout: float = 10.0,
    ) -> None:
        self.url = (url or "").strip().rstrip("/")
        self.service_role_key = (service_role_key or "").strip()
        self.session = session or requests.Session()
        self.timeout = (connect_timeout, read_timeout)

    def is_configured(self) -> bool:
        return bool(self.url and self.service_role_key)

    def _headers(self, *, upsert: bool = False) -> Dict[str, str]:
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if upsert:
            headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
        return headers

    def _request(
        self,
        method: str,
        table: str,
        *,
        params: Optional[Dict[str, str]] = None,
        json_body: Optional[Any] = None,
        upsert: bool = False,
    ) -> Any:
        if not self.is_configured():
            raise SupabaseMemoryError("Supabase nao configurado.")
        started = time.monotonic()
        logger.info(
            "request_started route=postgrest external_service=supabase operation=%s table=%s",
            method,
            table,
        )
        try:
            response = self.session.request(
                method=method,
                url=f"{self.url}/rest/v1/{table}",
                params=params,
                json=json_body,
                headers=self._headers(upsert=upsert),
                timeout=self.timeout,
            )
            response.raise_for_status()
            if not response.content:
                logger.info(
                    "external_service=supabase duration_ms=%s result=success",
                    int((time.monotonic() - started) * 1000),
                )
                return None
            data = response.json()
            logger.info(
                "external_service=supabase duration_ms=%s result=success",
                int((time.monotonic() - started) * 1000),
            )
            return data
        except requests.Timeout as exc:
            logger.warning(
                "external_service=supabase duration_ms=%s result=timeout",
                int((time.monotonic() - started) * 1000),
            )
            raise SupabaseMemoryError("Timeout no armazenamento Supabase.") from exc
        except (requests.RequestException, ValueError) as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning(
                "external_service=supabase duration_ms=%s result=error status=%s",
                int((time.monotonic() - started) * 1000),
                status_code,
            )
            raise SupabaseMemoryError("Falha no armazenamento Supabase.") from exc

    @staticmethod
    def _scope(
        external_user_id: str, chat_id: str, conversation_id: str
    ) -> Dict[str, str]:
        return {
            "external_user_id": external_user_id,
            "chat_id": chat_id,
            "conversation_id": conversation_id,
        }

    def ensure_conversation(
        self, external_user_id: str, chat_id: str, conversation_id: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            **self._scope(external_user_id, chat_id, conversation_id),
            "updated_at": now,
        }
        self._request(
            "POST",
            "telegram_conversations",
            params={"on_conflict": "external_user_id,chat_id,conversation_id"},
            json_body=payload,
            upsert=True,
        )

    def save_message(
        self,
        external_user_id: str,
        chat_id: str,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        self.ensure_conversation(external_user_id, chat_id, conversation_id)
        self._request(
            "POST",
            "telegram_messages",
            json_body={
                **self._scope(external_user_id, chat_id, conversation_id),
                "role": role,
                "content": content,
            },
        )

    def load_recent_messages(
        self,
        external_user_id: str,
        chat_id: str,
        conversation_id: str,
        limit: int,
    ) -> List[Dict[str, str]]:
        bounded_limit = max(1, min(limit, MAX_HISTORY_ROWS))
        rows = self._request(
            "GET",
            "telegram_messages",
            params={
                "select": "role,content,created_at,id",
                "external_user_id": f"eq.{external_user_id}",
                "chat_id": f"eq.{chat_id}",
                "conversation_id": f"eq.{conversation_id}",
                "order": "created_at.desc,id.desc",
                "limit": str(bounded_limit),
            },
        ) or []
        return [
            {"role": row["role"], "content": row["content"]}
            for row in reversed(rows)
            if isinstance(row, dict) and row.get("role") and row.get("content")
        ]

    def save_memory(
        self,
        external_user_id: str,
        chat_id: str,
        conversation_id: str,
        memory_key: str,
        content: str,
    ) -> None:
        self.ensure_conversation(external_user_id, chat_id, conversation_id)
        self._request(
            "POST",
            "telegram_memories",
            params={
                "on_conflict": "external_user_id,chat_id,conversation_id,memory_key"
            },
            json_body={
                **self._scope(external_user_id, chat_id, conversation_id),
                "memory_key": memory_key,
                "content": content,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            upsert=True,
        )

    def load_memories(
        self,
        external_user_id: str,
        chat_id: str,
        conversation_id: str,
        limit: int,
    ) -> List[Dict[str, str]]:
        bounded_limit = max(1, min(limit, MAX_MEMORY_ROWS))
        rows = self._request(
            "GET",
            "telegram_memories",
            params={
                "select": "memory_key,content,updated_at",
                "external_user_id": f"eq.{external_user_id}",
                "chat_id": f"eq.{chat_id}",
                "conversation_id": f"eq.{conversation_id}",
                "order": "updated_at.desc",
                "limit": str(bounded_limit),
            },
        ) or []
        return [
            {"memory_key": row["memory_key"], "content": row["content"]}
            for row in rows
            if isinstance(row, dict) and row.get("memory_key") and row.get("content")
        ]
