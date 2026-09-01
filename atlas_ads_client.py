import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

ATLAS_OFFLINE_FRIENDLY_MESSAGE = "Não consegui falar com o Atlas Ads AI agora. Tenta novamente em alguns instantes."
ATLAS_INVALID_JSON_MESSAGE = "Não consegui interpretar a resposta do Atlas Ads AI."
ATLAS_AUTH_REJECTED_MESSAGE = "O Atlas Ads AI rejeitou a autenticação. Verifique a chave configurada."
ATLAS_ENDPOINT_NOT_FOUND_MESSAGE = "O endpoint do Atlas Ads AI não foi encontrado."
ATLAS_RATE_LIMIT_MESSAGE = "O Atlas Ads AI recebeu muitas requisições agora. Tenta novamente em instantes."


@dataclass
class AtlasAdsResult:
    ok: bool
    status_code: Optional[int]
    data: Any
    error: Optional[str]
    friendly_message: str


class AtlasAdsClient:
    def __init__(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
        timeout: int = 20,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.session = session or requests.Session()

    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _build_url(self, path: str) -> str:
        cleaned_path = path if path.startswith("/") else f"/{path}"
        return urljoin(f"{self.base_url}/", cleaned_path.lstrip("/"))

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _parse_json(response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    @staticmethod
    def _friendly_message(status_code: Optional[int]) -> str:
        if status_code is None:
            return ATLAS_OFFLINE_FRIENDLY_MESSAGE
        if status_code in (401, 403):
            return ATLAS_AUTH_REJECTED_MESSAGE
        if status_code == 404:
            return ATLAS_ENDPOINT_NOT_FOUND_MESSAGE
        if status_code == 429:
            return ATLAS_RATE_LIMIT_MESSAGE
        if status_code >= 500:
            return ATLAS_OFFLINE_FRIENDLY_MESSAGE
        return ATLAS_OFFLINE_FRIENDLY_MESSAGE

    @staticmethod
    def _extract_error(data: Any, response: Optional[requests.Response] = None) -> Optional[str]:
        if isinstance(data, dict):
            for key in ("error", "detail", "message", "reason"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(data, str) and data.strip():
            return data.strip()
        if response is not None:
            text = response.text.strip()
            if text:
                return text[:500]
        return None

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> AtlasAdsResult:
        if not self.is_configured():
            return AtlasAdsResult(
                ok=False,
                status_code=None,
                data=None,
                error="Atlas Ads AI não configurado.",
                friendly_message="O Atlas Ads AI ainda não está configurado neste bot.",
            )

        url = self._build_url(path)
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.Timeout:
            logger.warning("Timeout ao chamar o Atlas Ads AI em %s %s", method.upper(), path)
            return AtlasAdsResult(
                ok=False,
                status_code=None,
                data=None,
                error="Timeout ao chamar o Atlas Ads AI.",
                friendly_message=ATLAS_OFFLINE_FRIENDLY_MESSAGE,
            )
        except requests.ConnectionError:
            logger.warning("Falha de conexão ao chamar o Atlas Ads AI em %s %s", method.upper(), path)
            return AtlasAdsResult(
                ok=False,
                status_code=None,
                data=None,
                error="Falha de conexão ao chamar o Atlas Ads AI.",
                friendly_message=ATLAS_OFFLINE_FRIENDLY_MESSAGE,
            )
        except requests.RequestException:
            logger.exception("Erro de rede ao chamar o Atlas Ads AI em %s %s", method.upper(), path)
            return AtlasAdsResult(
                ok=False,
                status_code=None,
                data=None,
                error="Erro de rede ao chamar o Atlas Ads AI.",
                friendly_message=ATLAS_OFFLINE_FRIENDLY_MESSAGE,
            )

        data = self._parse_json(response)
        if 200 <= response.status_code < 300:
            if data is None:
                logger.warning(
                    "Atlas Ads AI retornou HTTP %s com JSON inválido em %s %s",
                    response.status_code,
                    method.upper(),
                    path,
                )
                return AtlasAdsResult(
                    ok=False,
                    status_code=response.status_code,
                    data=None,
                    error="Resposta JSON inválida do Atlas Ads AI.",
                    friendly_message=ATLAS_INVALID_JSON_MESSAGE,
                )

            return AtlasAdsResult(
                ok=True,
                status_code=response.status_code,
                data=data,
                error=None,
                friendly_message="",
            )

        logger.warning(
            "Atlas Ads AI respondeu com HTTP %s em %s %s",
            response.status_code,
            method.upper(),
            path,
        )
        return AtlasAdsResult(
            ok=False,
            status_code=response.status_code,
            data=data,
            error=self._extract_error(data, response),
            friendly_message=self._friendly_message(response.status_code),
        )

    def health_check(self) -> AtlasAdsResult:
        return self.request("GET", "/api/integrations/telegram/health")

    def send_command(
        self,
        *,
        message: str,
        external_user_id: str,
        conversation_id: str,
        action_id: Optional[str] = None,
        confirmation_id: Optional[str] = None,
    ) -> AtlasAdsResult:
        payload: Dict[str, Any] = {
            "message": message,
            "external_user_id": external_user_id,
            "conversation_id": conversation_id,
        }
        if action_id:
            payload["action_id"] = action_id
        if confirmation_id:
            payload["confirmation_id"] = confirmation_id
        return self.request("POST", "/api/integrations/telegram/command", json_body=payload)
