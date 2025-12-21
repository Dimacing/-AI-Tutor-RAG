from __future__ import annotations

from dataclasses import dataclass, field
import time
import uuid
from threading import Lock

import requests


class LLMClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


@dataclass
class OpenAIClient(LLMClient):
    api_key: str
    model: str

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


@dataclass
class DeepSeekClient(LLMClient):
    api_key: str
    model: str
    base_url: str

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "")


@dataclass
class OllamaClient(LLMClient):
    base_url: str
    model: str
    timeout: int = 60

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        return message.get("content", "")


@dataclass
class GeminiClient(LLMClient):
    api_key: str
    model: str

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "")


@dataclass
class GigaChatClient(LLMClient):
    access_token: str | None
    model: str
    base_url: str
    auth_key: str | None = None
    ca_bundle: str | None = None
    verify_ssl: bool = True
    token_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    scope: str = "GIGACHAT_API_PERS"
    _token_expires_at: float = 0.0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def _requests_verify(self) -> bool | str:
        if self.ca_bundle:
            return self.ca_bundle
        return self.verify_ssl

    def _fetch_token(self) -> None:
        if not self.auth_key:
            raise RuntimeError("GIGACHAT_AUTH_KEY is not set.")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.auth_key}",
        }
        resp = requests.post(
            self.token_url,
            headers=headers,
            data={"scope": self.scope},
            verify=self._requests_verify(),
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("access_token") or payload.get("accessToken")
        if not token:
            raise RuntimeError("GIGACHAT access_token not found in response.")
        expires_at = payload.get("expires_at") or payload.get("expiresAt")
        expires_in = payload.get("expires_in") or payload.get("expiresIn")
        if expires_at:
            try:
                expires_value = float(expires_at)
                if expires_value > 1e12:
                    expires_value /= 1000.0
                self._token_expires_at = expires_value
            except (TypeError, ValueError):
                self._token_expires_at = time.time() + 1700
        elif expires_in:
            try:
                self._token_expires_at = time.time() + float(expires_in)
            except (TypeError, ValueError):
                self._token_expires_at = time.time() + 1700
        else:
            self._token_expires_at = time.time() + 1700
        self.access_token = token

    def _ensure_token(self) -> None:
        if self.access_token and not self.auth_key:
            return
        now = time.time()
        if self.access_token and self._token_expires_at and now < self._token_expires_at - 60:
            return
        with self._lock:
            now = time.time()
            if self.access_token and self._token_expires_at and now < self._token_expires_at - 60:
                return
            self._fetch_token()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_token()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        base = self.base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }
        resp = requests.post(
            f"{base}/api/v1/chat/completions",
            headers=headers,
            json=payload,
            verify=self._requests_verify(),
            timeout=60,
        )
        if resp.status_code in {401, 403} and self.auth_key:
            with self._lock:
                self._fetch_token()
            headers["Authorization"] = f"Bearer {self.access_token}"
            resp = requests.post(
                f"{base}/api/v1/chat/completions",
                headers=headers,
                json=payload,
                verify=self._requests_verify(),
                timeout=60,
            )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "")


def create_llm(provider: str, model: str | None = None, **kwargs) -> LLMClient | None:
    provider = provider.lower()
    if provider == "openai":
        api_key = kwargs.get("api_key")
        if not api_key or not model:
            return None
        return OpenAIClient(api_key=api_key, model=model)
    if provider in {"deepseek", "deepsick"}:
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        if not api_key or not model or not base_url:
            return None
        return DeepSeekClient(api_key=api_key, model=model, base_url=base_url)
    if provider == "ollama":
        base_url = kwargs.get("base_url")
        timeout = kwargs.get("timeout", 60)
        if not base_url or not model:
            return None
        return OllamaClient(base_url=base_url, model=model, timeout=int(timeout))
    if provider == "gemini":
        api_key = kwargs.get("api_key")
        if not api_key or not model:
            return None
        return GeminiClient(api_key=api_key, model=model)
    if provider == "gigachat":
        access_token = kwargs.get("access_token")
        auth_key = kwargs.get("auth_key")
        base_url = kwargs.get("base_url")
        ca_bundle = kwargs.get("ca_bundle")
        verify_ssl = kwargs.get("verify_ssl", True)
        if not base_url or not model:
            return None
        if not access_token and not auth_key:
            return None
        return GigaChatClient(
            access_token=access_token,
            auth_key=auth_key,
            model=model,
            base_url=base_url,
            ca_bundle=ca_bundle,
            verify_ssl=bool(verify_ssl),
        )
    return None
