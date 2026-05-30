"""OpenRouter LLM client — OpenAI-compatible interface."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from openai import OpenAI, AsyncOpenAI

from efa.config import OPENROUTER_BASE_URL, DEFAULT_MODEL, OPENROUTER_API_KEY


def _make_client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def _make_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = _make_client()
        self._async_client = _make_async_client()

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(model=self.model, messages=messages, temperature=temperature)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def complete_json(self, prompt: str, system: Optional[str] = None) -> dict:
        raw = self.complete(prompt, system=system, json_mode=True, temperature=0.2)
        return json.loads(raw)

    async def _async_complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def complete_parallel(
        self,
        prompts: list[str],
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> list[str]:
        """Run multiple completions concurrently via asyncio."""
        async def _run():
            tasks = [self._async_complete(p, system=system, temperature=temperature) for p in prompts]
            return list(await asyncio.gather(*tasks))

        try:
            asyncio.get_running_loop()
            # Already inside an async context — run in a new thread with its own loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _run()).result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run directly
            return asyncio.run(_run())
