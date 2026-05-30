"""OpenRouter LLM client — OpenAI-compatible interface."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI

from efa.config import OPENROUTER_BASE_URL, DEFAULT_MODEL, OPENROUTER_API_KEY


def _make_client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = _make_client()

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

    def complete_parallel(
        self,
        prompts: list[str],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_workers: int = 5,
    ) -> list[str]:
        """Run multiple completions concurrently via threads (sync clients).

        Uses ThreadPoolExecutor with the sync OpenAI client — avoids asyncio
        DNS issues on Windows while still achieving true concurrency for I/O-bound LLM calls.
        Each thread gets its own OpenAI client to avoid connection sharing issues.
        """
        def _call(prompt: str) -> str:
            client = _make_client()  # fresh client per thread
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as pool:
            future_to_idx = {pool.submit(_call, p): i for i, p in enumerate(prompts)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[ERROR: {e}]"
        return results
