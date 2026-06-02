import json
import os
import re
import time
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible API client with retry logic."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str = "DASHSCOPE_API_KEY",
        api_key: str | None = None,
        model: str = "qwen2.5-72b-instruct",
        max_retries: int = 3,
    ):
        if api_key is None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise EnvironmentError(
                    f"API key not found. Set the {api_key_env} environment variable."
                )
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {e}")
                    raise


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown blocks and leading text."""
    # Try parsing directly
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
