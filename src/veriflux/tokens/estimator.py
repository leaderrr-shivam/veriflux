"""Token estimation for multi-turn chat messages and prompts.

Accounts for provider-specific message framing tokens (<|im_start|>, roles, function calls).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from veriflux.tokens.counter import count_tokens, estimate_tokens


class FastEstimator:
    """Estimates token consumption across complex conversational message payloads."""

    # Message framing overhead (OpenAI/Anthropic chat format typically adds 3-4 tokens per message)
    PER_MESSAGE_TOKENS = 3
    REPLY_PRIMER_TOKENS = 3

    def estimate(self, text: str, model: str = "gpt-4o") -> int:
        """Estimate token count for a raw string."""
        return count_tokens(text, model=model)

    def estimate_messages(self, messages: Sequence[Dict[str, Any]], model: str = "gpt-4o") -> int:
        """Estimate total tokens consumed by a standard chat completion message list.

        Args:
            messages: List of message dicts (e.g. `[{"role": "user", "content": "hello"}]`).
            model: Target model name.

        Returns:
            Total estimated tokens including chat framing delimiters.
        """
        total = self.REPLY_PRIMER_TOKENS

        for msg in messages:
            total += self.PER_MESSAGE_TOKENS
            role = str(msg.get("role", ""))
            total += count_tokens(role, model=model)

            content = msg.get("content", "")
            if isinstance(content, str):
                total += count_tokens(content, model=model)
            elif isinstance(content, list):
                # Multimodal or composite content blocks
                for block in content:
                    if isinstance(block, dict):
                        b_type = block.get("type")
                        if b_type == "text":
                            total += count_tokens(str(block.get("text", "")), model=model)
                        elif b_type == "image_url":
                            # Standard detail: low = 85 tokens, high = ~1000 tokens
                            total += 500
                        else:
                            total += count_tokens(json.dumps(block), model=model)
                    else:
                        total += count_tokens(str(block), model=model)

            # Tool/function call arguments
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    total += 10
                    fn = tc.get("function", {})
                    total += count_tokens(fn.get("name", ""), model=model)
                    total += count_tokens(fn.get("arguments", ""), model=model)

        return total

