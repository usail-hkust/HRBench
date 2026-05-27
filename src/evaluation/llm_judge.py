"""
LLM-as-Judge evaluator using Azure GPT-4o.
Used for GPQA and other open-ended tasks where exact match is insufficient.
"""
import json
import re
import time
from typing import Optional

from src.utils.prompts import LLM_JUDGE_SYSTEM, LLM_JUDGE_USER_TEMPLATE


class LLMJudge:
    """
    GPT-4o based LLM-as-Judge for evaluating open-ended answers.
    Uses Azure OpenAI endpoint.
    """

    def __init__(
        self,
        deployment_name: str = "gpt4o-20241120",
        api_key: str = "e8e2fc75fd004f259a8078e3dacefa6c",
        api_base: str = "https://aipd-application-tech-north-central-us.openai.azure.com/",
        api_version: str = "2024-02-01",
        max_tokens: int = 256,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._llm = None

    def _ensure_client(self):
        if self._llm is None:
            try:
                from langchain_openai import AzureChatOpenAI
            except ImportError:
                from langchain.chat_models import AzureChatOpenAI
            self._llm = AzureChatOpenAI(
                azure_deployment=self.deployment_name,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                azure_endpoint=self.api_base,
                api_version=self.api_version,
            )

    def judge(
        self,
        problem: str,
        reference: str,
        response: str,
    ) -> bool:
        """
        Judge whether a response is correct given the problem and reference answer.

        Args:
            problem: The original problem text
            reference: Ground truth / reference answer
            response: Model's full response

        Returns:
            True if the judge deems the answer correct
        """
        self._ensure_client()
        try:
            from langchain.schema import HumanMessage, SystemMessage
        except ImportError:
            from langchain_core.messages import HumanMessage, SystemMessage

        # Truncate long responses to save tokens
        response_truncated = response[:3000] if len(response) > 3000 else response

        user_content = LLM_JUDGE_USER_TEMPLATE.format(
            problem=problem[:1500],
            reference=reference[:500],
            response=response_truncated,
        )

        messages = [
            SystemMessage(content=LLM_JUDGE_SYSTEM),
            HumanMessage(content=user_content),
        ]

        for attempt in range(self.max_retries):
            try:
                ai_message = self._llm.invoke(messages)
                result_text = ai_message.content.strip()

                # Parse JSON response
                json_match = re.search(r'\{[^}]+\}', result_text)
                if json_match:
                    result = json.loads(json_match.group())
                    return result.get("correct", False) is True

                # Fallback: check for keywords
                lower = result_text.lower()
                if "true" in lower or "correct" in lower:
                    return True
                return False

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"[LLMJudge] Failed after {self.max_retries} retries: {e}")
                    return False

    def judge_batch(
        self,
        problems: list,
        references: list,
        responses: list,
    ) -> list:
        """Judge a batch of responses sequentially."""
        return [
            self.judge(p, r, resp)
            for p, r, resp in zip(problems, references, responses)
        ]
