"""LLM-based fraud reasoner — uses LLM API for prediction."""

import re
import torch
from typing import Optional


REASONING_SYSTEM_PROMPT = """You are a fraud detection analyst. You will be given evidence about a target node in a transaction/review network. Analyze the evidence and predict whether the target node is involved in fraud (1) or is legitimate (0).

Your response must follow this exact format:
Prediction: 0 or 1
Rationale: <one sentence explanation>

Be concise and analytical."""


class LLMReasoner:
    """LLM-based fraud reasoner that takes a prompt and returns a prediction."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 api_base: str = None, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature

    def _call_llm(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[AGEA] LLM reasoner call failed: {e}")
            return "Prediction: 0\nRationale: LLM unavailable"

    def predict(self, prompt: str) -> tuple:
        """Run LLM on the prompt and extract prediction.

        Returns: (prediction_prob, prediction_label, rationale, raw_response)
        """
        response = self._call_llm(prompt)

        # Parse prediction
        pred_match = re.search(r"Prediction:\s*([01])", response)
        if pred_match:
            label = int(pred_match.group(1))
            prob = 0.8 if label == 1 else 0.2
        else:
            # Fallback: look for 0 or 1
            if "1" in response.split("\n")[0]:
                label = 1
                prob = 0.8
            else:
                label = 0
                prob = 0.2

        # Parse rationale
        rat_match = re.search(r"Rationale:\s*(.+)", response)
        rationale = rat_match.group(1).strip() if rat_match else response

        return prob, label, rationale, response

    def predict_batch(self, prompts: list) -> list:
        """Predict for a batch of prompts."""
        results = []
        for p in prompts:
            results.append(self.predict(p))
        return results
