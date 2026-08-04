"""
DeBERTa NLI HTTP Client.

Follows the same pattern as clip_reward_service.py (MedImageInsightsClient).
Communicates with the remote DeBERTa FastAPI service for entailment scoring.
"""

import time
from typing import List

import requests


class DebertaClient:
    """HTTP client for the remote DeBERTa NLI service."""

    def __init__(
        self,
        service_url: str = "http://localhost:8000",
        timeout: int = 60,
        max_batch_size: int = 32,
        retry_count: int = 3,
    ):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.retry_count = retry_count
        self._session = requests.Session()

    def check_health(self) -> bool:
        try:
            resp = self._session.get(f"{self.service_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _request_with_retry(self, endpoint: str, payload: dict) -> dict:
        last_error = None
        for attempt in range(self.retry_count):
            try:
                resp = self._session.post(
                    f"{self.service_url}{endpoint}",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"DeBERTa request timeout (attempt {attempt + 1}/{self.retry_count})")
            except requests.exceptions.ConnectionError as e:
                last_error = e
                print(f"DeBERTa connection error (attempt {attempt + 1}/{self.retry_count})")
            except requests.exceptions.HTTPError as e:
                last_error = e
                print(f"DeBERTa HTTP error: {e.response.status_code} (attempt {attempt + 1}/{self.retry_count})")
            except Exception as e:
                last_error = e
                print(f"DeBERTa request error: {e} (attempt {attempt + 1}/{self.retry_count})")

            if attempt < self.retry_count - 1:
                time.sleep(2 ** attempt)

        raise RuntimeError(f"DeBERTa service failed after {self.retry_count} attempts: {last_error}")

    def _scores(self, endpoint: str, texts1: List[str], texts2: List[str]) -> List[float]:
        if not texts1:
            return []

        all_scores = []
        for i in range(0, len(texts1), self.max_batch_size):
            batch_t1 = texts1[i : i + self.max_batch_size]
            batch_t2 = texts2[i : i + self.max_batch_size]
            pairs = [{"text1": t1, "text2": t2} for t1, t2 in zip(batch_t1, batch_t2)]
            result = self._request_with_retry(endpoint, {"pairs": pairs})
            all_scores.extend(result["scores"])

        return all_scores

    def get_entailment_scores(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Compute P(entailment) for each (text1, text2) pair.

        Args:
            texts1: List of premise texts
            texts2: List of hypothesis texts

        Returns:
            List of entailment probabilities in [0, 1]
        """
        return self._scores("/entailment_batch", texts1, texts2)

    def get_contradiction_scores(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Compute P(contradiction) for each (premise, hypothesis) pair.

        This is the only head ThinkAlign uses. Entailment from this MNLI checkpoint is
        near the floor on medical text (P=0.033 for a textbook-valid inference),
        but contradiction is reliable: a laterality flip scores 0.96 where a
        paraphrase of the same finding scores 0.02.
        """
        return self._scores("/contradiction_batch", texts1, texts2)
