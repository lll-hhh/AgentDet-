"""
PIGuard prompt-injection detection adapter.

Default checkpoint: leolee99/PIGuard on Hugging Face.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from core.defense_config import load_json_config

logger = logging.getLogger(__name__)


@dataclass
class PIGuardDetectionResult:
    is_injection: bool
    score: float
    label: str
    raw: dict[str, Any]


class PIGuardService:
    """
    Thin wrapper around the HF PIGuard checkpoint.

    Default model:
      leolee99/PIGuard
    """

    def __init__(
        self,
        *,
        model_name: str = "leolee99/PIGuard",
        threshold: float = 0.5,
        trust_remote_code: bool = True,
        device: int = -1,
        positive_labels: set[str] | None = None,
    ):
        default_cfg = load_json_config("piguard_model.json", default={}) or {}
        configured_model = str(default_cfg.get("model_name", "") or "").strip()
        model_candidate = model_name
        if model_name == "leolee99/PIGuard" and configured_model:
            model_candidate = configured_model

        self.model_name = model_candidate
        if threshold == 0.5 and "threshold" in default_cfg:
            threshold = float(default_cfg.get("threshold", 0.5))
        if trust_remote_code is True and "trust_remote_code" in default_cfg:
            trust_remote_code = bool(default_cfg.get("trust_remote_code", True))
        if device == -1 and "device" in default_cfg:
            device = int(default_cfg.get("device", -1))

        self.threshold = float(threshold)
        self.trust_remote_code = bool(trust_remote_code)
        self.device = int(device)
        self.positive_labels = {
            x.strip().lower()
            for x in (
                positive_labels
                or {
                    "label_1",
                    "injection",
                    "malicious",
                    "prompt_injection",
                    "unsafe",
                    "attack",
                }
            )
        }
        self._classifier = None
        self._classifier_load_lock = threading.Lock()

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier

        with self._classifier_load_lock:
            if self._classifier is not None:
                return self._classifier

            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
            import torch

            device = self.device
            if device >= 0 and not torch.cuda.is_available():
                device = -1

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            self._classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                device=device,
            )
        return self._classifier

    @staticmethod
    def _extract_record(raw_output: Any) -> dict[str, Any]:
        if isinstance(raw_output, list) and raw_output:
            item = raw_output[0]
            if isinstance(item, dict):
                return dict(item)
        if isinstance(raw_output, dict):
            return dict(raw_output)
        return {}

    def classify(self, text: str) -> PIGuardDetectionResult:
        content = str(text or "").strip()
        if not content:
            return PIGuardDetectionResult(
                is_injection=False,
                score=0.0,
                label="",
                raw={},
            )

        classifier = self._get_classifier()
        raw = self._extract_record(classifier([content]))
        label = str(raw.get("label", "") or "")
        score = float(raw.get("score", 0.0) or 0.0)

        normalized = label.strip().lower()
        if normalized in self.positive_labels:
            is_injection = score >= self.threshold
        elif normalized in {"label_0", "benign", "safe", "clean", "not_injection"}:
            is_injection = False
        else:
            is_injection = score >= self.threshold and normalized not in {"", "unknown"}

        return PIGuardDetectionResult(
            is_injection=bool(is_injection),
            score=score,
            label=label,
            raw=raw,
        )
