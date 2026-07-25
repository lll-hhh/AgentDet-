"""
ProtectAI prompt-injection detection adapter.

Default checkpoint: protectai/deberta-v3-base-prompt-injection-v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from core.defense_config import load_json_config


@dataclass
class ProtectAIDetectionResult:
    is_injection: bool
    score: float
    label: str
    raw: dict[str, Any]


class ProtectAIService:
    """
    Prompt-injection detector backed by ProtectAI model/scanner.

    Preferred path:
      llm_guard.input_scanners.PromptInjection
    Fallback path:
      HF model "protectai/deberta-v3-base-prompt-injection-v2"
    """

    def __init__(
        self,
        *,
        threshold: float = 0.92,
        use_onnx: bool = False,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
        device: int = -1,
    ):
        default_cfg = load_json_config("protectai_model.json", default={}) or {}
        configured_model = str(default_cfg.get("model_name", "") or "").strip()
        model_candidate = model_name
        if model_name == "protectai/deberta-v3-base-prompt-injection-v2" and configured_model:
            model_candidate = configured_model

        if threshold == 0.92 and "threshold" in default_cfg:
            threshold = float(default_cfg.get("threshold", 0.92))
        if use_onnx is False and "use_onnx" in default_cfg:
            use_onnx = bool(default_cfg.get("use_onnx", False))
        if device == -1 and "device" in default_cfg:
            device = int(default_cfg.get("device", -1))

        self.threshold = float(threshold)
        self.use_onnx = bool(use_onnx)
        self.model_name = model_candidate
        self.device = int(device)
        self._scanner = None
        self._classifier = None
        self._scanner_load_lock = threading.Lock()
        self._classifier_load_lock = threading.Lock()

    def _get_scanner(self):
        if self._scanner is not None:
            return self._scanner

        with self._scanner_load_lock:
            if self._scanner is not None:
                return self._scanner
            try:
                from llm_guard.input_scanners import PromptInjection
                from llm_guard.model import Model

                model = None
                local_model_path = Path(self.model_name)
                if self.use_onnx and local_model_path.exists():
                    model = Model(
                        path=str(local_model_path),
                        onnx_path=str(local_model_path),
                        onnx_subfolder="onnx",
                        onnx_filename="model.onnx",
                        pipeline_kwargs={
                            "return_token_type_ids": False,
                            "max_length": 512,
                            "truncation": True,
                        },
                    )

                self._scanner = PromptInjection(
                    model=model,
                    threshold=self.threshold,
                    use_onnx=self.use_onnx,
                )
                return self._scanner
            except Exception:
                return None

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier

        with self._classifier_load_lock:
            if self._classifier is not None:
                return self._classifier

            from transformers import pipeline
            from transformers import AutoTokenizer
            import torch

            device = self.device
            if device >= 0 and not torch.cuda.is_available():
                device = -1

            local_model_path = Path(self.model_name)
            if self.use_onnx and local_model_path.exists():
                import json
                import numpy as np
                import onnxruntime as ort

                tokenizer = AutoTokenizer.from_pretrained(str(local_model_path / "onnx"))
                config_path = local_model_path / "onnx" / "config.json"
                id2label = {}
                if config_path.exists():
                    with config_path.open("r", encoding="utf-8") as handle:
                        id2label = (json.load(handle).get("id2label") or {})

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if device >= 0
                    else ["CPUExecutionProvider"]
                )
                session = ort.InferenceSession(
                    str(local_model_path / "onnx" / "model.onnx"),
                    providers=providers,
                )
                input_names = {item.name for item in session.get_inputs()}

                class _OnnxClassifier:
                    def __call__(self, texts):
                        encoded = tokenizer(
                            list(texts),
                            truncation=True,
                            max_length=512,
                            padding=True,
                            return_tensors="np",
                            return_token_type_ids=False,
                        )
                        feed = {k: v for k, v in encoded.items() if k in input_names}
                        logits = session.run(None, feed)[0]
                        logits = np.asarray(logits)
                        logits = logits - logits.max(axis=-1, keepdims=True)
                        probs = np.exp(logits)
                        probs = probs / probs.sum(axis=-1, keepdims=True)
                        rows = []
                        for row in probs:
                            idx = int(row.argmax())
                            rows.append(
                                {
                                    "label": str(id2label.get(str(idx), f"LABEL_{idx}")),
                                    "score": float(row[idx]),
                                }
                            )
                        return rows

                self._classifier = _OnnxClassifier()
                return self._classifier

            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
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

    def classify(self, text: str) -> ProtectAIDetectionResult:
        content = str(text or "").strip()
        if not content:
            return ProtectAIDetectionResult(
                is_injection=False,
                score=0.0,
                label="",
                raw={},
            )

        scanner = self._get_scanner()
        if scanner is not None:
            _, is_valid, risk_score = scanner.scan(content)
            score = float(risk_score if risk_score >= 0 else 0.0)
            return ProtectAIDetectionResult(
                is_injection=not bool(is_valid),
                score=score,
                label="llm_guard",
                raw={"is_valid": bool(is_valid), "risk_score": score},
            )

        classifier = self._get_classifier()
        raw = self._extract_record(classifier([content]))
        label = str(raw.get("label", "") or "")
        score = float(raw.get("score", 0.0) or 0.0)
        normalized = label.strip().lower()
        is_injection = (
            ("inject" in normalized or "attack" in normalized or normalized in {"label_1", "unsafe"})
            and score >= self.threshold
        )
        return ProtectAIDetectionResult(
            is_injection=bool(is_injection),
            score=score,
            label=label,
            raw=raw,
        )
