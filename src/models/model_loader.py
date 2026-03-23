from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from src.config.config import settings


ID2LABEL = {0: "O", 1: "B-Term", 2: "I-Term"}


class ABTELSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        logits = self.fc(outputs)
        return {"logits": logits}


@dataclass
class LoadedModel:
    model_type: str
    model: nn.Module
    tokenizer: object


class ABTEModelService:
    def __init__(self) -> None:
        self._cache: Dict[str, LoadedModel] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_root = Path(settings.saved_models_dir)

    def list_models(self) -> List[str]:
        if not self.models_root.exists():
            return []
        return sorted([p.name for p in self.models_root.iterdir() if p.is_dir()])

    def load(self, model_name: str) -> LoadedModel:
        if model_name in self._cache:
            return self._cache[model_name]

        model_dir = self.models_root / model_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. Place your weights under saved_models/."
            )

        is_hf = (model_dir / "config.json").exists()
        if is_hf:
            loaded = self._load_huggingface_model(model_dir)
        elif model_name == "lstm_model":
            loaded = self._load_lstm_model(model_dir)
        else:
            raise ValueError(
                f"Unsupported model '{model_name}'. Supported: distilbert_model (HF) and lstm_model."
            )

        self._cache[model_name] = loaded
        return loaded

    def predict(self, text: str, model_name: str) -> Dict[str, object]:
        loaded_model = self.load(model_name)
        words = text.strip().split()
        if not words:
            return {"model_name": model_name, "tokens": [], "labels": [], "terms": []}

        if loaded_model.model_type == "hf":
            labels = self._predict_hf(loaded_model, words)
        else:
            labels = self._predict_lstm(loaded_model, words)

        terms = self._extract_terms(words, labels)
        return {
            "model_name": model_name,
            "tokens": words,
            "labels": labels,
            "terms": terms,
        }

    def _load_huggingface_model(self, model_dir: Path) -> LoadedModel:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(self.device)
        model.eval()
        return LoadedModel(model_type="hf", model=model, tokenizer=tokenizer)

    def _load_lstm_model(self, model_dir: Path) -> LoadedModel:
        state_path = model_dir / "model_state_dict.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing LSTM weights at: {state_path}")

        configured_tokenizer_dir = Path(settings.lstm_tokenizer_dir)
        if configured_tokenizer_dir.exists():
            tokenizer_dir = configured_tokenizer_dir
        elif (model_dir / "tokenizer.json").exists():
            tokenizer_dir = model_dir
        else:
            raise FileNotFoundError(
                "Missing tokenizer for LSTM. "
                f"Checked: {configured_tokenizer_dir} and {model_dir}. "
                "Set LSTM_TOKENIZER_DIR or place tokenizer files in lstm_model/."
            )

        tokenizer = self._load_lstm_tokenizer(tokenizer_dir)
        pad_token_id = tokenizer.pad_token_id or 0
        model = ABTELSTMClassifier(
            vocab_size=len(tokenizer.get_vocab()),
            num_classes=3,
            pad_idx=pad_token_id,
        ).to(self.device)
        state_dict = torch.load(state_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return LoadedModel(model_type="lstm", model=model, tokenizer=tokenizer)

    def _predict_hf(self, loaded_model: LoadedModel, words: List[str]) -> List[str]:
        tokenizer = loaded_model.tokenizer
        model = loaded_model.model

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        # DistilBERT does not use token_type_ids.
        if "token_type_ids" in encoding:
            encoding.pop("token_type_ids")
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0]
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()

        word_ids = encoding.word_ids(batch_index=0)
        word_to_label: Dict[int, str] = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in word_to_label:
                continue
            word_to_label[word_idx] = ID2LABEL[int(pred_ids[token_idx])]

        return [word_to_label.get(i, "O") for i in range(len(words))]

    def _predict_lstm(self, loaded_model: LoadedModel, words: List[str]) -> List[str]:
        tokenizer = loaded_model.tokenizer
        model = loaded_model.model

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=settings.lstm_max_length,
        )
        input_ids = encoding["input_ids"].to(self.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids)["logits"][0]
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()

        valid_len = min(len(words), len(pred_ids))
        return [ID2LABEL[int(pred_ids[i])] for i in range(valid_len)]

    @staticmethod
    def _load_lstm_tokenizer(tokenizer_dir: Path):
        try:
            return AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
        except Exception:
            tokenizer_json = tokenizer_dir / "tokenizer.json"
            if not tokenizer_json.exists():
                raise
            return PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_json),
                unk_token="<unk>",
                pad_token="<pad>",
            )

    @staticmethod
    def _extract_terms(tokens: List[str], labels: List[str]) -> List[str]:
        terms: List[str] = []
        current_term: List[str] = []

        for token, label in zip(tokens, labels):
            if label == "B-Term":
                if current_term:
                    terms.append(" ".join(current_term))
                current_term = [token]
            elif label == "I-Term":
                if current_term:
                    current_term.append(token)
                else:
                    current_term = [token]
            else:
                if current_term:
                    terms.append(" ".join(current_term))
                    current_term = []

        if current_term:
            terms.append(" ".join(current_term))

        return terms
