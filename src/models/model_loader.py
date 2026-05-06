from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from src.config.config import settings


ABTE_ID2LABEL = {0: "O", 1: "B-Term", 2: "I-Term"}
ABTE_LABEL2ID = {"O": 0, "B-Term": 1, "I-Term": 2}
ABSC_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
ABSC_LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}


class LSTMTokenClassifierConfig(PretrainedConfig):
    model_type = "lstm-token-classifier"

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_labels: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        ignore_index: int = -100,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(id2label=id2label, label2id=label2id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.ignore_index = ignore_index


class LSTMForTokenClassification(PreTrainedModel):
    config_class = LSTMTokenClassifierConfig
    all_tied_weights_keys: List[str] = []

    def __init__(self, config: LSTMTokenClassifierConfig) -> None:
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=config.bidirectional,
        )
        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(lstm_output_dim, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        outputs, _ = self.lstm(x)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.num_labels),
                labels.reshape(-1),
                ignore_index=self.config.ignore_index,
            )

        return TokenClassifierOutput(loss=loss, logits=logits)



class LSTMSequenceClassifierConfig(PretrainedConfig):
    model_type = "lstm-sequence-classifier"

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_labels: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(id2label=id2label, label2id=label2id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional


class LSTMForSequenceClassification(PreTrainedModel):
    config_class = LSTMSequenceClassifierConfig
    all_tied_weights_keys: List[str] = []

    def __init__(self, config: LSTMSequenceClassifierConfig) -> None:
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=config.bidirectional,
        )
        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(lstm_output_dim, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        outputs, _ = self.lstm(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = outputs.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)



@dataclass
class LoadedModel:
    task: str
    model_type: str
    model: nn.Module
    tokenizer: object


class ABSAService:
    def __init__(self) -> None:
        self._cache: Dict[str, LoadedModel] = {}
        self.device = self._resolve_device(settings.device)
        self.models_root = Path(settings.saved_models_dir)

    def list_models(self) -> Dict[str, List[str]]:
        abte_models: List[str] = []
        absc_models: List[str] = []
        if not self.models_root.exists():
            return {"abte": abte_models, "absc": absc_models}

        for model_dir in sorted([p for p in self.models_root.iterdir() if p.is_dir()]):
            model_type = self._read_model_type(model_dir)
            name = model_dir.name
            if model_type in {"lstm-token-classifier", "token-classification"}:
                abte_models.append(name)
            elif model_type in {"lstm-sequence-classifier", "sequence-classification"}:
                absc_models.append(name)
            elif name.startswith("abte-"):
                abte_models.append(name)
            elif name.startswith("absc-"):
                absc_models.append(name)

        return {"abte": abte_models, "absc": absc_models}

    def predict_absa(
        self,
        sentence: str,
        abte_model_name: str,
        absc_model_name: str,
        term: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, object]:
        if device:
            self.device = self._resolve_device(device)
        if not sentence.strip():
            return {
                "sentence": sentence,
                "abte_model": abte_model_name,
                "absc_model": absc_model_name,
                "tokens": [],
                "labels": [],
                "terms": [],
                "results": [],
                "message": "Empty sentence.",
            }

        tokens: List[str] = []
        labels: List[str] = []
        if term is None or term.strip() == "":
            abte_loaded = self._load_model(abte_model_name, task="abte")
            tokens = sentence.strip().split()
            labels = self._predict_abte(abte_loaded, tokens)
            terms = self._extract_terms(tokens, labels)
        else:
            terms = [term.strip()]

        if not terms:
            return {
                "sentence": sentence,
                "abte_model": abte_model_name,
                "absc_model": absc_model_name,
                "tokens": tokens,
                "labels": labels,
                "terms": [],
                "results": [],
                "message": "No aspect term found.",
            }

        absc_loaded = self._load_model(absc_model_name, task="absc")
        results = [
            self._predict_absc(absc_loaded, sentence, aspect)
            for aspect in terms
        ]

        return {
            "sentence": sentence,
            "abte_model": abte_model_name,
            "absc_model": absc_model_name,
            "tokens": tokens,
            "labels": labels,
            "terms": terms,
            "results": results,
        }

    def _load_model(self, model_name: str, task: str) -> LoadedModel:
        cache_key = f"{task}:{model_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        model_dir = self.models_root / model_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. Place your weights under saved_models/."
            )

        model_type = self._read_model_type(model_dir)
        if task == "abte":
            loaded = self._load_abte_model(model_dir, model_type)
        elif task == "absc":
            loaded = self._load_absc_model(model_dir, model_type)
        else:
            raise ValueError(f"Unsupported task: {task}")

        self._cache[cache_key] = loaded
        return loaded

    def _load_abte_model(self, model_dir: Path, model_type: Optional[str]) -> LoadedModel:
        if model_type == "lstm-token-classifier":
            tokenizer = self._load_lstm_tokenizer(model_dir)
            config = LSTMTokenClassifierConfig.from_pretrained(str(model_dir))
            if not config.id2label:
                config.id2label = ABTE_ID2LABEL
                config.label2id = ABTE_LABEL2ID
            model = LSTMForTokenClassification(config).to(self.device)
            self._load_lstm_state_dict(model, model_dir)
            model.eval()
            return LoadedModel(task="abte", model_type="lstm", model=model, tokenizer=tokenizer)

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(self.device)
        model.eval()
        return LoadedModel(task="abte", model_type="hf", model=model, tokenizer=tokenizer)

    def _load_absc_model(self, model_dir: Path, model_type: Optional[str]) -> LoadedModel:
        if model_type == "lstm-sequence-classifier":
            tokenizer = self._load_lstm_tokenizer(model_dir)
            config = LSTMSequenceClassifierConfig.from_pretrained(str(model_dir))
            if not config.id2label:
                config.id2label = ABSC_ID2LABEL
                config.label2id = ABSC_LABEL2ID
            model = LSTMForSequenceClassification(config).to(self.device)
            self._load_lstm_state_dict(model, model_dir)
            model.eval()
            return LoadedModel(task="absc", model_type="lstm", model=model, tokenizer=tokenizer)

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device)
        model.eval()
        return LoadedModel(task="absc", model_type="hf", model=model, tokenizer=tokenizer)

    def _predict_abte(self, loaded_model: LoadedModel, tokens: List[str]) -> List[str]:
        tokenizer = loaded_model.tokenizer
        model = loaded_model.model

        if loaded_model.model_type == "lstm":
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                add_special_tokens=False,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=settings.lstm_max_length,
            )
            inputs = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                logits = model(**inputs).logits[0]
                pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()

            valid_len = min(len(tokens), len(pred_ids))
            return [ABTE_ID2LABEL[int(pred_ids[i])] for i in range(valid_len)]

        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
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
            word_to_label[word_idx] = ABTE_ID2LABEL[int(pred_ids[token_idx])]

        return [word_to_label.get(i, "O") for i in range(len(tokens))]

    def _predict_absc(self, loaded_model: LoadedModel, sentence: str, aspect: str) -> Dict[str, object]:
        tokenizer = loaded_model.tokenizer
        model = loaded_model.model

        if loaded_model.model_type == "lstm":
            encoding = tokenizer(
                sentence,
                aspect,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=settings.lstm_max_length,
            )
        else:
            encoding = tokenizer(
                sentence,
                aspect,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
        if "token_type_ids" in encoding:
            encoding.pop("token_type_ids")
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            probs = F.softmax(logits, dim=-1)
            score, pred_id = torch.max(probs, dim=-1)

        pred_id_int = int(pred_id.item())
        score_float = float(score.item())
        label_map = getattr(model, "config", None)
        if label_map is not None and getattr(label_map, "id2label", None):
            sentiment = label_map.id2label.get(pred_id_int, str(pred_id_int))
        else:
            sentiment = ABSC_ID2LABEL.get(pred_id_int, str(pred_id_int))

        return {
            "term": aspect,
            "sentiment_id": pred_id_int,
            "sentiment": sentiment,
            "sentiment_score": score_float,
        }

    def _read_model_type(self, model_dir: Path) -> Optional[str]:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data.get("model_type")
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "auto").lower()
        if normalized == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if normalized == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _load_lstm_tokenizer(model_dir: Path):
        configured_tokenizer_dir = Path(settings.lstm_tokenizer_dir)
        tokenizer_dir = (
            configured_tokenizer_dir
            if configured_tokenizer_dir.exists()
            else model_dir
        )
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

    def _load_lstm_state_dict(self, model: nn.Module, model_dir: Path) -> None:
        safetensors_path = model_dir / "model.safetensors"
        pytorch_path = model_dir / "pytorch_model.bin"

        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file as safetensors_load
            except ImportError as exc:
                raise ImportError(
                    "Missing safetensors dependency. Install safetensors or save as pytorch_model.bin."
                ) from exc
            state_dict = safetensors_load(str(safetensors_path))
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location=self.device)
        else:
            raise FileNotFoundError(
                f"Missing LSTM weights. Checked: {safetensors_path} and {pytorch_path}."
            )

        model.load_state_dict(state_dict)

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
