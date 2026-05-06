import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    saved_models_dir: str = os.getenv("SAVED_MODELS_DIR", "saved_models")
    default_abte_model_name: str = os.getenv("DEFAULT_ABTE_MODEL_NAME", "abte-minilm")
    default_absc_model_name: str = os.getenv("DEFAULT_ABSC_MODEL_NAME", "absc-minilm")
    device: str = os.getenv("INFERENCE_DEVICE", "auto")
    lstm_tokenizer_dir: str = os.getenv(
        "LSTM_TOKENIZER_DIR", "saved_models/restaurant_word_tokenizer"
    )
    lstm_max_length: int = int(os.getenv("LSTM_MAX_LENGTH", "128"))
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
