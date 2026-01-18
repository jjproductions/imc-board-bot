# src/my_project/settings.py
# from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, AnyUrl, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field

# ---- Project paths ----------------------------------------------------------
# Adjust parents[...] if your nesting differs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Determine which environment to load (default: development)
# OS env has highest precedence. If not set, we use "development".
ENV = os.getenv("ENVIRONMENT", "development").strip().lower()

# Select the appropriate .env file
# e.g., .env.development, .env.staging, .env.production
ENV_FILE = PROJECT_ROOT / f".env.{ENV}"
# Fallback: if the env-specific file is missing, try .env
if not ENV_FILE.exists():
    ENV_FILE = PROJECT_ROOT / ".env"

# ---- Settings sections ------------------------------------------------------


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True  # dev convenience
    workers: int = 1


class SecuritySettings(BaseModel):
    secret_key: str = "CHANGE_ME_dev_only"
    allow_origins: list[str] = ["*"]  # tighten in prod
    allow_credentials: bool = True
    allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: list[str] = ["*"]


class DatabaseSettings(BaseModel):
    url: Optional[AnyUrl] = None  # e.g. postgresql+asyncpg://user:pass@host/db
    echo: bool = False
    pool_size: int = 10


class VectorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # enables VECTOR__EMBEDDING_MODEL → embedding_model
        env_prefix="VECTOR__",  # make the mapping clean for all fields
        extra="ignore",
    )

    # Your default preferences
    embedding_model: str = Field(default="BAAI/bge-m3", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1024, ge=1)
    normalized: bool = True
    similarity_metric: Literal["cosine", "dot"] = "cosine"


class QdrantSettings(BaseModel):
    url: AnyUrl = "http://localhost:6333"
    api_key: Optional[str] = None  # None is fine for local
    default_collection: str = "board-policies"


class DoclingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # enables VECTOR__EMBEDDING_MODEL → embedding_model
        env_prefix="DOCLING__",  # make the mapping clean for all fields
        extra="ignore",
    )
    url: Optional[AnyUrl] = None
    artifact_path: Optional[str] = "NA"


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # enables CHUNKING__MAX_TOKENS → max_tokens
        env_prefix="CHUNKING__",  # make the mapping clean for all fields
        extra="ignore",
    )
    max_tokens: int = 800
    overlap_tokens: int = 100
    batch_size: int = 256
    # Optional: normalize spaced OCR letters (e.g. "O N E" -> "ONE"). Disabled by default.
    ocr_normalize_spaced_letters: bool = False


class AppSettings(BaseSettings):
    """
    Top-level configuration. Loads from:
    1) OS environment variables
    2) .env.<ENV> file (or .env fallback)
    3) Defaults in code (here)
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core
    app_name: str = "Board Policy Bot API"
    environment: Literal["development", "staging", "production"] = ENV  # bound to ENV
    debug: bool = ENV == "development"
    document_repository_path: Optional[str] = Field(
        default=None, env="DOCUMENT_REPOSITORY_PATH"
    )

    # Sections
    server: ServerSettings = ServerSettings()
    security: SecuritySettings = SecuritySettings()
    database: DatabaseSettings = DatabaseSettings()
    vector: VectorSettings = VectorSettings()
    qdrant: QdrantSettings = QdrantSettings()
    docling: DoclingSettings = DoclingSettings()
    chunking: ChunkingSettings = ChunkingSettings()

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # Derived / helpers
    @computed_field
    @property
    def is_prod(self) -> bool:
        return self.environment == "production"

    @computed_field
    @property
    def cors_kwargs(self) -> dict:
        return {
            "allow_origins": self.security.allow_origins,
            "allow_credentials": self.security.allow_credentials,
            "allow_methods": self.security.allow_methods,
            "allow_headers": self.security.allow_headers,
        }


# Instantiate a singleton and fail fast on invalid configuration
try:
    settings = AppSettings()
    print(
        f"models dir: {settings.models_dir}, \
          embedding_model: {settings.vector.embedding_model}, \
            docling_artifact_path: {settings.docling.artifact_path}"
    )  # --- IGNORE ---
    print("ENV_FILE:", ENV_FILE)  # --- IGNORE ---
except ValidationError as e:
    # Raise a clear error at startup if configuration is invalid
    raise RuntimeError(f"Invalid configuration in {ENV_FILE}: {e}") from e
