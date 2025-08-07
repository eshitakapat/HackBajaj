"""Application configuration settings."""
from pathlib import Path
from typing import List, Union

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Insurance Helper API"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # File upload settings
    MAX_FILE_SIZE_MB: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    UPLOAD_DIR: str = Field(default="uploads", env="UPLOAD_DIR")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Vector Store (Pinecone)
    PINECONE_API_KEY: SecretStr = Field(..., env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = Field(..., env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX: str = Field(..., env="PINECONE_INDEX")
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # LLM Configuration
    OPENROUTER_API_KEY: SecretStr = Field(..., env="OPENROUTER_API_KEY")
    DEFAULT_LLM_MODEL: str = Field(
        default="mistralai/mistral-7b-instruct",
        env="DEFAULT_LLM_MODEL"
    )
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Project paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR_PATH: Path = BASE_DIR / "uploads"
    
    @validator("UPLOAD_DIR_PATH")
    def create_upload_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
