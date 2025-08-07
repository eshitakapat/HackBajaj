"""Application configuration settings."""
from pathlib import Path
from typing import List, Union, Optional
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings

class Settings(PydanticBaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Insurance Helper API"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["*"],  # Allow all origins for now
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
    
    # Vector Store (Pinecone) - Optional
    PINECONE_API_KEY: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX: str = Field(default="insurance-helper", env="PINECONE_INDEX")
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # LLM Configuration - Optional
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = Field(
        default="mistralai/mistral-small-3.1-24b-instruct:free",
        env="OPENROUTER_MODEL"
    )
    OPENROUTER_API_BASE: str = Field(
        default="https://openrouter.ai/api/v1",
        env="OPENROUTER_API_BASE"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        env="EMBEDDING_MODEL"
    )
    
    # Database - Optional
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Project paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR_PATH: Path = Field(default=None)
    
    @validator("UPLOAD_DIR_PATH", pre=True, always=True)
    def create_upload_dir(cls, v, values):
        base_dir = values.get('BASE_DIR', Path(__file__).resolve().parent.parent)
        upload_dir = base_dir / values.get('UPLOAD_DIR', 'uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()