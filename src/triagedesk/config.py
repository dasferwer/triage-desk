from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://triagedesk:triagedesk@database:5432/triagedesk"
    jwt_secret: str = Field(
        default="local-demo-triagedesk-replace-before-deployment", min_length=32
    )
    token_minutes: int = Field(default=60, ge=1, le=1440)
    testing: bool = False
    model_dir: str = "models"
    amqp_url: str = "amqp://triagedesk:triagedesk@rabbitmq:5672/"
    worker_interval: float = Field(default=0.5, ge=0.05, le=60)
    lease_seconds: float = Field(default=15, ge=2, le=120)
    max_attempts: int = Field(default=3, ge=1, le=10)
    inference_delay_seconds: float = Field(default=0, ge=0, le=30)


settings = Settings()
