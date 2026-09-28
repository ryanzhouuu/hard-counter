from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLASH_SOS_",
        extra="ignore",
    )

    app_name: str = "Clash SoS API"
    database_url: str = "postgresql+psycopg://clash_sos:clash_sos@localhost:5432/clash_sos"
    log_level: str = "INFO"
    royale_api_token: SecretStr | None = Field(
        default=None, validation_alias="CLASH_ROYALE_API_TOKEN"
    )
    royale_api_base_url: str = "https://api.clashroyale.com/v1/"
    active_model_path: Path = Path("models/kaggle-v6-ranked16-attention-v1")
    live_catalog_path: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
