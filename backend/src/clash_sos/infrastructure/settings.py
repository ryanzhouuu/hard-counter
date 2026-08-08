from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
