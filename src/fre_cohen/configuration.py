"""Configuration for the fre_cohen package."""
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration class, with the service keys"""

    openai_api_key: str = Field(
        validation_alias=AliasChoices("openai_api_key", "OPENAI_API_KEY")
    )
    request_timeout_seconds: float = Field(120.0)

    model_config = SettingsConfigDict(env_prefix="fre_cohen_")
