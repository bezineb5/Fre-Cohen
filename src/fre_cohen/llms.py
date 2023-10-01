from typing import Type

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BaseOutputParser

from fre_cohen.configuration import Config


def get_openai_llm(config: Config) -> ChatOpenAI:
    """Returns the OpenAI LLM"""
    # Timeout is in seconds
    return ChatOpenAI(
        openai_api_key=config.openai_api_key,
        model="gpt-3.5-turbo",
        request_timeout=config.request_timeout_seconds,
    )


def get_llm(config: Config) -> BaseChatModel:
    """Returns the LLM"""
    return get_openai_llm(config)


def build_llm_chain(
    config: Config,
    pydantic_message: Type[BaseModel],
    prompts: list[BaseMessagePromptTemplate],
) -> LLMChain:
    """Builds the LLM chain"""
    return _JsonLLMChain(
        config=config, pydantic_message=pydantic_message, prompts=prompts
    ).llm_chain()


class _JsonLLMChain:
    def __init__(
        self,
        config: Config,
        pydantic_message: Type[BaseModel],
        prompts: list[BaseMessagePromptTemplate],
    ) -> None:
        super().__init__()
        self._config = config
        self._pydantic_message = pydantic_message
        self._prompts = prompts

    def _prompt_template(
        self, parser: BaseOutputParser, prompts: list[BaseMessagePromptTemplate]
    ) -> ChatPromptTemplate:
        """Returns the prompt template"""
        all_prompts = [] if not prompts else list(prompts)

        all_prompts.append(
            SystemMessagePromptTemplate.from_template("{format_instructions}")
        )

        return ChatPromptTemplate.from_messages(all_prompts).partial(
            format_instructions=parser.get_format_instructions()
        )

    def _output_parser(self) -> BaseOutputParser:
        """Returns the pydantic output parser"""
        return PydanticOutputParser(
            pydantic_object=self._pydantic_message,
        )

    def llm_chain(self) -> LLMChain:
        """Returns the LLM chain"""
        parser = self._output_parser()
        return LLMChain(
            llm=get_llm(self._config),
            prompt=self._prompt_template(parser, self._prompts),
            output_parser=parser,
        )
