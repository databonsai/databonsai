# llm_providers/base_provider.py
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    @abstractmethod
    def __init__(
        self,
        model: str = "",
        temperature: float = 0,
    ):
        """
        Initializes the LLMProvider with an API key and retry parameters.

        Parameters:

        model (str): The default model to use for text generation.
        temperature (float): The temperature parameter for text generation.
        """
        pass

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Generates a text completion using the provider's API, with a given system and user prompt.
        This method should be decorated with retry logic to handle temporary failures.

        Parameters:
        system_prompt (str): The system prompt to provide context or instructions for the generation.
        user_prompt (str): The user's prompt, based on which the text completion is generated.
        max_tokens (int): The maximum number of tokens to generate in the response.

        Returns:
        str: The generated text completion.
        """
        pass
