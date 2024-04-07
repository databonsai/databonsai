# llm_providers/base_provider.py
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    @abstractmethod
    def __init__(
        self,
        api_key: Optional[str] = None,
        multiplier: int = 1,
        min_wait: int = 1,
        max_wait: int = 60,
        max_tries: int = 10,
        model: str = "",
        temperature: float = 0,
    ):
        """
        Initializes the LLMProvider with an API key and retry parameters.

        Parameters:
        api_key (Optional[str]): API key for the provider.
        multiplier (int): The multiplier for the exponential backoff in retries.
        min_wait (int): The minimum wait time between retries.
        max_wait (int): The maximum wait time between retries.
        max_tries (int): The maximum number of attempts before giving up.
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

    @abstractmethod
    def retry_with_exponential_backoff(self, method):
        """
        Decorator to apply retry logic with exponential backoff to an instance method.
        It captures the 'self' context to access instance attributes for retry configuration.
        """
        pass
