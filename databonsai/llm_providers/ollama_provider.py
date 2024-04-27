from typing import Optional
from ollama import Client, chat
from .llm_provider import LLMProvider
from databonsai.utils.logs import logger


class OllamaProvider(LLMProvider):
    """
    A provider class to interact with Ollama's API.
    Supports exponential backoff retries, since we'll often deal with large datasets.
    """

    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0,
        host: Optional[str] = None,
    ):
        """
        Initializes the OllamaProvider with an optional Ollama client or host, and retry parameters.

        Parameters:
        model (str): The default model to use for text generation.
        temperature (float): The temperature parameter for text generation.
        host (str): The host URL for the Ollama API.
        """
        # Provider related configs
        self.model = model
        self.temperature = temperature

        if host:
            self.client = Client(host=host)
        else:
            self.client = None

    def _chat(self, messages, options):
        if self.client:
            return self.client.chat(
                model=self.model, messages=messages, options=options
            )
        else:
            return chat(model=self.model, messages=messages, options=options)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens=1000) -> str:
        """
        Generates a text completion using Ollama's API, with a given system and user prompt.
        This method is decorated with retry logic to handle temporary failures.
        Parameters:
        system_prompt (str): The system prompt to provide context or instructions for the generation.
        user_prompt (str): The user's prompt, based on which the text completion is generated.
        max_tokens (int): The maximum number of tokens to generate in the response.
        Returns:
        str: The generated text completion.
        """
        if not system_prompt:
            raise ValueError("System prompt is required.")
        if not user_prompt:
            raise ValueError("User prompt is required.")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = self._chat(
                messages,
                options={"temperature": self.temperature, "num_predict": max_tokens},
            )
            completion = response["message"]["content"]

            return completion
        except Exception as e:
            logger.warning(f"Error occurred during generation: {str(e)}")
            raise
