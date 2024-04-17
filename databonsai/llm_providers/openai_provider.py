from typing import List
from openai import OpenAI
from .llm_provider import LLMProvider
import os
from functools import wraps
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()


class OpenAIProvider(LLMProvider):
    """
    A provider class to interact with OpenAI's API.
    Supports exponential backoff retries, since we'll often deal with large datasets.
    """

    def __init__(
        self,
        api_key: str = None,
        multiplier: int = 1,
        min_wait: int = 1,
        max_wait: int = 60,
        max_tries: int = 10,
        model: str = "gpt-4-turbo",
        temperature: float = 0,
    ):
        """
        Initializes the OpenAIProvider with an API key and retry parameters.

        Parameters:
        api_key (str): OpenAI API key.
        multiplier (int): The multiplier for the exponential backoff in retries.
        min_wait (int): The minimum wait time between retries.
        max_wait (int): The maximum wait time between retries.
        max_tries (int): The maximum number of attempts before giving up.
        temperature (float): The temperature parameter for text generation.
        """
        super().__init__()

        # Provider related configs
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided.")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        try:
            self.client.models.retrieve(model)
        except Exception as e:
            raise ValueError(f"Invalid OpenAI model: {model}") from e
        self.temperature = temperature
        self.input_tokens = 0
        self.output_tokens = 0

        # Retry related configs
        self.multiplier = multiplier
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.max_tries = max_tries

    def retry_with_exponential_backoff(method):
        """
        Decorator to apply retry logic with exponential backoff to an instance method.
        It captures the 'self' context to access instance attributes for retry configuration.
        """

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            retry_decorator = retry(
                wait=wait_exponential(
                    multiplier=self.multiplier, min=self.min_wait, max=self.max_wait
                ),
                stop=stop_after_attempt(self.max_tries),
            )
            return retry_decorator(method)(self, *args, **kwargs)

        return wrapper

    @retry_with_exponential_backoff
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens=1000, json=False
    ) -> str:
        """
        Generates a text completion using OpenAI's API, with a given system and user prompt.
        This method is decorated with retry logic to handle temporary failures.

        Parameters:
        system_prompt (str): The system prompt to provide context or instructions for the generation.
        user_prompt (str): The user's prompt, based on which the text completion is generated.

        Returns:
        str: The generated text completion.
        """
        if not system_prompt:
            raise ValueError("System prompt is required.")
        if not user_prompt:
            raise ValueError("User prompt is required.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}"},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_p=0.1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"} if json else {"type": "text"},
        )
        self.input_tokens += response.usage.prompt_tokens
        self.output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    # @retry_with_exponential_backoff
    def generate_batch(
        self, system_prompt: str, user_prompts: List[str], max_tokens=1000, json=False
    ) -> str:
        """
        Generates a text completion using OpenAI's API, with a given system and user prompt.
        This method is decorated with retry logic to handle temporary failures.

        Parameters:
        system_prompt (str): The system prompt to provide context or instructions for the generation.
        user_prompt (str): The user's prompt, based on which the text completion is generated.

        Returns:
        str: The generated text completion.
        """
        if not system_prompt:
            raise ValueError("System prompt is required.")
        if len(user_prompts) == 0:
            raise ValueError("User prompt is required.")
        input_data_prompt = ", ".join(
            [f"Content {idx+1}: {prompt}" for idx, prompt in enumerate(user_prompts)]
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_data_prompt},
        ]
        # print(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_p=0.1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"} if json else {"type": "text"},
        )
        self.input_tokens += response.usage.prompt_tokens
        self.output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content
