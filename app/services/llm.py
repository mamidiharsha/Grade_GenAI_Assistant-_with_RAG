"""
LLM service using Google Gemini.
Handles prompt submission, response parsing, and error handling.
"""

import google.generativeai as genai
import time
from typing import Tuple
from app.config import get_settings
from app.utils.logger import logger


class LLMService:
    """
    Google Gemini LLM integration.
    Manages API calls with retry logic, timeout handling, and token tracking.
    """

    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model_name = settings.LLM_MODEL
        self._temperature = settings.LLM_TEMPERATURE
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=genai.GenerationConfig(
                temperature=self._temperature,
                max_output_tokens=1024,
            ),
        )
        logger.info(
            f"LLM service initialized: model={self._model_name}, "
            f"temperature={self._temperature}"
        )

    def generate_response(self, prompt: str) -> Tuple[str, int]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The complete prompt including context and history.
            
        Returns:
            Tuple of (response_text, token_count).
            
        Raises:
            Exception: If all retry attempts fail.
        """
        max_retries = 3
        tokens_used = 0

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self._model.generate_content(prompt)
                elapsed = time.time() - start_time

                # Extract response text
                if response and response.text:
                    response_text = response.text
                else:
                    response_text = (
                        "I apologize, but I was unable to generate a response. "
                        "Please try again."
                    )

                # Extract token usage if available
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    prompt_tokens = getattr(usage, "prompt_token_count", 0)
                    completion_tokens = getattr(
                        usage, "candidates_token_count", 0
                    )
                    tokens_used = prompt_tokens + completion_tokens

                    logger.info(
                        f"LLM response generated in {elapsed:.2f}s | "
                        f"Prompt tokens: {prompt_tokens} | "
                        f"Completion tokens: {completion_tokens} | "
                        f"Total: {tokens_used}"
                    )
                else:
                    # Estimate tokens if metadata not available
                    tokens_used = len(prompt.split()) + len(
                        response_text.split()
                    )
                    logger.info(
                        f"LLM response generated in {elapsed:.2f}s | "
                        f"Estimated tokens: {tokens_used}"
                    )

                return response_text, tokens_used

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(
                    f"LLM error (attempt {attempt + 1}/{max_retries}): "
                    f"Type={type(e).__name__}, Message={e}"
                )

                # Check for rate limit / quota errors (be specific to avoid false positives)
                if "rate_limit" in error_msg or "quota" in error_msg or "resource_exhausted" in error_msg or "429" in error_msg:
                    wait_time = (attempt + 1) * 10
                    logger.warning(
                        f"Rate limit hit, waiting {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)

                elif "api key" in error_msg or "api_key_invalid" in error_msg:
                    logger.error("Invalid Gemini API key")
                    raise ValueError(
                        "Invalid Gemini API key. Please verify your "
                        "GEMINI_API_KEY in .env file."
                    )

                elif "timeout" in error_msg:
                    logger.warning(
                        f"LLM timeout (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt == max_retries - 1:
                        raise TimeoutError(
                            "LLM request timed out after multiple attempts."
                        )
                    time.sleep(5)

                else:
                    # For any other error, retry with a short delay
                    if attempt == max_retries - 1:
                        raise

                    time.sleep(3)

        raise RuntimeError("Failed to generate LLM response after retries")
