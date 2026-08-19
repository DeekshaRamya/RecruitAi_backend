import time
import logging
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from app.core.config import settings

logger = logging.getLogger("recruitai-backend.azure_openai")

class AzureOpenAIClient:
    def __init__(self):
        self.endpoint = settings.AZURE_OPENAI_ENDPOINT
        self.api_key = settings.AZURE_OPENAI_API_KEY
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.api_version = settings.AZURE_OPENAI_API_VERSION or "2024-12-01-preview"

        if not self.endpoint:
            logger.error("Azure OpenAI Endpoint environment variable (AZURE_OPENAI_ENDPOINT) is not set.")
            raise ValueError("AZURE_OPENAI_ENDPOINT is not configured.")

        # Initialize the Azure OpenAI Client
        # Fall back to DefaultAzureCredential if API Key is not set or empty
        if self.api_key and self.api_key.strip():
            logger.info("Initializing Azure OpenAI client with API Key authentication.")
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        else:
            logger.info("Initializing Azure OpenAI client with Azure AD (DefaultAzureCredential) authentication.")
            self.credential = DefaultAzureCredential()
            self.token_provider = get_bearer_token_provider(
                self.credential,
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                azure_ad_token_provider=self.token_provider,
                api_version=self.api_version
            )

    async def generate_chat_completion(
        self, 
        prompt: str, 
        system_message: str = None,
        feature_name: str = "Assessment Evaluation",
        user_id = None,
        user_name: str = None,
        role: str = None,
        db = None
    ) -> str:
        """
        Sends a request to Azure OpenAI Chat Completions API.
        Logs usage into ai_usage_logs table.
        Returns the raw string content (expected to be JSON).
        """
        from datetime import datetime, timezone
        from app.services.ai_usage_service import AiUsageService

        if not self.deployment_name:
            logger.error("Azure OpenAI Deployment Name (AZURE_OPENAI_DEPLOYMENT_NAME) is not set.")
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME is not configured.")

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        req_time = datetime.now(timezone.utc)
        start_ticks = time.perf_counter()
        logger.info(f"Sending request to Azure OpenAI deployment '{self.deployment_name}' for feature '{feature_name}'")

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            response_time = int((time.perf_counter() - start_ticks) * 1000)  # ms
            logger.info(f"Azure Response Time: {response_time} ms")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Azure OpenAI returned an empty response.")

            # Extract token usage if available
            inp_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
            out_tokens = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
            tot_tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or (inp_tokens + out_tokens)

            await AiUsageService.log_usage(
                user_id=user_id,
                user_name=user_name,
                role=role,
                ai_provider="Azure OpenAI",
                model_name=self.deployment_name or "gpt-4o",
                feature_name=feature_name,
                request_id=getattr(response, "id", None),
                input_tokens=inp_tokens,
                output_tokens=out_tokens,
                total_tokens=tot_tokens,
                request_time=req_time,
                response_time_ms=response_time,
                status="Success",
                db=db
            )

            return content

        except Exception as e:
            response_time = int((time.perf_counter() - start_ticks) * 1000)  # ms
            logger.error(f"Error while calling Azure OpenAI: {str(e)}")
            await AiUsageService.log_usage(
                user_id=user_id,
                user_name=user_name,
                role=role,
                ai_provider="Azure OpenAI",
                model_name=self.deployment_name or "gpt-4o",
                feature_name=feature_name,
                request_time=req_time,
                response_time_ms=response_time,
                status="Failed",
                error_message=str(e),
                db=db
            )
            raise e
