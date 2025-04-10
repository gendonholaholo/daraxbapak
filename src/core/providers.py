from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import os
import aiohttp
from .config import settings
from .logging import logger
from .errors import AGNOError

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the given prompt"""
        pass

    @abstractmethod
    async def get_embeddings(self, text: str) -> list:
        """Get embeddings for the given text"""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "gpt-3.5-turbo")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1000)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                messages = []
                if kwargs.get("system_message"):
                    messages.append({"role": "system", "content": kwargs["system_message"]})
                messages.append({"role": "user", "content": prompt})
                
                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise AGNOError(f"OpenAI API error: {error_data.get('error', {}).get('message', 'Unknown error')}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            raise AGNOError(f"OpenAI generation failed: {str(e)}")

    async def get_embeddings(self, text: str) -> list:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "text-embedding-ada-002",
                    "input": text
                }
                
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise AGNOError(f"OpenAI API error: {error_data.get('error', {}).get('message', 'Unknown error')}")
                    
                    result = await response.json()
                    return result["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {str(e)}")
            raise AGNOError(f"OpenAI embeddings failed: {str(e)}")

class GroqProvider(LLMProvider):
    def __init__(self, **kwargs):
        if not settings.GROQ_API_KEY:
            raise AGNOError("Groq API key not configured")
        self.api_key = settings.GROQ_API_KEY
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = kwargs.get("model", "llama-3.2-90b-vision-preview")

    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000)
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise AGNOError(f"Groq API error: {error_data.get('error', {}).get('message', 'Unknown error')}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq generation error: {str(e)}")
            raise AGNOError(f"Groq generation failed: {str(e)}")

    async def get_embeddings(self, text: str) -> list:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "text-embedding-3-small",
                    "input": text
                }
                
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise AGNOError(f"Groq API error: {error_data.get('error', {}).get('message', 'Unknown error')}")
                    
                    result = await response.json()
                    return result["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Groq embeddings error: {str(e)}")
            raise AGNOError(f"Groq embeddings failed: {str(e)}")

class ProviderFactory:
    _providers = {}
    
    @classmethod
    def get_provider(cls, provider_name: str = None) -> LLMProvider:
        if not provider_name:
            provider_name = settings.DEFAULT_PROVIDER
            
        if provider_name not in cls._providers:
            if provider_name == "openai":
                cls._providers[provider_name] = OpenAIProvider()
            elif provider_name == "groq":
                cls._providers[provider_name] = GroqProvider()
            else:
                raise AGNOError(f"Unsupported provider: {provider_name}")
                
        return cls._providers[provider_name]

provider_factory = ProviderFactory() 