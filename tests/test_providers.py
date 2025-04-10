import pytest
from src.providers.base import ProviderFactory, OpenAIProvider
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_openai_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "Test response"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        },
        "model": "gpt-3.5-turbo"
    }

@pytest.mark.asyncio
async def test_provider_factory():
    # Test getting OpenAI provider
    provider = ProviderFactory.get_provider("openai")
    assert isinstance(provider, OpenAIProvider)
    
    # Test invalid provider
    with pytest.raises(ValueError):
        ProviderFactory.get_provider("invalid_provider")

@pytest.mark.asyncio
async def test_openai_provider(mock_openai_response):
    with patch("openai.ChatCompletion.acreate") as mock_create:
        # Setup mock
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))],
            usage=MagicMock(dict=lambda: {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }),
            model="gpt-3.5-turbo"
        )
        
        # Create provider
        provider = OpenAIProvider()
        
        # Test chat completion
        response = await provider.chat_completion(
            messages=[{"role": "user", "content": "Test message"}],
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Verify response
        assert response.content == "Test response"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == "openai"
        assert response.usage["total_tokens"] == 30
        
        # Verify API call
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        assert call_args["model"] == "gpt-3.5-turbo"
        assert call_args["temperature"] == 0.7
        assert call_args["messages"] == [{"role": "user", "content": "Test message"}]

@pytest.mark.asyncio
async def test_openai_provider_error():
    with patch("openai.ChatCompletion.acreate") as mock_create:
        # Setup mock to raise exception
        mock_create.side_effect = Exception("API Error")
        
        # Create provider
        provider = OpenAIProvider()
        
        # Test error handling
        with pytest.raises(Exception) as exc_info:
            await provider.chat_completion(
                messages=[{"role": "user", "content": "Test message"}]
            )
        
        assert "API Error" in str(exc_info.value) 