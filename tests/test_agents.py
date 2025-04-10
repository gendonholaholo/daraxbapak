import pytest
from src.agents import agent_factory, BaseAgent, InterviewerAgent
from src.core.errors import AGNOError

def test_agent_factory():
    """Test agent factory"""
    # Test getting interviewer agent
    agent = agent_factory.get_agent("interviewer")
    assert isinstance(agent, InterviewerAgent)
    
    # Test getting unsupported agent
    with pytest.raises(AGNOError):
        agent_factory.get_agent("unsupported")

@pytest.mark.asyncio
async def test_interviewer_agent():
    """Test interviewer agent functionality"""
    agent = agent_factory.get_agent("interviewer")
    
    # Test initial greeting
    response = await agent.process({
        "session_id": "test_session",
        "message": "Hello"
    })
    assert "response" in response
    assert "context" in response
    assert "prompt_type" in response
    assert response["prompt_type"] == "greeting"
    
    # Test question
    response = await agent.process({
        "session_id": "test_session",
        "message": "Tell me about your experience",
        "topic": "Python programming",
        "question": "what projects have you worked on?"
    })
    assert response["prompt_type"] == "question"
    
    # Test follow-up
    response = await agent.process({
        "session_id": "test_session",
        "message": "Can you elaborate?",
        "point": "your experience"
    })
    assert response["prompt_type"] == "follow_up"
    
    # Test context management
    context = await agent.get_context("test_session")
    assert len(context) == 3  # Three interactions
    
    # Test clearing context
    await agent.clear_context("test_session")
    context = await agent.get_context("test_session")
    assert len(context) == 0 