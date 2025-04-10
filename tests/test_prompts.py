import pytest
from src.prompts import prompt_manager, PromptTemplate, AGNOError

def test_prompt_template():
    """Test prompt template functionality"""
    template = PromptTemplate(
        "Hello {name}, welcome to {place}!",
        ["name", "place"]
    )
    
    # Test successful formatting
    result = template.format(name="John", place="AGNO")
    assert result == "Hello John, welcome to AGNO!"
    
    # Test missing variable
    with pytest.raises(AGNOError):
        template.format(name="John")

def test_prompt_manager():
    """Test prompt manager functionality"""
    # Test getting existing template
    template = prompt_manager.get_template("greeting")
    assert isinstance(template, PromptTemplate)
    
    # Test getting non-existent template
    with pytest.raises(AGNOError):
        prompt_manager.get_template("non_existent")
    
    # Test formatting prompts
    greeting = prompt_manager.format_prompt(
        "greeting",
        greeting_message="Welcome to the interview!"
    )
    assert "Welcome to the interview!" in greeting
    
    question = prompt_manager.format_prompt(
        "question",
        topic="Python",
        question="what do you know about it?"
    )
    assert "Python" in question
    assert "what do you know about it?" in question
    
    follow_up = prompt_manager.format_prompt(
        "follow_up",
        point="your experience"
    )
    assert "your experience" in follow_up
    
    closing = prompt_manager.format_prompt(
        "closing",
        closing_message="Thank you for your time!"
    )
    assert "Thank you for your time!" in closing 