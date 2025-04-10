import pytest
from src.prompts.engine import PromptEngine
import os
from pathlib import Path

@pytest.fixture
def prompt_engine():
    return PromptEngine()

@pytest.fixture
def sample_template():
    return """
    You are a helpful assistant.
    Context: {{ context.text if context else 'No context' }}
    User: {{ user_input }}
    """

def test_create_template(prompt_engine, sample_template, tmp_path):
    # Create a template
    template_name = "test_template"
    prompt_engine.create_template(template_name, sample_template)
    
    # Verify template was created
    assert template_name in prompt_engine.templates
    template_path = prompt_engine.template_dir / f"{template_name}.j2"
    assert template_path.exists()
    
    # Verify template content
    with open(template_path, "r") as f:
        content = f.read()
        assert content.strip() == sample_template.strip()

def test_render_prompt(prompt_engine, sample_template):
    # Create and test template
    template_name = "test_render"
    prompt_engine.create_template(template_name, sample_template)
    
    # Test rendering without context
    variables = {"user_input": "Hello"}
    result = prompt_engine.render_prompt(template_name, variables)
    assert "No context" in result
    assert "Hello" in result
    
    # Test rendering with context
    context = {"text": "Some context"}
    result = prompt_engine.render_prompt(template_name, variables, context)
    assert "Some context" in result
    assert "Hello" in result

def test_optimize_prompt(prompt_engine):
    # Test prompt optimization
    original_prompt = "  This    is    a    test    prompt    with    extra    spaces    "
    optimized = prompt_engine.optimize_prompt(original_prompt)
    
    # Verify whitespace was removed
    assert "  " not in optimized
    assert optimized == "This is a test prompt with extra spaces"
    
    # Test length constraint
    max_length = 20
    truncated = prompt_engine.optimize_prompt(original_prompt, max_length=max_length)
    assert len(truncated) <= max_length
    assert truncated.endswith("...")

def test_get_available_templates(prompt_engine, sample_template):
    # Create some templates
    templates = ["template1", "template2", "template3"]
    for name in templates:
        prompt_engine.create_template(name, sample_template)
    
    # Get available templates
    available = prompt_engine.get_available_templates()
    
    # Verify all templates are listed
    assert all(name in available for name in templates)

def test_delete_template(prompt_engine, sample_template):
    # Create a template
    template_name = "test_delete"
    prompt_engine.create_template(template_name, sample_template)
    
    # Verify template exists
    assert template_name in prompt_engine.templates
    template_path = prompt_engine.template_dir / f"{template_name}.j2"
    assert template_path.exists()
    
    # Delete template
    prompt_engine.delete_template(template_name)
    
    # Verify template was deleted
    assert template_name not in prompt_engine.templates
    assert not template_path.exists()

def test_error_handling(prompt_engine):
    # Test rendering non-existent template
    with pytest.raises(ValueError) as exc_info:
        prompt_engine.render_prompt("nonexistent", {})
    assert "Template nonexistent not found" in str(exc_info.value)
    
    # Test template creation with invalid name
    with pytest.raises(Exception):
        prompt_engine.create_template("", "content")
    
    # Test template deletion of non-existent template
    with pytest.raises(Exception):
        prompt_engine.delete_template("nonexistent") 