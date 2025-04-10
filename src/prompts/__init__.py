from typing import Dict, List, Optional
from pathlib import Path
import json
from ..core.logging import logger
from ..core.errors import AGNOError

class PromptTemplate:
    def __init__(self, template: str, variables: List[str]):
        self.template = template
        self.variables = variables

    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise AGNOError(f"Missing required variable: {e}")

class PromptManager:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load prompt templates from files"""
        try:
            # Load interview prompts
            interview_prompts = {
                "greeting": PromptTemplate(
                    "Hello! I'm your interviewer. {greeting_message}",
                    ["greeting_message"]
                ),
                "question": PromptTemplate(
                    "Based on your experience with {topic}, {question}",
                    ["topic", "question"]
                ),
                "follow_up": PromptTemplate(
                    "That's interesting. Could you elaborate on {point}?",
                    ["point"]
                ),
                "closing": PromptTemplate(
                    "Thank you for your time. {closing_message}",
                    ["closing_message"]
                )
            }
            self.templates.update(interview_prompts)
            
            logger.info("Loaded prompt templates")
            
        except Exception as e:
            logger.error(f"Failed to load prompt templates: {str(e)}")
            raise AGNOError(f"Prompt template loading failed: {str(e)}")

    def get_template(self, template_name: str) -> PromptTemplate:
        """Get template by name"""
        if template_name not in self.templates:
            raise AGNOError(f"Template not found: {template_name}")
        return self.templates[template_name]

    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format prompt using template"""
        template = self.get_template(template_name)
        return template.format(**kwargs)

prompt_manager = PromptManager() 