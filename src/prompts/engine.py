from typing import Dict, List, Optional
import json
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from ..core.config import get_settings
from loguru import logger

settings = get_settings()

class PromptEngine:
    def __init__(self):
        self.settings = get_settings()
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
        
        # Load prompt templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Load all prompt templates from the templates directory."""
        templates = {}
        for template_file in self.template_dir.glob("*.j2"):
            template_name = template_file.stem
            templates[template_name] = self.env.get_template(template_file.name)
        return templates
    
    def create_template(self, name: str, content: str) -> None:
        """Create a new prompt template."""
        template_path = self.template_dir / f"{name}.j2"
        with open(template_path, "w") as f:
            f.write(content)
        self.templates[name] = self.env.get_template(f"{name}.j2")
    
    def render_prompt(
        self,
        template_name: str,
        variables: Dict,
        context: Optional[Dict] = None
    ) -> str:
        """Render a prompt using the specified template and variables."""
        try:
            if template_name not in self.templates:
                raise ValueError(f"Template {template_name} not found")
            
            # Merge context with variables if provided
            if context:
                variables = {**variables, "context": context}
            
            return self.templates[template_name].render(**variables)
        except Exception as e:
            logger.error(f"Error rendering prompt: {str(e)}")
            raise
    
    def optimize_prompt(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        remove_redundancy: bool = True
    ) -> str:
        """Optimize a prompt by removing redundancy and ensuring length constraints."""
        try:
            # Remove redundant whitespace
            if remove_redundancy:
                prompt = " ".join(prompt.split())
            
            # Truncate if max_length is specified
            if max_length and len(prompt) > max_length:
                prompt = prompt[:max_length].rsplit(" ", 1)[0] + "..."
            
            return prompt
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            raise
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())
    
    def delete_template(self, name: str) -> None:
        """Delete a prompt template."""
        try:
            template_path = self.template_dir / f"{name}.j2"
            if template_path.exists():
                template_path.unlink()
                del self.templates[name]
        except Exception as e:
            logger.error(f"Error deleting template: {str(e)}")
            raise 