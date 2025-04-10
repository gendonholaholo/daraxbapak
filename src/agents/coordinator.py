from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from ..core.config import get_settings
from ..providers.base import BaseProvider, ProviderFactory
from ..context.manager import ContextManager
from ..prompts.engine import PromptEngine
from loguru import logger

settings = get_settings()

class AgentRole(Enum):
    ANALYZER = "analyzer"
    RESEARCHER = "researcher"
    SUMMARIZER = "summarizer"
    VALIDATOR = "validator"

@dataclass
class AgentTask:
    role: AgentRole
    input_data: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 1

class AgentCoordinator:
    def __init__(self):
        self.settings = get_settings()
        self.context_manager = ContextManager()
        self.prompt_engine = PromptEngine()
        self.provider_factory = ProviderFactory()
        
        # Initialize agents
        self.agents = {
            AgentRole.ANALYZER: self._create_agent(AgentRole.ANALYZER),
            AgentRole.RESEARCHER: self._create_agent(AgentRole.RESEARCHER),
            AgentRole.SUMMARIZER: self._create_agent(AgentRole.SUMMARIZER),
            AgentRole.VALIDATOR: self._create_agent(AgentRole.VALIDATOR)
        }
        
        # Task queue
        self.task_queue = asyncio.PriorityQueue()
        self.results = {}
    
    def _create_agent(self, role: AgentRole) -> BaseProvider:
        """Create an agent with specific role configuration."""
        provider = self.provider_factory.get_provider(settings.DEFAULT_PROVIDER)
        return provider
    
    async def submit_task(self, task: AgentTask) -> str:
        """Submit a task to the coordinator."""
        task_id = f"{task.role.value}_{len(self.results)}"
        await self.task_queue.put((task.priority, task_id, task))
        return task_id
    
    async def process_tasks(self) -> Dict[str, Any]:
        """Process all tasks in the queue."""
        while not self.task_queue.empty():
            priority, task_id, task = await self.task_queue.get()
            
            try:
                # Check dependencies
                if task.dependencies:
                    for dep_id in task.dependencies:
                        if dep_id not in self.results:
                            # Requeue task if dependencies not met
                            await self.task_queue.put((priority, task_id, task))
                            continue
                
                # Get agent for the task
                agent = self.agents[task.role]
                
                # Prepare context
                context = self.context_manager.retrieve_context(task_id)
                
                # Generate prompt
                prompt = self.prompt_engine.render_prompt(
                    template_name=f"{task.role.value}_task",
                    variables=task.input_data,
                    context=context
                )
                
                # Execute task
                result = await agent.chat_completion(
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Store result
                self.results[task_id] = {
                    "role": task.role.value,
                    "result": result.content,
                    "status": "completed"
                }
                
                # Update context
                self.context_manager.update_context(
                    task_id,
                    {"last_result": result.content}
                )
                
            except Exception as e:
                logger.error(f"Error processing task {task_id}: {str(e)}")
                self.results[task_id] = {
                    "role": task.role.value,
                    "error": str(e),
                    "status": "failed"
                }
        
        return self.results
    
    async def coordinate(
        self,
        tasks: List[AgentTask],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Coordinate multiple tasks and return results."""
        # Submit all tasks
        for task in tasks:
            await self.submit_task(task)
        
        # Process tasks with timeout
        try:
            if timeout:
                results = await asyncio.wait_for(
                    self.process_tasks(),
                    timeout=timeout
                )
            else:
                results = await self.process_tasks()
            
            return results
        except asyncio.TimeoutError:
            logger.warning("Task coordination timed out")
            return {
                "status": "timeout",
                "completed_tasks": self.results
            }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a specific task."""
        return self.results.get(task_id, {"status": "not_found"})
    
    def clear_results(self) -> None:
        """Clear all task results."""
        self.results.clear() 