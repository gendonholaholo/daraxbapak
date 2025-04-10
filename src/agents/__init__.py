from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ..core.logging import logger
from ..core.errors import AGNOError
from ..core.context import context_manager
from ..core.providers import provider_factory
from ..prompts import prompt_manager
from datetime import datetime

class BaseAgent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.context = context_manager
        self.provider = provider_factory.get_provider()
        logger.info(f"Initialized agent: {agent_id}")

    @abstractmethod
    async def process(self, input_data: Dict) -> Dict:
        """Process input data and return response"""
        pass

    async def get_context(self, session_id: str) -> List[Dict]:
        """Get context for a session"""
        return self.context.get_context(session_id)

    async def add_context(self, session_id: str, context: Dict) -> None:
        """Add context to a session"""
        self.context.add_context(session_id, context)

    async def clear_context(self, session_id: str) -> None:
        """Clear context for a session"""
        self.context.clear_context(session_id)

class InterviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__("interviewer")
        self.model_path = "src/models/interviewer_transformer.pth"
        self.vocab_path = "src/models/vocab.json"
        logger.info("Initialized InterviewerAgent")

    async def process(self, input_data: Dict) -> Dict:
        """Process interview input and generate response"""
        try:
            # Validate input
            if not input_data.get("session_id"):
                raise AGNOError("session_id is required")
            if not input_data.get("message"):
                raise AGNOError("message is required")
            
            logger.info(f"Processing interview request: {input_data}")
            
            # Get session context
            session_id = input_data.get("session_id")
            context = await self.get_context(session_id)
            logger.info(f"Retrieved context for session {session_id}: {context}")
            
            # Determine prompt type based on context
            prompt_type = self._determine_prompt_type(input_data, context)
            logger.info(f"Determined prompt type: {prompt_type}")
            
            # Build system message
            system_message = self._build_system_message(prompt_type)
            
            # Build user message
            user_message = self._build_user_message(prompt_type, input_data, context)
            
            logger.info(f"System message: {system_message}")
            logger.info(f"User message: {user_message}")
            
            # Generate response using LLM with system message
            response = await self.provider.generate(user_message, system_message=system_message)
            logger.info(f"Generated response: {response}")
            
            # Update context
            context_update = {
                "input": input_data,
                "response": response,
                "prompt_type": prompt_type,
                "timestamp": str(datetime.now())
            }
            await self.add_context(session_id, context_update)
            logger.info(f"Updated context with: {context_update}")
            
            return {
                "response": response,
                "context": context,
                "prompt_type": prompt_type
            }
            
        except Exception as e:
            logger.error(f"Interviewer agent error: {str(e)}")
            raise AGNOError(f"Interview processing failed: {str(e)}")

    def _determine_prompt_type(self, input_data: Dict, context: List[Dict]) -> str:
        """Determine appropriate prompt type based on context"""
        if not context:
            logger.info("No context found, using greeting prompt type")
            return "greeting"
        
        last_interaction = context[-1]
        if input_data.get("point"):
            logger.info("Point parameter found, using follow_up prompt type")
            return "follow_up"
        
        logger.info("Using question prompt type")
        return "question"

    def _build_system_message(self, prompt_type: str) -> str:
        """Build system message based on prompt type"""
        if prompt_type == "greeting":
            return "You are an AI interviewer conducting a professional interview. Start with a warm greeting and introduction."
        elif prompt_type == "follow_up":
            return "You are an AI interviewer. Ask relevant follow-up questions based on the interviewee's previous response."
        else:
            return "You are an AI interviewer. Ask clear and professional questions related to the topic."

    def _build_user_message(self, prompt_type: str, input_data: Dict, context: List[Dict]) -> str:
        """Build user message using prompt manager"""
        try:
            if prompt_type == "greeting":
                return f"Halo! Saya akan mewawancarai Anda tentang {input_data.get('topic', 'pengalaman Anda')}. {input_data.get('message', '')}"
            
            elif prompt_type == "question":
                context_str = ""
                if context:
                    last_response = context[-1].get("response", "")
                    context_str = f"\nSebelumnya Anda menyebutkan: {last_response}\n"
                
                return f"{context_str}Mengenai {input_data.get('topic', 'pengalaman Anda')}, {input_data.get('message', '')}"
            
            elif prompt_type == "follow_up":
                last_response = context[-1].get("response", "") if context else ""
                point = input_data.get("point", "hal tersebut")
                return f"Berdasarkan jawaban Anda: '{last_response}'\nBisa dijelaskan lebih detail tentang {point}?"
            
            else:
                return input_data.get("message", "")
                
        except Exception as e:
            logger.error(f"Error building user message: {str(e)}")
            raise AGNOError(f"Failed to build user message: {str(e)}")

class AgentFactory:
    _agents: Dict[str, BaseAgent] = {}

    @classmethod
    def get_agent(cls, agent_type: str) -> BaseAgent:
        """Get agent instance"""
        if agent_type not in cls._agents:
            if agent_type == "interviewer":
                cls._agents[agent_type] = InterviewerAgent()
            else:
                raise AGNOError(f"Unsupported agent type: {agent_type}")
        
        return cls._agents[agent_type]

agent_factory = AgentFactory() 