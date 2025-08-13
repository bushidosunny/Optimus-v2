"""
Enhanced PydanticAI Multi-LLM Handler (2025)
Supports multiple LLM providers with type-safe agents
"""

import streamlit as st
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import Optional, List, Dict, Any, Union
from models import LLMProvider, ChatMessage, MemorySearchResult
import httpx
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import prompts

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Type-safe dependencies for PydanticAI agents"""
    memory_manager: Any
    user_id: str
    session_data: Dict[str, Any]


class CustomLLMModel(Model):
    """Base class for custom LLM implementations"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )
    
    async def complete(self, messages: List[Dict], **kwargs) -> str:
        """Complete a chat conversation"""
        raise NotImplementedError
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


class GroqModel(CustomLLMModel):
    """Groq LLM implementation"""
    
    def __init__(self, api_key: str, model_name: str = "mixtral-8x7b-32768"):
        super().__init__(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model_name=model_name
        )
    
    async def complete(self, messages: List[Dict], **kwargs) -> str:
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class MistralModel(CustomLLMModel):
    """Mistral AI implementation"""
    
    def __init__(self, api_key: str, model_name: str = "mistral-large-latest"):
        super().__init__(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            model_name=model_name
        )
    
    async def complete(self, messages: List[Dict], **kwargs) -> str:
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class XAIModel(CustomLLMModel):
    """X.AI (Grok) implementation - 2025 version"""
    
    def __init__(self, api_key: str, model_name: str = "grok-4-0709"):
        super().__init__(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            model_name=model_name
        )
    
    async def complete(self, messages: List[Dict], **kwargs) -> str:
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class LLMHandler:
    """Enhanced LLM handler with PydanticAI integration"""
    
    def __init__(self):
        self.current_model_id = None
        self.agent = None
        self.model_registry = self._build_model_registry()
        self._setup_monitoring()
        logger.info("Initialized LLM handler")
    
    def _build_model_registry(self) -> Dict[str, Model]:
        """Build registry of available models with 2025 providers"""
        registry = {}
        
        # Set API keys in environment variables for PydanticAI
        import os
        
        # OpenAI models (2025 lineup with GPT-5)
        if st.secrets.get("OPENAI_API_KEY"):
            try:
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                # GPT-5 models (released August 2025)
                registry["openai:gpt-5"] = OpenAIModel("gpt-5")
                registry["openai:gpt-5-mini"] = OpenAIModel("gpt-5-mini")
                registry["openai:gpt-5-nano"] = OpenAIModel("gpt-5-nano")

                logger.info("Registered OpenAI models")
            except Exception as e:
                logger.error(f"Failed to register OpenAI models: {e}")
        
        # Anthropic models (2025 lineup with Claude 4)
        if st.secrets.get("ANTHROPIC_API_KEY"):
            try:
                os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
                # Claude 4 models (2025)
                registry["anthropic:claude-4-opus"] = AnthropicModel("claude-4-opus-20250514")
                registry["anthropic:claude-4-sonnet"] = AnthropicModel("claude-4-sonnet-20250514")
                logger.info("Registered Anthropic models")
            except Exception as e:
                logger.error(f"Failed to register Anthropic models: {e}")
        
        # Google models
        if st.secrets.get("GOOGLE_API_KEY"):
            try:
                # Would use GeminiModel when available in PydanticAI
                # For now, using a custom implementation
                pass
            except Exception as e:
                logger.error(f"Failed to register Google models: {e}")
        
        # Groq models - TODO: Implement custom model class
        # if st.secrets.get("GROQ_API_KEY"):
        #     try:
        #         registry["groq:mixtral-8x7b"] = GroqModel(
        #             st.secrets["GROQ_API_KEY"],
        #             "mixtral-8x7b-32768"
        #         )
        #         registry["groq:llama-3.1-70b"] = GroqModel(
        #             st.secrets["GROQ_API_KEY"],
        #             "llama-3.1-70b-versatile"
        #         )
        #         logger.info("Registered Groq models")
        #     except Exception as e:
        #         logger.error(f"Failed to register Groq models: {e}")
        
        # Mistral models
        if st.secrets.get("MISTRAL_API_KEY"):
            try:
                registry["mistral:mistral-large"] = MistralModel(
                    st.secrets["MISTRAL_API_KEY"],
                    "mistral-large-latest"
                )
                registry["mistral:mistral-medium"] = MistralModel(
                    st.secrets["MISTRAL_API_KEY"],
                    "mistral-medium-latest"
                )
                logger.info("Registered Mistral models")
            except Exception as e:
                logger.error(f"Failed to register Mistral models: {e}")
        
        # X.AI models (2025 lineup with Grok-4) - Using OpenAI-compatible API
        if st.secrets.get("XAI_API_KEY"):
            try:
                os.environ["XAI_API_KEY"] = st.secrets["XAI_API_KEY"]
                # X.AI uses OpenAI-compatible API, so we use OpenAIModel with custom provider
                xai_provider = OpenAIProvider(
                    base_url="https://api.x.ai/v1",
                    api_key=st.secrets["XAI_API_KEY"]
                )
                registry["xai:grok-4"] = OpenAIModel("grok-4-0709", provider=xai_provider)
                registry["xai:grok-beta"] = OpenAIModel("grok-beta", provider=xai_provider)
                logger.info("Registered X.AI models")
            except Exception as e:
                logger.error(f"Failed to register X.AI models: {e}")
        
        return registry
    
    def _setup_monitoring(self):
        """Setup monitoring for LLM operations"""
        try:
            from prometheus_client import Counter, Histogram, Gauge
            
            self.metrics = {
                'llm_requests': Counter(
                    'llm_requests_total',
                    'Total LLM requests',
                    ['model', 'status']
                ),
                'llm_tokens': Counter(
                    'llm_tokens_total',
                    'Total tokens used',
                    ['model', 'type']
                ),
                'llm_latency': Histogram(
                    'llm_request_duration_seconds',
                    'LLM request duration',
                    ['model']
                )
            }
        except ImportError:
            self.metrics = None
            logger.warning("Prometheus client not available")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model identifiers"""
        return list(self.model_registry.keys())
    
    def create_agent(self, model_id: str) -> Agent[AgentDependencies, str]:
        """Create a type-safe agent with selected model"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not available")
        
        model = self.model_registry[model_id]
        
        # Create agent with type-safe dependencies
        agent = Agent(
            model=model,
            deps_type=AgentDependencies,
            output_type=str,
            system_prompt=prompts.system_instruction
        )
        
        # Register tools with type safety
        @agent.tool
        async def search_memories(
            ctx: RunContext[AgentDependencies], 
            query: str, 
            limit: int = 5
        ) -> List[Dict[str, Any]]:
            """Search through user's memories"""
            results = await ctx.deps.memory_manager.search_memories(
                query, 
                ctx.deps.user_id, 
                limit=limit
            )
            
            return [
                {
                    "content": r.memory.content,
                    "score": r.score,
                    "created": r.memory.created_at.isoformat(),
                    "type": r.memory.memory_type.value,
                    "highlights": r.highlights
                }
                for r in results
            ]
        
        @agent.tool
        async def add_note(
            ctx: RunContext[AgentDependencies], 
            note: str,
            importance: float = 0.5,
            category: str = "general"
        ) -> str:
            """Add a note to memory for future reference"""
            memory = await ctx.deps.memory_manager.add_memory(
                note,
                ctx.deps.user_id,
                {
                    "category": category, 
                    "type": "note",
                    "importance": importance
                }
            )
            return f"Note added with ID: {memory.id}"
        
        @agent.tool
        async def get_memory_stats(
            ctx: RunContext[AgentDependencies]
        ) -> Dict[str, Any]:
            """Get statistics about user's memories"""
            stats = ctx.deps.memory_manager.get_statistics(ctx.deps.user_id)
            return stats
        
        self.agent = agent
        logger.info(f"Created agent with model {model_id}")
        return agent
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def chat(
        self, 
        message: str,
        model_id: str,
        memory_manager: Any,
        user_id: str,
        context: Optional[List[MemorySearchResult]] = None,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """Send chat message with memory context using type-safe agent"""
        
        start_time = datetime.now()
        
        try:
            # Track request
            if self.metrics:
                self.metrics['llm_requests'].labels(
                    model=model_id, status='started'
                ).inc()
            
            # Create or get agent for model
            if not self.agent or self.current_model_id != model_id:
                self.create_agent(model_id)
                self.current_model_id = model_id
            
            # Prepare dependencies
            deps = AgentDependencies(
                memory_manager=memory_manager,
                user_id=user_id,
                session_data={
                    "context": context or [],
                    "history": conversation_history or [],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Build enhanced prompt with context
            prompt_parts = []
            
            # Add memory context if available
            if context:
                memories_text = "\n".join([
                    f"[Relevance: {r.score:.2f}] {r.memory.content}"
                    for r in context[:3]  # Top 3 most relevant
                ])
                prompt_parts.append(f"Relevant memories from our past conversations:\n{memories_text}")
            
            # Add recent conversation history
            if conversation_history:
                history_text = "\n".join([
                    f"{msg.role}: {msg.content}"
                    for msg in conversation_history[-5:]  # Last 5 messages
                ])
                prompt_parts.append(f"Recent conversation:\n{history_text}")
            
            # Add current message
            prompt_parts.append(f"User: {message}")
            full_prompt = "\n\n".join(prompt_parts)
            
            # Run agent with dependencies (PydanticAI doesn't accept temperature/max_tokens)
            result = await self.agent.run(
                full_prompt, 
                deps=deps
            )
            
            # Calculate metrics
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate tokens (rough approximation)
            prompt_tokens = len(full_prompt.split()) * 1.3
            completion_tokens = len(result.output.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Track metrics
            if self.metrics:
                self.metrics['llm_requests'].labels(
                    model=model_id, status='success'
                ).inc()
                self.metrics['llm_tokens'].labels(
                    model=model_id, type='prompt'
                ).inc(prompt_tokens)
                self.metrics['llm_tokens'].labels(
                    model=model_id, type='completion'
                ).inc(completion_tokens)
                self.metrics['llm_latency'].labels(
                    model=model_id
                ).observe(generation_time)
            
            metadata = {
                "model": model_id,
                "tokens": {
                    "prompt": int(prompt_tokens),
                    "completion": int(completion_tokens),
                    "total": total_tokens
                },
                "generation_time": generation_time
            }
            
            logger.info(f"Generated response with {model_id}: {total_tokens} tokens in {generation_time:.2f}s")
            
            return result.output, metadata
            
        except Exception as e:
            if self.metrics:
                self.metrics['llm_requests'].labels(
                    model=model_id, status='error'
                ).inc()
            logger.error(f"Chat generation failed: {e}")
            raise
    
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_id not in self.model_registry:
            return {"error": "Model not found"}
        
        provider, model_name = model_id.split(":", 1)
        
        info = {
            "id": model_id,
            "provider": provider,
            "model": model_name,
            "features": []
        }
        
        # Add model-specific features
        if "gpt-4" in model_name:
            info["features"].extend(["vision", "function_calling", "128k_context"])
        elif "claude-3" in model_name:
            info["features"].extend(["200k_context", "vision", "code_analysis"])
        elif "mixtral" in model_name:
            info["features"].extend(["32k_context", "fast_inference", "multilingual"])
        elif "gemini" in model_name:
            info["features"].extend(["1M_context", "multimodal", "free_tier"])
        
        return info
    
    async def stream_chat(
        self,
        message: str,
        model_id: str,
        memory_manager: Any,
        user_id: str,
        context: Optional[List[MemorySearchResult]] = None,
        conversation_history: Optional[List[ChatMessage]] = None
    ):
        """Stream chat responses using PydanticAI's run_stream"""
        start_time = datetime.now()
        
        try:
            # Track request
            if self.metrics:
                self.metrics['llm_requests'].labels(
                    model=model_id, status='started'
                ).inc()
            
            # Create or get agent for model
            if not self.agent or self.current_model_id != model_id:
                self.create_agent(model_id)
                self.current_model_id = model_id
            
            # Prepare dependencies
            deps = AgentDependencies(
                memory_manager=memory_manager,
                user_id=user_id,
                session_data={
                    "context": context or [],
                    "history": conversation_history or [],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Build enhanced prompt with context
            prompt_parts = []
            
            # Add memory context if available
            if context:
                memories_text = "\n".join([
                    f"[Relevance: {r.score:.2f}] {r.memory.content}"
                    for r in context[:3]  # Top 3 most relevant
                ])
                prompt_parts.append(f"Relevant memories from our past conversations:\n{memories_text}")
            
            # Add recent conversation history
            if conversation_history:
                history_text = "\n".join([
                    f"{msg.role}: {msg.content}"
                    for msg in conversation_history[-5:]  # Last 5 messages
                ])
                prompt_parts.append(f"Recent conversation:\n{history_text}")
            
            # Add current message
            prompt_parts.append(f"User: {message}")
            full_prompt = "\n\n".join(prompt_parts)
            
            # Stream response using PydanticAI's run_stream
            full_response = ""
            async with self.agent.run_stream(full_prompt, deps=deps) as result:
                async for text in result.stream_text(delta=True):
                    full_response += text
                    yield text
            
            # Calculate metrics after streaming completes
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate tokens (rough approximation)
            prompt_tokens = len(full_prompt.split()) * 1.3
            completion_tokens = len(full_response.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Track metrics
            if self.metrics:
                self.metrics['llm_requests'].labels(
                    model=model_id, status='success'
                ).inc()
                self.metrics['llm_tokens'].labels(
                    model=model_id, type='prompt'
                ).inc(prompt_tokens)
                self.metrics['llm_tokens'].labels(
                    model=model_id, type='completion'
                ).inc(completion_tokens)
                self.metrics['llm_latency'].labels(
                    model=model_id
                ).observe(generation_time)
            
            logger.info(f"Streamed response with {model_id}: {total_tokens} tokens in {generation_time:.2f}s")
            
        except Exception as e:
            if self.metrics:
                self.metrics['llm_requests'].labels(
                    model=model_id, status='error'
                ).inc()
            logger.error(f"Stream chat generation failed: {e}")
            raise
    
    def get_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """Get usage summary for a user"""
        # This would typically query from a database
        # For now, returning mock data
        return {
            "user_id": user_id,
            "period": "current_month",
            "usage": {
                "total_requests": 150,
                "total_tokens": 45000,
                "by_model": {
                    "openai:gpt-3.5-turbo": {
                        "requests": 100,
                        "tokens": 30000
                    },
                    "anthropic:claude-3-haiku": {
                        "requests": 50,
                        "tokens": 15000
                    }
                }
            },
            "limits": {
                "daily_tokens": 100000,
                "monthly_tokens": 1000000
            }
        }
    
    async def close(self):
        """Cleanup resources"""
        # Close any custom model clients
        for model_id, model in self.model_registry.items():
            if isinstance(model, CustomLLMModel):
                await model.client.aclose()
        logger.info("LLM handler closed")