"""
Enhanced Memory Manager with Mem0 + Qdrant Integration
2025 version with performance optimizations and new features
"""

import streamlit as st
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    BinaryQuantization, BinaryQuantizationConfig,
    OptimizersConfigDiff, Filter, FieldCondition,
    MatchValue, Range, SearchParams, HnswConfigDiff
)
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any, Optional, Union
from models import Memory as MemoryModel, MemoryType, MemorySearchResult
import numpy as np
from datetime import datetime, timedelta, timezone
import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class MemoryManager:
    """Enhanced memory manager with 2025 optimizations"""
    
    def __init__(self):
        """Initialize memory manager components"""
        self.init_qdrant()
        self.init_mem0()
        self.init_embedder()
        self._setup_monitoring()
    
    def init_qdrant(self):
        """Initialize Qdrant client with 2025 enhancements"""
        try:
            # Use gRPC for better performance if available
            use_grpc = st.secrets.get("QDRANT_USE_GRPC", False)
            
            if use_grpc:
                self.qdrant = QdrantClient(
                    url=st.secrets["QDRANT_URL"],
                    api_key=st.secrets["QDRANT_API_KEY"],
                    prefer_grpc=True,
                    timeout=30
                )
                logger.info("Initialized Qdrant with gRPC")
            else:
                self.qdrant = QdrantClient(
                    url=st.secrets["QDRANT_URL"],
                    api_key=st.secrets["QDRANT_API_KEY"],
                    timeout=30
                )
                logger.info("Initialized Qdrant with HTTP")
            
            # Create collection if not exists with 2025 optimizations
            self._ensure_collection_exists()
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Ensure collection exists with optimized configuration"""
        collections = self.qdrant.get_collections().collections
        
        # Check if memories collection exists and has correct dimensions
        memories_exists = any(c.name == "memories" for c in collections)
        needs_recreation = False
        
        if memories_exists:
            # Check collection info to see if dimensions match
            try:
                collection_info = self.qdrant.get_collection("memories")
                current_size = collection_info.config.params.vectors.size
                if current_size != 1536:
                    logger.info(f"Collection has wrong dimensions ({current_size}), recreating...")
                    self.qdrant.delete_collection("memories")
                    needs_recreation = True
            except Exception as e:
                logger.warning(f"Could not check collection dimensions: {e}")
                needs_recreation = True
        
        if not memories_exists or needs_recreation:
            # Create optimized collection for 2025
            self.qdrant.create_collection(
                collection_name="memories",
                vectors_config=VectorParams(
                    size=1536,  # OpenAI text-embedding-3-small dimension
                    distance=Distance.COSINE,
                    # Enable binary quantization for 4x memory reduction
                    quantization_config=BinaryQuantization(
                        binary=BinaryQuantizationConfig(
                            always_ram=True  # Keep quantized vectors in RAM
                        )
                    )
                ),
                # HNSW index configuration for better search performance
                hnsw_config=HnswConfigDiff(
                    m=16,  # Number of connections
                    ef_construct=100,  # Search depth during construction
                    full_scan_threshold=10000
                ),
                # Optimizer configuration
                optimizers_config=OptimizersConfigDiff(
                    memmap_threshold=20000,  # Use memmap for large collections
                    indexing_threshold=10000,  # Start indexing after 10k vectors
                    flush_interval_sec=5,  # Flush to disk every 5 seconds
                )
            )
            logger.info("Created optimized memories collection")
    
    def init_mem0(self):
        """Initialize Mem0 with 2025 enhancements"""
        try:
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "url": st.secrets["QDRANT_URL"],
                        "api_key": st.secrets["QDRANT_API_KEY"],
                        "collection_name": "memories"
                        # Removed prefer_grpc as it's not supported
                    }
                },
                "embedder": {
                    "provider": "openai",  # Use OpenAI embeddings instead
                    "config": {
                        "model": "text-embedding-3-small",
                        "api_key": st.secrets.get("OPENAI_API_KEY")
                    }
                },
                "llm": {
                    "provider": "openai",  # For memory extraction
                    "config": {
                        "model": "gpt-3.5-turbo",
                        "api_key": st.secrets.get("OPENAI_API_KEY"),
                        "temperature": 0.1  # Lower temperature for consistency
                    }
                }
            }
            self.mem0 = Memory.from_config(config)
            logger.info("Initialized Mem0 with enhanced pipeline")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            raise
    
    def init_embedder(self):
        """Initialize local embedder with optimization"""
        @st.cache_resource
        def load_embedder():
            model = SentenceTransformer('all-MiniLM-L6-v2')
            # Enable FP16 for memory efficiency if CUDA available
            if hasattr(model, 'half'):
                try:
                    import torch
                    if torch.cuda.is_available():
                        model = model.half()
                        logger.info("Enabled FP16 for embedder")
                except ImportError:
                    pass
            return model
        
        self.embedder = load_embedder()
    
    def _setup_monitoring(self):
        """Setup monitoring for memory operations"""
        try:
            from prometheus_client import Counter, Histogram, Gauge
            
            self.metrics = {
                'memory_operations': Counter(
                    'memory_operations_total', 
                    'Total memory operations',
                    ['operation', 'status']
                ),
                'search_duration': Histogram(
                    'memory_search_duration_seconds',
                    'Memory search duration'
                ),
                'memory_count': Gauge(
                    'total_memories',
                    'Total number of memories'
                ),
                'embedding_time': Histogram(
                    'embedding_generation_seconds',
                    'Time to generate embeddings'
                )
            }
        except ImportError:
            self.metrics = None
            logger.warning("Prometheus client not available, metrics disabled")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def add_memory(
        self, 
        content: str, 
        user_id: str, 
        metadata: Optional[Dict] = None,
        memory_type: MemoryType = MemoryType.CONVERSATION
    ) -> MemoryModel:
        """Add a new memory with 2025 two-phase pipeline"""
        try:
            # Track operation
            if self.metrics:
                self.metrics['memory_operations'].labels(
                    operation='add', status='started'
                ).inc()
            
            # Phase 1: Extraction - Use Mem0's intelligent extraction
            extraction_result = await self._extract_memory_components(
                content, user_id, metadata
            )
            
            # Phase 2: Update - Compare with existing memories and merge
            memory_data = await self._update_or_create_memory(
                extraction_result, user_id, memory_type
            )
            
            # Create memory model
            memory = MemoryModel(
                id=memory_data.get('id', str(uuid.uuid4())),
                user_id=user_id,
                content=memory_data.get('content', content),
                metadata=memory_data.get('metadata', metadata or {}),
                memory_type=memory_type,
                entities=memory_data.get('entities', []),
                relationships=memory_data.get('relationships', []),
                importance=memory_data.get('importance', 0.5),
                source='user_input',
                created_at=datetime.now(timezone.utc)
            )
            
            # Add to Mem0
            mem0_result = self.mem0.add(
                memory.content,
                user_id=user_id,
                metadata={
                    **memory.metadata,
                    'memory_type': memory_type.value,
                    'entities': memory.entities,
                    'importance': memory.importance
                }
            )
            
            if self.metrics:
                self.metrics['memory_operations'].labels(
                    operation='add', status='success'
                ).inc()
                self.metrics['memory_count'].inc()
            
            logger.info(f"Added memory {memory.id} for user {user_id}")
            return memory
            
        except Exception as e:
            if self.metrics:
                self.metrics['memory_operations'].labels(
                    operation='add', status='error'
                ).inc()
            logger.error(f"Failed to add memory: {e}")
            raise
    
    async def _extract_memory_components(
        self, 
        content: str, 
        user_id: str, 
        metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Extract memory components using Mem0's extraction pipeline"""
        # This would use Mem0's extraction capabilities
        # For now, returning a simplified version
        return {
            'content': content,
            'entities': self._extract_entities(content),
            'relationships': [],
            'importance': self._calculate_importance(content, metadata),
            'metadata': metadata or {}
        }
    
    def _extract_entities(self, content: str) -> List[str]:
        """Simple entity extraction (would use NER in production)"""
        # This is a placeholder - in production, use spaCy or similar
        import re
        # Extract capitalized words as potential entities
        entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', content)
        return list(set(entities))
    
    def _calculate_importance(self, content: str, metadata: Optional[Dict]) -> float:
        """Calculate memory importance score"""
        importance = 0.5
        
        # Increase importance for certain keywords
        important_keywords = ['important', 'remember', 'critical', 'key', 'essential']
        if any(keyword in content.lower() for keyword in important_keywords):
            importance += 0.2
        
        # Check metadata
        if metadata:
            if metadata.get('importance') == 'high':
                importance += 0.3
            elif metadata.get('importance') == 'low':
                importance -= 0.2
        
        return min(max(importance, 0.0), 1.0)
    
    async def _update_or_create_memory(
        self, 
        extraction_result: Dict[str, Any],
        user_id: str,
        memory_type: MemoryType
    ) -> Dict[str, Any]:
        """Update existing memory or create new one"""
        # Check for similar memories
        try:
            similar = await self.search_memories(
                extraction_result['content'], 
                user_id, 
                limit=3,
                threshold=0.9
            )
        except Exception as e:
            logger.error(f"Error searching for similar memories during update: {e}")
            # If search fails, just create a new memory
            similar = []
        
        if similar and similar[0].score > 0.95:
            # Update existing memory
            existing = similar[0].memory
            logger.info(f"Updating existing memory {existing.id}")
            
            # Merge metadata
            merged_metadata = {**existing.metadata, **extraction_result['metadata']}
            
            # Update importance
            new_importance = (existing.importance + extraction_result['importance']) / 2
            
            return {
                'id': existing.id,
                'content': extraction_result['content'],
                'metadata': merged_metadata,
                'entities': list(set(existing.entities + extraction_result['entities'])),
                'relationships': existing.relationships + extraction_result['relationships'],
                'importance': new_importance
            }
        else:
            # Create new memory
            return {
                'id': str(uuid.uuid4()),
                **extraction_result
            }
    
    async def search_memories(
        self, 
        query: str, 
        user_id: str, 
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        threshold: float = 0.0,
        date_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[MemorySearchResult]:
        """Enhanced memory search with filters"""
        start_time = datetime.now()
        try:
            # Build filters dict for Mem0 (not Qdrant Filter objects)
            filters = {}
            
            if memory_types:
                filters["memory_type"] = [mt.value for mt in memory_types]
            
            if date_range:
                filters["created_at"] = {
                    "gte": date_range[0].isoformat(),
                    "lte": date_range[1].isoformat()
                }
            
            # Use Mem0 search - returns dict with 'results' key
            search_response = self.mem0.search(
                query=query,
                user_id=user_id,
                limit=limit * 2,  # Get more results for post-filtering
                filters=filters if filters else None
            )
            
            
            # Handle response format - extract results from dict
            try:
                if isinstance(search_response, dict):
                    results = search_response.get('results', [])
                elif isinstance(search_response, list):
                    results = search_response
                elif isinstance(search_response, str):
                    # Sometimes Mem0 returns just a string
                    results = [search_response] if search_response.strip() else []
                else:
                    logger.warning(f"Unexpected search response format: {type(search_response)}")
                    results = []
            except Exception as e:
                logger.error(f"Error extracting results from search response: {e}")
                logger.error(f"Response type: {type(search_response)}, Content: {search_response}")
                results = []
            
            # Convert to MemorySearchResult objects
            search_results = []
            for i, result in enumerate(results):
                try:
                    
                    # Handle different result formats
                    if isinstance(result, dict):
                        try:
                            # Standard dict format
                            memory_content = result.get('memory', result.get('text', result.get('content', '')))
                            score = float(result.get('score', 0.0))
                            
                            if score >= threshold and memory_content:
                                # Create memory model from result
                                memory = MemoryModel(
                                    id=result.get('id', str(uuid.uuid4())),
                                    user_id=user_id,
                                    content=memory_content,
                                    metadata=result.get('metadata', {}),
                                    memory_type=MemoryType.CONVERSATION,
                                    entities=[],
                                    importance=score,
                                    created_at=datetime.now(timezone.utc)
                                )
                                
                                search_result = MemorySearchResult(
                                    memory=memory,
                                    score=score,
                                    highlights=self._extract_highlights(query, memory_content)
                                )
                                search_results.append(search_result)
                        except Exception as dict_error:
                            logger.error(f"Error processing dict result {i}: {dict_error}")
                            continue
                    
                    elif isinstance(result, str) and result.strip():
                        try:
                            # Handle string results
                            memory = MemoryModel(
                                id=str(uuid.uuid4()),
                                user_id=user_id,
                                content=result.strip(),
                                metadata={},
                                memory_type=MemoryType.CONVERSATION,
                                entities=[],
                                importance=0.5,
                                created_at=datetime.now(timezone.utc)
                            )
                            
                            search_result = MemorySearchResult(
                                memory=memory,
                                score=0.5,
                                highlights=self._extract_highlights(query, result.strip())
                            )
                            search_results.append(search_result)
                        except Exception as str_error:
                            logger.error(f"Error processing string result {i}: {str_error}")
                            continue
                    
                    else:
                        logger.warning(f"Skipping unexpected result format: {type(result)} - {str(result)[:50]}...")
                
                except Exception as e:
                    logger.error(f"Error processing search result {i}: {e}")
                    continue
            
            # Sort by score and limit
            search_results.sort(key=lambda x: x.score, reverse=True)
            search_results = search_results[:limit]
            
            # Record metrics
            if self.metrics:
                duration = (datetime.now() - start_time).total_seconds()
                self.metrics['search_duration'].observe(duration)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _result_to_memory(self, result: Union[Dict, str], user_id: str) -> MemoryModel:
        """Convert search result to Memory model"""
        try:
            
            # Handle string results (Mem0 sometimes returns just the text)
            if isinstance(result, str):
                return MemoryModel(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    content=result,
                    metadata={},
                    memory_type=MemoryType.CONVERSATION,
                    entities=[],
                    importance=0.5,
                    created_at=datetime.now(timezone.utc)
                )
            
            # Handle dict results only
            if not isinstance(result, dict):
                logger.error(f"Unexpected result type in _result_to_memory: {type(result)}")
                # Return a default memory for unexpected types
                return MemoryModel(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    content=str(result),
                    metadata={},
                    memory_type=MemoryType.CONVERSATION,
                    entities=[],
                    importance=0.5,
                    created_at=datetime.now(timezone.utc)
                )
            
            # Now we know result is a dict
            metadata = result.get('metadata', {})
            
            return MemoryModel(
                id=result.get('id', str(uuid.uuid4())),
                user_id=user_id,
                content=result.get('text', result.get('content', result.get('memory', ''))),
                metadata=metadata,
                memory_type=MemoryType(metadata.get('memory_type', 'conversation')),
                entities=metadata.get('entities', []),
                importance=metadata.get('importance', 0.5),
                created_at=datetime.fromisoformat(
                    metadata.get('created_at', datetime.now(timezone.utc).isoformat())
                ) if 'created_at' in metadata else datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Error in _result_to_memory: {e}")
            # Return a safe default
            return MemoryModel(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=str(result),
                metadata={},
                memory_type=MemoryType.CONVERSATION,
                entities=[],
                importance=0.5,
                created_at=datetime.now(timezone.utc)
            )
    
    def _extract_highlights(self, query: str, content: str, context_size: int = 50) -> List[str]:
        """Extract highlighted snippets from content"""
        highlights = []
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find query terms in content
        query_terms = query_lower.split()
        for term in query_terms:
            if len(term) > 2:  # Skip short words
                index = content_lower.find(term)
                if index != -1:
                    # Extract context around the term
                    start = max(0, index - context_size)
                    end = min(len(content), index + len(term) + context_size)
                    highlight = content[start:end]
                    
                    # Add ellipsis if truncated
                    if start > 0:
                        highlight = "..." + highlight
                    if end < len(content):
                        highlight = highlight + "..."
                    
                    highlights.append(highlight)
        
        return highlights[:3]  # Return top 3 highlights
    
    async def get_all_memories(
        self, 
        user_id: str,
        offset: int = 0,
        limit: int = 100
    ) -> List[MemoryModel]:
        """Get all memories for a user with pagination"""
        try:
            # Use Mem0's get_all (doesn't support offset, so get all and slice)
            response = self.mem0.get_all(
                user_id=user_id,
                limit=offset + limit  # Get enough to slice
            )
            
            # Handle both dict response (v1.1+) and list response (v1.0)
            if isinstance(response, dict):
                all_memories = response.get('results', [])
            else:
                all_memories = response if isinstance(response, list) else []
            
            # Convert to Memory models
            memories = []
            for mem_data in all_memories:
                # Skip if mem_data is not a dict
                if not isinstance(mem_data, dict):
                    logger.warning(f"Skipping non-dict memory data: {type(mem_data)}")
                    continue
                memory = self._result_to_memory(mem_data, user_id)
                memories.append(memory)
            
            # Apply offset and limit manually
            return memories[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []
    
    async def update_memory(
        self, 
        memory_id: str, 
        content: str, 
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> MemoryModel:
        """Update an existing memory"""
        try:
            # Update via Mem0
            result = self.mem0.update(
                memory_id,
                content,
                user_id=user_id,
                metadata=metadata
            )
            
            # Get the updated memory from Mem0
            updated_data = self.mem0.get(memory_id)
            
            # Convert to memory model
            if updated_data:
                updated_memory = self._result_to_memory(updated_data, user_id)
                updated_memory.last_modified = datetime.now(timezone.utc)
            else:
                # If get fails, create a basic memory from the update request
                updated_memory = MemoryModel(
                    id=memory_id,
                    user_id=user_id,
                    content=content,
                    metadata=metadata or {},
                    last_modified=datetime.now(timezone.utc)
                )
            
            logger.info(f"Updated memory {memory_id}")
            return updated_memory
            
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory"""
        try:
            # Mem0's delete method only takes memory_id
            self.mem0.delete(memory_id)
            
            if self.metrics:
                self.metrics['memory_count'].dec()
            
            logger.info(f"Deleted memory {memory_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            raise
    
    async def delete_all_memories(self, user_id: str) -> int:
        """Delete all memories for a user"""
        try:
            memories = await self.get_all_memories(user_id)
            deleted_count = 0
            
            for memory in memories:
                await self.delete_memory(memory.id, user_id)
                deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} memories for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}")
            raise
    
    async def consolidate_memories(self, user_id: str) -> Dict[str, int]:
        """Consolidate and deduplicate memories"""
        try:
            logger.info(f"Starting memory consolidation for user {user_id}")
            
            # Get all memories
            memories = await self.get_all_memories(user_id)
            if not memories:
                return {"total": 0, "removed": 0, "merged": 0}
            
            # Extract contents and generate embeddings
            contents = [m.content for m in memories]
            
            embedding_start = datetime.now()
            embeddings = self.embedder.encode(contents, batch_size=32, show_progress_bar=False)
            
            if self.metrics:
                embedding_duration = (datetime.now() - embedding_start).total_seconds()
                self.metrics['embedding_time'].observe(embedding_duration)
            
            # Calculate similarity matrix
            similarity_matrix = np.inner(embeddings, embeddings)
            
            # Find duplicates and near-duplicates
            duplicates = []
            merged = []
            
            for i in range(len(memories)):
                for j in range(i + 1, len(memories)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity > 0.95:
                        # Near duplicate - mark for removal
                        duplicates.append((i, j, similarity))
                    elif similarity > 0.85:
                        # Similar - consider merging
                        merged.append((i, j, similarity))
            
            # Remove duplicates (keep the older one)
            removed_count = 0
            for i, j, _ in duplicates:
                older_idx = i if memories[i].created_at < memories[j].created_at else j
                newer_idx = j if older_idx == i else i
                
                await self.delete_memory(memories[newer_idx].id, user_id)
                removed_count += 1
            
            # Merge similar memories
            merged_count = 0
            for i, j, _ in merged:
                if merged_count < 10:  # Limit merges per consolidation
                    # Merge metadata and update importance
                    merged_metadata = {
                        **memories[i].metadata,
                        **memories[j].metadata,
                        'merged_from': [memories[i].id, memories[j].id]
                    }
                    
                    new_importance = (memories[i].importance + memories[j].importance) / 2
                    
                    # Update the older memory with merged content
                    older_idx = i if memories[i].created_at < memories[j].created_at else j
                    await self.update_memory(
                        memories[older_idx].id,
                        f"{memories[i].content}\n\n{memories[j].content}",
                        user_id,
                        {**merged_metadata, 'importance': new_importance}
                    )
                    
                    merged_count += 1
            
            results = {
                "total": len(memories),
                "removed": removed_count,
                "merged": merged_count
            }
            
            logger.info(f"Consolidation complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return {"total": 0, "removed": 0, "merged": 0, "error": str(e)}
    
    async def export_memories(
        self, 
        user_id: str,
        format: str = "json",
        include_embeddings: bool = False
    ) -> Union[Dict, str]:
        """Export user memories for GDPR compliance"""
        try:
            memories = await self.get_all_memories(user_id)
            
            export_data = {
                "user_id": user_id,
                "export_date": datetime.now(timezone.utc).isoformat(),
                "memory_count": len(memories),
                "memories": []
            }
            
            for memory in memories:
                mem_dict = memory.model_dump()
                
                # Remove embeddings unless specifically requested
                if not include_embeddings and 'embedding' in mem_dict:
                    del mem_dict['embedding']
                
                # Convert datetime objects to strings
                for key, value in mem_dict.items():
                    if isinstance(value, datetime):
                        mem_dict[key] = value.isoformat()
                
                export_data["memories"].append(mem_dict)
            
            if format == "json":
                return export_data
            else:
                # Convert to other formats as needed
                raise NotImplementedError(f"Export format {format} not implemented")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        try:
            memories = self.mem0.get_all(user_id=user_id)
            
            if not memories:
                return {
                    "total_memories": 0,
                    "memory_types": {},
                    "avg_importance": 0,
                    "total_entities": 0,
                    "date_range": None
                }
            
            # Calculate statistics
            memory_types = {}
            importances = []
            all_entities = set()
            dates = []
            
            for mem in memories:
                # Handle both dict and object formats
                if isinstance(mem, str):
                    continue  # Skip string entries
                    
                metadata = mem.get('metadata', {}) if isinstance(mem, dict) else {}
                
                # Count memory types
                mem_type = metadata.get('memory_type', 'conversation')
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                
                # Collect importance scores
                importances.append(metadata.get('importance', 0.5))
                
                # Collect entities
                entities = metadata.get('entities', [])
                all_entities.update(entities)
                
                # Collect dates
                if 'created_at' in metadata:
                    try:
                        date = datetime.fromisoformat(metadata['created_at'])
                        dates.append(date)
                    except:
                        pass
            
            # Calculate date range
            date_range = None
            if dates:
                date_range = {
                    "earliest": min(dates).isoformat(),
                    "latest": max(dates).isoformat()
                }
            
            return {
                "total_memories": len(memories),
                "memory_types": memory_types,
                "avg_importance": sum(importances) / len(importances) if importances else 0,
                "total_entities": len(all_entities),
                "unique_entities": list(all_entities)[:20],  # Top 20 entities
                "date_range": date_range
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "total_memories": 0,
                "error": str(e)
            }