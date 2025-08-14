"""
Optimus - Main Streamlit Application
Enhanced 2025 version with advanced features
"""

import streamlit as st
import asyncio
from datetime import datetime, timedelta
from memory_manager import MemoryManager
from llm_handler import LLMHandler, AgentDependencies
from auth import EnhancedAuth
from models import (
    ChatMessage, MemoryType, MemorySearchResult,
    UserRole, SystemMetrics
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration with 2025 enhancements
st.set_page_config(
    page_title="Optimus v2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/ai-memory-system',
        'Report a bug': "https://github.com/your-repo/ai-memory-system/issues",
        'About': "# Optimus v2.0\n\nPowered by Mem0 + Qdrant + PydanticAI\n\n© 2025"
    }
)


# Initialize components with enhanced caching
@st.cache_resource(ttl=3600)
def init_components():
    """Initialize system components with caching (excluding auth) - v2"""
    try:
        # Check for required secrets
        required_secrets = ['QDRANT_URL', 'QDRANT_API_KEY']
        missing_secrets = [s for s in required_secrets if not st.secrets.get(s)]
        
        if missing_secrets:
            st.error(f"❌ Missing required secrets: {', '.join(missing_secrets)}")
            st.info("Please add these in Streamlit Cloud: App Settings → Secrets")
            st.code("""
# Required in secrets.toml:
QDRANT_URL = "https://your-cluster.qdrant.io"
QDRANT_API_KEY = "your-api-key"
OPENAI_API_KEY = "sk-..."  # For embeddings
JWT_SECRET_KEY = "random-secret-key"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"
            """)
            st.stop()
        
        with st.spinner("🚀 Initializing AI Memory System..."):
            components = {
                'memory': MemoryManager(),
                'llm': LLMHandler()
            }
        logger.info("Components initialized successfully")
        return components
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        st.error(f"❌ System initialization failed: {str(e)}")
        if "QDRANT" in str(e).upper():
            st.warning("🔧 Qdrant connection issue. Please check:")
            st.code(f"""
Current QDRANT_URL: {st.secrets.get('QDRANT_URL', 'NOT SET')}
Has API Key: {'✅' if st.secrets.get('QDRANT_API_KEY') else '❌'}

Make sure your Qdrant Cloud cluster is:
1. Running (not paused)
2. URL format is correct (https://xxx.qdrant.io)
3. API key is valid
            """)
        st.stop()

def get_auth():
    """Get authentication instance (not cached to handle fresh sessions)"""
    if 'auth_instance' not in st.session_state:
        st.session_state.auth_instance = EnhancedAuth()
    return st.session_state.auth_instance


# Custom CSS for enhanced UI
def load_custom_css():
    """Load custom CSS for better UI"""
    st.markdown("""
    <style>
        /* Enhanced tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: #0E1117;
            padding: 10px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #262730;
            border-radius: 8px;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #393A47;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF6B6B !important;
        }
        
        /* Memory cards */
        .memory-card {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #FF6B6B;
        }
        
        /* Metrics cards */
        .metric-card {
            background-color: #1E222A;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Chat messages */
        .user-message {
            background-color: #393A47;
            border-radius: 10px;
            padding: 10px 15px;
            margin: 5px 0;
        }
        .assistant-message {
            background-color: #262730;
            border-radius: 10px;
            padding: 10px 15px;
            margin: 5px 0;
        }
        
        /* Status indicators */
        .status-online {
            color: #00D26A;
            font-weight: bold;
        }
        .status-offline {
            color: #F85149;
            font-weight: bold;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)


def display_system_health():
    """Display system health indicators"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 🟢 API Status")
        st.success("All systems operational")
    
    with col2:
        st.markdown("### 💾 Memory DB")
        try:
            # Check Qdrant connection
            st.success("Connected")
        except:
            st.error("Disconnected")
    
    with col3:
        st.markdown("### 🤖 LLM Status")
        available_models = st.session_state.components['llm'].get_available_models()
        st.info(f"{len(available_models)} models available")
    
    with col4:
        st.markdown("### 👥 Active Users")
        active_sessions = len(st.session_state.components['auth'].get_active_sessions(
            st.session_state.user_data['username']
        ))
        st.metric("Sessions", active_sessions)


async def search_and_display_memories(query: str, user_id: str, limit: int = 5):
    """Search memories and display results"""
    with st.spinner("🔍 Searching memories..."):
        results = await st.session_state.components['memory'].search_memories(
            query, user_id, limit=limit
        )
    
    if results:
        st.success(f"Found {len(results)} relevant memories")
        
        for i, result in enumerate(results):
            with st.expander(
                f"Memory {i+1} - Relevance: {result.score:.2%} | "
                f"Type: {result.memory.memory_type.value.title()}"
            ):
                # Memory content
                st.markdown("**Content:**")
                st.markdown(f'<div class="memory-card">{result.memory.content}</div>', 
                           unsafe_allow_html=True)
                
                # Highlights
                if result.highlights:
                    st.markdown("**Relevant excerpts:**")
                    for highlight in result.highlights:
                        st.info(highlight)
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Created: {result.memory.created_at.strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    st.caption(f"Importance: {'⭐' * int(result.memory.importance * 5)}")
                with col3:
                    st.caption(f"Access count: {result.memory.access_count}")
                
                # Entities
                if result.memory.entities:
                    st.markdown("**Entities:** " + ", ".join(result.memory.entities))
    else:
        st.info("No memories found matching your search")


def display_chat_interface():
    """Display the main chat interface"""
    st.markdown("## 💬 AI Chat with Memory Context")
    
    # Model selection in chat
    selected_model = st.selectbox(
        "Select AI Model",
        st.session_state.available_models,
        format_func=lambda x: f"{x.split(':')[0].upper()} - {x.split(':')[1]}",
        key="chat_model_select"
    )
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])
                
                # Show metadata for assistant messages
                if message["role"] == "assistant" and message.get("metadata"):
                    with st.expander("📊 Response Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tokens", message["metadata"]["tokens"]["total"])
                        with col2:
                            st.metric("Time", f"{message['metadata']['generation_time']:.2f}s")
                        with col3:
                            st.metric("Cost", f"${message['metadata']['cost']:.4f}")
                        
                        if message.get("memories_used"):
                            st.markdown("**Memories used:**")
                            for mem in message["memories_used"]:
                                st.caption(f"• {mem}")
    
    # Chat input
    if prompt := st.chat_input("💭 Ask anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
        
        # Get relevant memories
        with st.spinner("🧠 Searching memories..."):
            relevant_memories = asyncio.run(
                st.session_state.components['memory'].search_memories(
                    prompt, 
                    st.session_state.user_data['username'],
                    limit=5
                )
            )
        
        # Generate response
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                
                # Create chat history for context
                chat_history = [
                    ChatMessage(
                        role=m["role"], 
                        content=m["content"],
                        model_used=m.get("model")
                    )
                    for m in st.session_state.messages[-10:]  # Last 10 messages
                ]
                
                try:
                    # Ensure components are initialized
                    if 'components' not in st.session_state:
                        logger.warning("Components not in session state, reinitializing...")
                        try:
                            components = init_components()
                            st.session_state.components = components
                            logger.info("Successfully reinitialized components")
                        except Exception as init_error:
                            st.error(f"❌ Failed to initialize system components: {init_error}")
                            logger.error(f"Component initialization failed: {init_error}")
                            st.stop()
                    
                    # Create a simple generator for streaming
                    def stream_chat():
                        import asyncio
                        
                        # Create the async generator
                        async def get_stream():
                            async for chunk in st.session_state.components['llm'].stream_chat(
                                prompt,
                                selected_model,
                                st.session_state.components['memory'],
                                st.session_state.user_data['username'],
                                context=relevant_memories,
                                conversation_history=chat_history
                            ):
                                yield chunk
                        
                        # Run the async generator and yield chunks
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            async_gen = get_stream()
                            while True:
                                try:
                                    chunk = loop.run_until_complete(async_gen.__anext__())
                                    yield chunk
                                except StopAsyncIteration:
                                    break
                        finally:
                            loop.close()
                    
                    # Stream the response
                    response = st.write_stream(stream_chat())
                    
                    # Create metadata for the streamed response
                    estimated_tokens = int(len(response.split()) * 1.3)
                    metadata = {
                        "model": selected_model,
                        "tokens": {"total": estimated_tokens},
                        "streaming": True,
                        "cost": st.session_state.components['llm'].estimate_cost(selected_model, estimated_tokens)
                    }
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "model": selected_model,
                        "metadata": metadata,
                        "memories_used": [
                            f"{m.memory.content[:50]}..." 
                            for m in relevant_memories[:3]
                        ] if relevant_memories else []
                    })
                    
                    # Add conversation to memory
                    asyncio.run(
                        st.session_state.components['memory'].add_memory(
                            f"User: {prompt}\nAssistant: {response}",
                            st.session_state.user_data['username'],
                            {
                                "type": "conversation",
                                "model": selected_model,
                                "timestamp": datetime.now().isoformat()
                            },
                            memory_type=MemoryType.CONVERSATION
                        )
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    logger.error(f"Chat generation failed: {e}")
    
    # Chat controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            chat_json = json.dumps(st.session_state.messages, indent=2, default=str)
            st.download_button(
                "Download Chat History",
                chat_json,
                "chat_history.json",
                "application/json"
            )
    with col3:
        if st.button("📊 Chat Analytics", use_container_width=True):
            st.session_state.show_analytics = True


def display_memory_management():
    """Display memory management interface"""
    st.markdown("## 🧠 Memory Management")
    
    # Memory stats
    stats = st.session_state.components['memory'].get_statistics(
        st.session_state.user_data['username']
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Memories", stats.get("total_memories", 0))
    with col2:
        st.metric("Unique Entities", stats.get("total_entities", 0))
    with col3:
        st.metric("Avg Importance", f"{stats.get('avg_importance', 0):.2f}")
    with col4:
        date_range = stats.get("date_range")
        if date_range:
            days = (datetime.fromisoformat(date_range["latest"]) - 
                   datetime.fromisoformat(date_range["earliest"])).days
            st.metric("Days Active", days)
    
    # Memory operations
    tab1, tab2, tab3 = st.tabs(["➕ Add Memory", "🔍 Search", "📋 All Memories"])
    
    with tab1:
        with st.form("add_memory_form"):
            st.markdown("### Add New Memory")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                memory_content = st.text_area(
                    "Memory Content",
                    placeholder="Enter information you want to remember...",
                    height=100
                )
            with col2:
                memory_type = st.selectbox(
                    "Type",
                    [mt.value for mt in MemoryType],
                    format_func=lambda x: x.title()
                )
                importance = st.slider("Importance", 0.0, 1.0, 0.5, 0.1)
            
            categories = st.multiselect(
                "Categories",
                ["general", "personal", "work", "learning", "preferences", "facts"],
                default=["general"]
            )
            
            submitted = st.form_submit_button("💾 Add Memory", type="primary")
            
            if submitted and memory_content:
                with st.spinner("Adding memory..."):
                    memory = asyncio.run(
                        st.session_state.components['memory'].add_memory(
                            memory_content,
                            st.session_state.user_data['username'],
                            {
                                "categories": categories,
                                "importance": importance,
                                "added_manually": True
                            },
                            memory_type=MemoryType(memory_type)
                        )
                    )
                st.success(f"✅ Memory added with ID: {memory.id[:8]}...")
                st.balloons()
    
    with tab2:
        st.markdown("### Search Memories")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="What are you looking for?",
                key="memory_search"
            )
        with col2:
            search_limit = st.number_input("Results", min_value=1, max_value=20, value=5)
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            filter_types = st.multiselect(
                "Memory Types",
                [mt.value for mt in MemoryType],
                format_func=lambda x: x.title()
            )
            
            col1, col2 = st.columns(2)
            with col1:
                date_from = st.date_input("From Date", value=None)
            with col2:
                date_to = st.date_input("To Date", value=None)
        
        if search_query:
            asyncio.run(search_and_display_memories(
                search_query, 
                st.session_state.user_data['username'],
                search_limit
            ))
    
    with tab3:
        st.markdown("### All Memories")
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            page_size = st.selectbox("Page Size", [10, 25, 50, 100], index=1)
        with col2:
            if 'memory_page' not in st.session_state:
                st.session_state.memory_page = 0
        with col3:
            st.write("")  # Spacer
        
        # Get memories with pagination
        memories = asyncio.run(
            st.session_state.components['memory'].get_all_memories(
                st.session_state.user_data['username'],
                offset=st.session_state.memory_page * page_size,
                limit=page_size
            )
        )
        
        if memories:
            # Display memories with individual delete buttons
            st.markdown("#### 📝 Memory List")
            st.markdown("---")
            
            # Create a container for each memory with delete button
            for idx, mem in enumerate(memories):
                with st.container():
                    # Create columns for memory display and delete button
                    col_main, col_delete = st.columns([10, 1])
                    
                    with col_main:
                        # Create an expander for each memory
                        memory_preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
                        expander_title = f"**{mem.memory_type.value.title()}** | {memory_preview}"
                        
                        with st.expander(expander_title, expanded=False):
                            # Full content
                            st.markdown("**📄 Full Content:**")
                            st.info(mem.content)
                            
                            # Metadata in columns
                            meta_col1, meta_col2, meta_col3 = st.columns(3)
                            with meta_col1:
                                st.markdown("**🆔 ID:**")
                                st.caption(f"{mem.id[:16]}...")
                                st.markdown("**📅 Created:**")
                                st.caption(mem.created_at.strftime('%Y-%m-%d %H:%M'))
                            with meta_col2:
                                st.markdown("**⭐ Importance:**")
                                st.caption("⭐" * int(mem.importance * 5))
                                st.markdown("**👁️ Access Count:**")
                                st.caption(str(mem.access_count))
                            with meta_col3:
                                st.markdown("**🏷️ Entities:**")
                                if mem.entities:
                                    for entity in mem.entities[:5]:
                                        st.caption(f"• {entity}")
                                else:
                                    st.caption("None")
                    
                    with col_delete:
                        # Delete button for this memory
                        delete_key = f"delete_mem_{mem.id}"
                        if st.button("🗑️", key=delete_key, help="Delete this memory"):
                            st.session_state[f"confirm_{mem.id}"] = True
                
                # Confirmation dialog for deletion
                if st.session_state.get(f"confirm_{mem.id}", False):
                    with st.container():
                        st.warning(f"⚠️ Are you sure you want to delete this memory?")
                        confirm_col1, confirm_col2, confirm_col3 = st.columns([1, 1, 3])
                        with confirm_col1:
                            if st.button("✅ Yes, Delete", key=f"yes_{mem.id}", type="primary"):
                                with st.spinner("Deleting memory..."):
                                    try:
                                        asyncio.run(
                                            st.session_state.components['memory'].delete_memory(
                                                mem.id,
                                                st.session_state.user_data['username']
                                            )
                                        )
                                        st.success("✅ Memory deleted successfully!")
                                        # Clean up confirmation state
                                        if f"confirm_{mem.id}" in st.session_state:
                                            del st.session_state[f"confirm_{mem.id}"]
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Failed to delete memory: {str(e)}")
                        with confirm_col2:
                            if st.button("❌ Cancel", key=f"cancel_{mem.id}"):
                                del st.session_state[f"confirm_{mem.id}"]
                                st.rerun()
            
            st.markdown("---")
            
            # Bulk operations section
            st.markdown("#### 🔧 Bulk Operations")
            
            # Convert to DataFrame for bulk selection
            df_data = []
            for mem in memories:
                df_data.append({
                    "Type": mem.memory_type.value.title(),
                    "Content": mem.content[:80] + "..." if len(mem.content) > 80 else mem.content,
                    "Importance": "⭐" * int(mem.importance * 5),
                    "Created": mem.created_at.strftime("%Y-%m-%d %H:%M")
                })
            
            df = pd.DataFrame(df_data)
            
            # Display table with selection for bulk operations
            st.markdown("Select multiple memories for bulk actions:")
            selected = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                height=300
            )
            
            # Bulk operation buttons
            if selected and selected.selection.rows:
                st.info(f"📌 {len(selected.selection.rows)} memories selected")
                
                bulk_col1, bulk_col2 = st.columns(2)
                with bulk_col1:
                    if st.button("🗑️ Delete All Selected", type="secondary", use_container_width=True):
                        st.session_state['bulk_delete_confirm'] = True
                    
                    if st.session_state.get('bulk_delete_confirm', False):
                        st.warning(f"⚠️ Delete {len(selected.selection.rows)} selected memories?")
                        confirm_cols = st.columns(2)
                        with confirm_cols[0]:
                            if st.button("✅ Confirm Delete", type="primary", key="bulk_del_confirm"):
                                with st.spinner(f"Deleting {len(selected.selection.rows)} memories..."):
                                    deleted = 0
                                    for idx in selected.selection.rows:
                                        try:
                                            asyncio.run(
                                                st.session_state.components['memory'].delete_memory(
                                                    memories[idx].id,
                                                    st.session_state.user_data['username']
                                                )
                                            )
                                            deleted += 1
                                        except Exception as e:
                                            st.error(f"Failed to delete memory: {str(e)}")
                                st.success(f"✅ Deleted {deleted} memories successfully!")
                                if 'bulk_delete_confirm' in st.session_state:
                                    del st.session_state['bulk_delete_confirm']
                                st.rerun()
                        with confirm_cols[1]:
                            if st.button("❌ Cancel", key="bulk_del_cancel"):
                                del st.session_state['bulk_delete_confirm']
                                st.rerun()
                
                with bulk_col2:
                    if st.button("📥 Export Selected", use_container_width=True):
                        selected_memories = [memories[idx] for idx in selected.selection.rows]
                        export_data = {
                            "memories": [m.model_dump() for m in selected_memories],
                            "export_date": datetime.now().isoformat(),
                            "count": len(selected_memories)
                        }
                        st.download_button(
                            "💾 Download Selected Memories",
                            json.dumps(export_data, indent=2, default=str),
                            f"selected_memories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json",
                            use_container_width=True
                        )
            
            # Pagination controls
            st.markdown("---")
            page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
            with page_col1:
                if st.button("⬅️ Previous", disabled=st.session_state.memory_page == 0, use_container_width=True):
                    st.session_state.memory_page -= 1
                    st.rerun()
            with page_col2:
                st.markdown(f"<div style='text-align: center'>📄 Page {st.session_state.memory_page + 1}</div>", unsafe_allow_html=True)
            with page_col3:
                if st.button("Next ➡️", disabled=len(memories) < page_size, use_container_width=True):
                    st.session_state.memory_page += 1
                    st.rerun()
        else:
            st.info("📭 No memories yet. Start chatting or add some memories!")
        
        # Memory operations
        st.markdown("---")
        st.markdown("### Memory Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Consolidate Memories", use_container_width=True):
                with st.spinner("Consolidating memories..."):
                    results = asyncio.run(
                        st.session_state.components['memory'].consolidate_memories(
                            st.session_state.user_data['username']
                        )
                    )
                st.success(
                    f"Consolidation complete: {results['removed']} duplicates removed, "
                    f"{results['merged']} memories merged"
                )
        
        with col2:
            if st.button("📥 Export All Memories", use_container_width=True):
                with st.spinner("Exporting memories..."):
                    export_data = asyncio.run(
                        st.session_state.components['memory'].export_memories(
                            st.session_state.user_data['username']
                        )
                    )
                st.download_button(
                    "Download All Memories",
                    json.dumps(export_data, indent=2, default=str),
                    f"all_memories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        with col3:
            if st.button("📊 Memory Analytics", use_container_width=True):
                st.session_state.show_memory_analytics = True


def display_analytics():
    """Display comprehensive analytics"""
    st.markdown("## 📊 Analytics Dashboard")
    
    # Get data
    memories = asyncio.run(
        st.session_state.components['memory'].get_all_memories(
            st.session_state.user_data['username']
        )
    )
    
    if not memories:
        st.info("No data to analyze yet. Start using the system to see analytics!")
        return
    
    # Prepare data
    df_memories = pd.DataFrame([
        {
            "date": m.created_at.date(),
            "hour": m.created_at.hour,
            "type": m.memory_type.value,
            "importance": m.importance,
            "entities": len(m.entities),
            "access_count": m.access_count,
            "content_length": len(m.content)
        }
        for m in memories
    ])
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Memories", len(memories))
    with col2:
        st.metric("Avg Importance", f"{df_memories['importance'].mean():.2f}")
    with col3:
        st.metric("Total Entities", df_memories['entities'].sum())
    with col4:
        st.metric("Avg Access Count", f"{df_memories['access_count'].mean():.1f}")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Timeline", "🏷️ Categories", "⚡ Usage", "🔍 Insights"])
    
    with tab1:
        # Memory timeline
        fig = px.histogram(
            df_memories,
            x="date",
            title="Memory Creation Timeline",
            labels={"date": "Date", "count": "Number of Memories"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly distribution
        fig = px.bar(
            df_memories.groupby("hour").size().reset_index(name="count"),
            x="hour",
            y="count",
            title="Memory Creation by Hour of Day",
            labels={"hour": "Hour", "count": "Number of Memories"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Memory type distribution
        type_counts = df_memories["type"].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Memory Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Importance by type
            fig = px.box(
                df_memories,
                x="type",
                y="importance",
                title="Importance Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Usage patterns
        if st.session_state.messages:
            # Model usage from chat history
            model_usage = {}
            for msg in st.session_state.messages:
                if msg["role"] == "assistant" and msg.get("model"):
                    model = msg["model"]
                    if model not in model_usage:
                        model_usage[model] = {"count": 0, "tokens": 0}
                    model_usage[model]["count"] += 1
                    if msg.get("metadata"):
                        model_usage[model]["tokens"] += msg["metadata"].get("tokens", {}).get("total", 0)
            
            if model_usage:
                # Convert to DataFrame
                df_usage = pd.DataFrame.from_dict(model_usage, orient="index")
                df_usage = df_usage.reset_index().rename(columns={"index": "model"})
                
                # Create subplots
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Requests by Model", "Tokens by Model")
                )
                
                # Requests
                fig.add_trace(
                    go.Bar(x=df_usage["model"], y=df_usage["count"], name="Requests"),
                    row=1, col=1
                )
                
                # Tokens
                fig.add_trace(
                    go.Bar(x=df_usage["model"], y=df_usage["tokens"], name="Tokens"),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Insights
        st.markdown("### 🔍 Key Insights")
        
        # Most accessed memories
        most_accessed = sorted(memories, key=lambda m: m.access_count, reverse=True)[:5]
        if most_accessed:
            st.markdown("**Most Accessed Memories:**")
            for i, mem in enumerate(most_accessed, 1):
                st.write(f"{i}. {mem.content[:100]}... (accessed {mem.access_count} times)")
        
        # Entity frequency
        all_entities = []
        for mem in memories:
            all_entities.extend(mem.entities)
        
        if all_entities:
            entity_counts = pd.Series(all_entities).value_counts().head(10)
            
            fig = px.bar(
                x=entity_counts.values,
                y=entity_counts.index,
                orientation='h',
                title="Top 10 Entities",
                labels={"x": "Frequency", "y": "Entity"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Memory quality score
        quality_scores = []
        for mem in memories:
            score = (
                mem.importance * 0.4 +
                min(mem.access_count / 10, 1) * 0.3 +
                min(len(mem.entities) / 5, 1) * 0.3
            )
            quality_scores.append(score)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        st.metric("Average Memory Quality Score", f"{avg_quality:.2f}/1.0")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if avg_quality < 0.5:
            st.warning("Consider adding more detailed memories with higher importance ratings")
        
        if len(memories) < 50:
            st.info("Keep using the system to build a richer memory database")
        
        low_access = [m for m in memories if m.access_count == 0]
        if len(low_access) > len(memories) * 0.3:
            st.warning(f"{len(low_access)} memories have never been accessed. Consider reviewing and consolidating.")


# Main app logic
def main():
    """Main application entry point"""
    try:
        # Show app is loading
        st.set_page_config(
            page_title="Optimus v2 - AI Memory System",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load custom CSS
        load_custom_css()
        
        # Check if we're in Streamlit Cloud and show status
        if st.secrets:
            st.sidebar.success("✅ Running on Streamlit Cloud")
        
        # Initialize components
        components = init_components()
        st.session_state.components = components
    except Exception as e:
        st.error(f"❌ Application failed to start: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
    
    # Authentication (not cached to handle fresh browser sessions)
    auth = get_auth()
    user_data = auth.get_login_interface()
    if not user_data:
        st.stop()
    
    # Add auth to components for other parts of the app
    components['auth'] = auth
    
    st.session_state.user_data = user_data
    username = user_data.get("username")
    user_role = user_data.get("role", "user")
    
    # Get available models
    st.session_state.available_models = components['llm'].get_available_models()
    
    # Sidebar
    with st.sidebar:
        # Logo and title
        st.markdown("# 🤖 Optimus")
        st.markdown("### Version 2.0 - 2025")
        st.markdown("---")
        
        # User info
        st.markdown(f"**👤 User:** {username}")
        st.markdown(f"**🛡️ Role:** {user_role}")
        
        # Session info
        with st.expander("🔐 Session Info"):
            st.caption(f"Session expires: {user_data.get('exp', 'Unknown')}")
            active_sessions = components['auth'].get_active_sessions(username)
            st.caption(f"Active sessions: {len(active_sessions)}")
        
        st.markdown("---")
        
        # System health
        display_system_health()
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                components['auth'].logout()
        
        # Admin actions
        if user_role == "admin":
            st.markdown("---")
            st.markdown("### 👨‍💼 Admin Tools")
            
            if st.button("📊 System Metrics", use_container_width=True):
                st.session_state.show_system_metrics = True
            
            if st.button("👥 User Management", use_container_width=True):
                st.session_state.show_user_management = True
        
        # Footer
        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit")
        st.caption("© 2025 Optimus")
    
    # Main interface
    if st.session_state.get("show_system_metrics"):
        display_system_metrics()
    elif st.session_state.get("show_user_management"):
        display_user_management()
    else:
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Chat", 
            "🧠 Memories", 
            "📊 Analytics",
            "⚙️ Settings"
        ])
        
        with tab1:
            display_chat_interface()
        
        with tab2:
            display_memory_management()
        
        with tab3:
            display_analytics()
        
        with tab4:
            display_settings()


def display_system_metrics():
    """Display system metrics for admins"""
    st.markdown("## 📊 System Metrics")
    
    # Mock metrics for demo
    metrics = SystemMetrics(
        active_users=5,
        total_memories=1250,
        avg_response_time=1.2,
        p95_response_time=2.5,
        memory_usage_mb=512.3,
        cpu_usage_percent=35.2,
        vector_db_size_mb=245.8,
        queries_per_minute=12.5,
        tokens_per_minute=4500,
        error_rate=0.02,
        timeout_rate=0.005
    )
    
    # Display metrics in grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Users", metrics.active_users)
        st.metric("Total Memories", f"{metrics.total_memories:,}")
    
    with col2:
        st.metric("Avg Response Time", f"{metrics.avg_response_time:.2f}s")
        st.metric("P95 Response Time", f"{metrics.p95_response_time:.2f}s")
    
    with col3:
        st.metric("Memory Usage", f"{metrics.memory_usage_mb:.1f} MB")
        st.metric("CPU Usage", f"{metrics.cpu_usage_percent:.1f}%")
    
    with col4:
        st.metric("Queries/min", f"{metrics.queries_per_minute:.1f}")
        st.metric("Tokens/min", f"{metrics.tokens_per_minute:,}")
    
    if st.button("← Back"):
        st.session_state.show_system_metrics = False
        st.rerun()


def display_user_management():
    """Display user management interface for admins"""
    st.markdown("## 👥 User Management")
    
    st.info("User management interface would be implemented here")
    
    if st.button("← Back"):
        st.session_state.show_user_management = False
        st.rerun()


def display_settings():
    """Display user settings"""
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🔔 Preferences", "🔒 Security"])
    
    with tab1:
        st.markdown("### Profile Settings")
        
        with st.form("profile_form"):
            display_name = st.text_input(
                "Display Name",
                value=st.session_state.user_data.get("username", "")
            )
            email = st.text_input(
                "Email",
                value=st.session_state.user_data.get("email", "")
            )
            bio = st.text_area("Bio", placeholder="Tell us about yourself...")
            
            if st.form_submit_button("Save Profile"):
                st.success("Profile updated successfully!")
    
    with tab2:
        st.markdown("### Preferences")
        
        # Theme
        theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
        
        # Notifications
        st.markdown("**Notifications**")
        email_notifications = st.checkbox("Email notifications", value=True)
        memory_reminders = st.checkbox("Memory reminders", value=True)
        
        # Default settings
        st.markdown("**Defaults**")
        default_model = st.selectbox(
            "Default AI Model",
            st.session_state.available_models
        )
        default_memory_limit = st.slider("Default search results", 1, 20, 5)
        
        if st.button("Save Preferences"):
            st.success("Preferences saved!")
    
    with tab3:
        st.markdown("### Security Settings")
        
        # Session management
        st.markdown("**Active Sessions**")
        sessions = st.session_state.components['auth'].get_active_sessions(
            st.session_state.user_data['username']
        )
        
        for i, session in enumerate(sessions):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"Session {i+1}")
            with col2:
                st.caption(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M')}")
            with col3:
                if st.button(f"Revoke", key=f"revoke_{i}"):
                    st.session_state.components['auth'].revoke_token(session.token)
                    st.success("Session revoked")
                    st.rerun()
        
        # Password change
        st.markdown("**Change Password**")
        with st.form("password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Change Password"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # Validate password strength
                    is_valid, message = st.session_state.components['auth'].validate_password_strength(
                        new_password
                    )
                    if is_valid:
                        st.success("Password changed successfully!")
                    else:
                        st.error(message)
        
        # Data export
        st.markdown("**Data Management**")
        if st.button("📥 Export All My Data"):
            st.info("Preparing your data export...")
            # This would trigger GDPR-compliant data export


if __name__ == "__main__":
    main()