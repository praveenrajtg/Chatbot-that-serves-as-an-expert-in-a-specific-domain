import streamlit as st
import pandas as pd
import plotly.express as px
from chatbot import DomainExpertChatbot
from visualization import VisualizationUtils
import base64
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="Domain Expert Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with interactive animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(31, 119, 180, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(31, 119, 180, 0.8)); }
    }
    
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        color: #000000 !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #2196f3;
        color: #000000 !important;
        animation: slideInRight 0.5s ease;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f3e5f5, #e1bee7);
        border-left: 5px solid #9c27b0;
        color: #000000 !important;
        animation: slideInLeft 0.5s ease;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #2196f3, #21cbf3);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.5);
        background: linear-gradient(45deg, #1976d2, #1cb5e0);
    }
    
    .paper-card {
        border: 2px solid transparent;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(white, white) padding-box,
                   linear-gradient(45deg, #2196f3, #9c27b0) border-box;
        transition: all 0.3s ease;
    }
    
    .paper-card:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .stExpander {
        border-radius: 10px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .stExpander:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.2);
    }
    
    /* Dark theme support */
    [data-theme="dark"] .user-message {
        background: linear-gradient(135deg, #1e3a8a, #1e40af);
        color: #ffffff !important;
    }
    [data-theme="dark"] .bot-message {
        background: linear-gradient(135deg, #7c2d92, #8b5cf6);
        color: #ffffff !important;
    }
    [data-theme="dark"] .chat-message {
        color: #ffffff !important;
    }
    
    .chat-message strong {
        color: inherit !important;
    }
    
    /* Typing animation */
    .typing {
        display: inline-block;
        animation: typing 1.5s infinite;
    }
    
    @keyframes typing {
        0%, 60%, 100% { opacity: 1; }
        30% { opacity: 0.5; }
    }
    
    /* Pulse animation for new messages */
    .new-message {
        animation: pulse 0.6s ease-in-out;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    """Load and initialize the chatbot"""
    chatbot = DomainExpertChatbot()
    chatbot.initialize(subset_size=500)  # Smaller subset for demo
    return chatbot

@st.cache_resource
def load_visualizer():
    """Load visualization utilities"""
    return VisualizationUtils()

def main():
    # Animated header with emoji
    st.markdown('<h1 class="main-header">🤖✨ Domain Expert Chatbot ✨🤖</h1>', unsafe_allow_html=True)
    
    # Interactive subtitle with progress
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown("### 🚀 AI-powered research assistant for Computer Science papers")
        if 'chatbot' not in st.session_state:
            progress_bar = st.progress(0)
            status_text = st.empty()
    
    # Initialize components with progress
    if 'chatbot' not in st.session_state:
        with st.spinner("🔄 Initializing AI models..."):
            progress_bar.progress(25)
            status_text.text("Loading chatbot...")
            st.session_state.chatbot = load_chatbot()
            progress_bar.progress(75)
            status_text.text("Loading visualizations...")
            st.session_state.visualizer = load_visualizer()
            progress_bar.progress(100)
            status_text.text("Ready! 🎉")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        st.balloons()  # Celebration animation
        st.success("🎉 Chatbot initialized successfully!")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_papers' not in st.session_state:
        st.session_state.current_papers = []
    
    # Interactive sidebar with animations
    with st.sidebar:
        st.markdown("## 🔧 Control Center")
        
        # Animated clear button
        if st.button("🗑️ Clear Conversation", help="Reset the entire conversation"):
            st.session_state.messages = []
            st.session_state.current_papers = []
            st.session_state.chatbot.clear_history()
            st.success("✨ Conversation cleared!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        st.markdown("## 🚀 Quick Actions")
        
        # Interactive sample queries with categories
        query_categories = {
            "🔍 Search Queries": [
                "Search for papers about neural networks",
                "Find research on computer vision",
                "Papers about natural language processing"
            ],
            "🧠 Concept Explanations": [
                "Explain transformer architecture",
                "What is reinforcement learning?",
                "Define convolutional neural networks"
            ],
            "📊 Advanced Topics": [
                "How do attention mechanisms work?",
                "What is transfer learning?",
                "Explain generative adversarial networks"
            ]
        }
        
        for category, queries in query_categories.items():
            with st.expander(category, expanded=False):
                for query in queries:
                    if st.button(f"💡 {query}", key=f"sample_{query}"):
                        st.session_state.messages.append({"role": "user", "content": query})
                        with st.spinner("🤔 Processing your query..."):
                            response = st.session_state.chatbot.chat(query)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            if response.get('papers'):
                                st.session_state.current_papers = response['papers']
                        st.success("✅ Response generated!")
                        time.sleep(0.5)
                        st.rerun()
        
        # Animated statistics
        st.markdown("---")
        st.markdown("## 📊 Live Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="💬 Messages", 
                value=len(st.session_state.messages),
                delta=1 if st.session_state.messages else 0
            )
        with col2:
            st.metric(
                label="📝 Papers", 
                value=len(st.session_state.current_papers),
                delta=len(st.session_state.current_papers) if st.session_state.current_papers else 0
            )
        
        # Fun facts
        if len(st.session_state.messages) > 0:
            st.info(f"🎆 You've had {len(st.session_state.messages)//2} conversations!")
        
        # Theme toggle (visual only)
        st.markdown("---")
        st.markdown("## 🎨 Interface")
        theme_choice = st.selectbox("🎨 Visual Theme", ["Auto", "Light", "Dark"])
        if theme_choice:
            st.info(f"✨ Theme set to {theme_choice}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat container
        chat_container = st.container()
        
        # Display conversation
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    response = message["content"]
                    st.markdown(f'<div class="chat-message bot-message"><strong>Bot:</strong> {response["answer"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Show follow-up questions
                    if response.get('follow_up_questions'):
                        st.markdown("**Suggested follow-up questions:**")
                        for i, question in enumerate(response['follow_up_questions']):
                            if st.button(f"❓ {question}", key=f"followup_{len(st.session_state.messages)}_{i}"):
                                st.session_state.messages.append({"role": "user", "content": question})
                                with st.spinner("Processing..."):
                                    new_response = st.session_state.chatbot.chat(question)
                                    st.session_state.messages.append({"role": "assistant", "content": new_response})
                                    if new_response.get('papers'):
                                        st.session_state.current_papers = new_response['papers']
                                st.rerun()
        
        # Chat input
        user_input = st.chat_input("Ask me about computer science research...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get bot response
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Update current papers
                if response.get('papers'):
                    st.session_state.current_papers = response['papers']
            
            st.rerun()
    
    with col2:
        st.header("📚 Paper Explorer")
        
        # Current papers
        if st.session_state.current_papers:
            st.subheader("Current Search Results")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📋 List", "📊 Visualizations", "🔍 Details"])
            
            with tab1:
                # Display papers as cards
                for i, paper in enumerate(st.session_state.current_papers):
                    with st.expander(f"📄 {paper['title'][:60]}..."):
                        st.write(f"**Authors:** {paper['authors']}")
                        st.write(f"**Categories:** {paper['categories']}")
                        st.write(f"**Relevance Score:** {paper['score']:.3f}")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        
                        # Action buttons
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button(f"📝 Summarize", key=f"summarize_{i}"):
                                summary_query = f"Summarize the paper: {paper['title']}"
                                st.session_state.messages.append({"role": "user", "content": summary_query})
                                with st.spinner("Summarizing..."):
                                    response = st.session_state.chatbot.chat(summary_query)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                st.rerun()
                        
                        with col_b:
                            if st.button(f"❓ Ask Question", key=f"ask_{i}"):
                                question_query = f"Tell me more about the methodology in: {paper['title']}"
                                st.session_state.messages.append({"role": "user", "content": question_query})
                                with st.spinner("Processing..."):
                                    response = st.session_state.chatbot.chat(question_query)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                st.rerun()
            
            with tab2:
                # Visualizations
                st.subheader("📊 Paper Analysis")
                
                # Category distribution
                category_fig = st.session_state.visualizer.create_category_distribution(st.session_state.current_papers)
                if category_fig:
                    st.plotly_chart(category_fig, width='stretch')
                
                # Paper timeline
                timeline_fig = st.session_state.visualizer.create_paper_timeline(st.session_state.current_papers)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, width='stretch')
                
                # Network visualization
                if len(st.session_state.current_papers) > 1:
                    network_fig = st.session_state.visualizer.create_paper_similarity_network(st.session_state.current_papers)
                    if network_fig:
                        st.plotly_chart(network_fig, width='stretch')
            
            with tab3:
                # Detailed table
                st.subheader("📋 Detailed Information")
                results_df = st.session_state.visualizer.create_search_results_table(st.session_state.current_papers)
                if not results_df.empty:
                    st.dataframe(results_df, width='stretch')
        
        else:
            # Welcome screen with interactive elements
            st.markdown("""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 1rem 0;'>
                <h2>🔍 Ready to Explore Research?</h2>
                <p>Search for papers to see interactive results here!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick start buttons
            st.markdown("### 🚀 Quick Start")
            
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                if st.button("🧠 AI Research", width='stretch'):
                    query = "Search for papers about artificial intelligence"
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.spinner("🔍 Searching..."):
                        response = st.session_state.chatbot.chat(query)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        if response.get('papers'):
                            st.session_state.current_papers = response['papers']
                    st.rerun()
            
            with col_y:
                if st.button("📊 Machine Learning", width='stretch'):
                    query = "Find research on machine learning"
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.spinner("🔍 Searching..."):
                        response = st.session_state.chatbot.chat(query)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        if response.get('papers'):
                            st.session_state.current_papers = response['papers']
                    st.rerun()
            
            with col_z:
                if st.button("👁️ Computer Vision", width='stretch'):
                    query = "Papers about computer vision"
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.spinner("🔍 Searching..."):
                        response = st.session_state.chatbot.chat(query)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        if response.get('papers'):
                            st.session_state.current_papers = response['papers']
                    st.rerun()
            
            # Interactive tips
            st.markdown("### 💡 Try These Queries:")
            tips = [
                "🔍 Search: 'Find papers about neural networks'",
                "🧠 Explain: 'What is deep learning?'",
                "📝 Summarize: 'Summarize transformer architecture'",
                "❓ Ask: 'How do CNNs work?'"
            ]
            
            for tip in tips:
                st.markdown(f"- {tip}")
            
            # Fun animation
            st.markdown("""
            <div style='text-align: center; margin-top: 2rem;'>
                <div class='typing'>🤖</div>
                <div class='typing'>📚</div>
                <div class='typing'>✨</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🤖 Domain Expert Chatbot | Powered by arXiv Dataset & Advanced NLP</p>
        <p>Built with Streamlit, Transformers, and FAISS</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()