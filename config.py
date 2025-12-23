# Configuration file for Domain Expert Chatbot

# Dataset Configuration
DATASET_CONFIG = {
    "subset_size": 1000,  # Number of papers to load (reduce for faster startup)
    "categories_filter": ["cs."],  # Filter papers by categories (cs. for computer science)
    "data_path": None,  # Path to arXiv dataset file (None for sample data)
}

# Model Configuration
MODEL_CONFIG = {
    "sentence_model": "all-MiniLM-L6-v2",  # Sentence transformer model
    "summarization_model": "facebook/bart-large-cnn",  # Summarization model
    "qa_model": "distilbert-base-cased-distilled-squad",  # Question answering model
    "max_summary_length": 150,  # Maximum summary length
    "search_top_k": 5,  # Number of papers to return in search
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Domain Expert Chatbot",
    "page_icon": "🤖",
    "layout": "wide",
    "theme": "light",  # light or dark
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "use_gpu": False,  # Set to True if GPU available
    "batch_size": 32,  # Batch size for processing
    "max_context_length": 1024,  # Maximum context length for models
    "similarity_threshold": 0.3,  # Threshold for paper similarity network
}

# Sample queries for the sidebar
SAMPLE_QUERIES = [
    "Search for papers about neural networks",
    "Explain transformer architecture", 
    "Find research on computer vision",
    "What is reinforcement learning?",
    "Papers about natural language processing",
    "Define convolutional neural networks",
    "How do attention mechanisms work?",
    "What is transfer learning?",
    "Explain deep learning optimization",
    "Papers about generative adversarial networks"
]