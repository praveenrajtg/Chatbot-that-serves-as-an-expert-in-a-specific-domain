# Chatbot-that-serves-as-an-expert-in-a-specific-domain
# 🤖 Domain Expert Chatbot - arXiv Research Assistant

A sophisticated AI-powered chatbot that serves as a domain expert in Computer Science, capable of answering complex queries, explaining concepts, and providing summaries of research papers from the arXiv dataset.

## 📋 Features

- **Intelligent Paper Search**: Semantic search using sentence embeddings and FAISS for fast retrieval
- **Concept Explanation**: Advanced NLP techniques to explain complex computer science concepts
- **Paper Summarization**: Automatic summarization using transformer models (BART/T5)
- **Question Answering**: Context-aware Q&A using DistilBERT
- **Interactive Visualizations**: 
  - Paper similarity networks
  - Category distributions
  - Timeline visualizations
  - Concept importance charts
- **Follow-up Questions**: Intelligent generation of relevant follow-up questions
- **Streamlit UI**: Beautiful, responsive web interface

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **NLP Models**: 
  - Sentence Transformers (all-MiniLM-L6-v2)
  - BART/T5 for summarization
  - DistilBERT for question answering
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Visualization**: Plotly, WordCloud, NetworkX
- **Data Processing**: Pandas, NumPy, NLTK

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for downloading models)

### Step 1: Clone or Download the Project

```bash
cd c:\DomainExpert_bot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
# source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will download several large models (~2GB). First-time setup may take 10-15 minutes.

### Step 4: Download NLTK Data

The application will automatically download required NLTK data on first run, but you can pre-download:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 📊 Dataset Setup

### Option 1: Using Sample Data (Quick Start)

The application includes sample data for immediate testing. No additional setup required!

### Option 2: Using Real arXiv Dataset

1. **Download the arXiv dataset** from Kaggle:
   - Visit: https://www.kaggle.com/datasets/Cornell-University/arxiv
   - Download `arxiv-metadata-oai-snapshot.json`

2. **Place the dataset**:
   ```bash
   # Create data directory
   mkdir data
   
   # Move the downloaded file
   move arxiv-metadata-oai-snapshot.json data\
   ```

3. **Update the code** to use real data:
   - Open `app.py`
   - Modify the `load_chatbot()` function:
   ```python
   chatbot = DomainExpertChatbot(data_path='data/arxiv-metadata-oai-snapshot.json')
   ```

## 🚀 Running the Application

### Basic Run

```bash
streamlit run app.py
```

The application will:
1. Start the Streamlit server
2. Load and initialize the chatbot (may take 2-3 minutes first time)
3. Open automatically in your default browser at `http://localhost:8501`

### Custom Port

```bash
streamlit run app.py --server.port 8080
```

### Network Access

```bash
streamlit run app.py --server.address 0.0.0.0
```

## 📖 Usage Guide

### 1. Starting a Conversation

Once the app loads, you can:
- Type questions in the chat input at the bottom
- Click on sample queries in the sidebar
- Use the suggested follow-up questions

### 2. Sample Queries

**Paper Search:**
```
- "Search for papers about neural networks"
- "Find research on computer vision"
- "Papers about natural language processing"
```

**Concept Explanation:**
```
- "Explain transformer architecture"
- "What is reinforcement learning?"
- "Define convolutional neural networks"
```

**Summarization:**
```
- "Summarize this paper"
- "Give me a summary of the first paper"
```

**Question Answering:**
```
- "How do transformers work?"
- "What are the applications of deep learning?"
- "What is the difference between supervised and unsupervised learning?"
```

### 3. Exploring Papers

- **List View**: Browse papers with expandable details
- **Visualizations**: View category distributions, timelines, and similarity networks
- **Details**: See comprehensive table with all paper information

### 4. Interactive Features

- **Summarize Button**: Get quick summaries of specific papers
- **Ask Question Button**: Ask follow-up questions about papers
- **Follow-up Suggestions**: Click on suggested questions for deeper exploration

## 🏗️ Project Structure

```
DomainExpert_bot/
│
├── app.py                  # Main Streamlit application
├── chatbot.py             # Core chatbot logic and conversation handling
├── data_processor.py      # arXiv data loading and preprocessing
├── nlp_processor.py       # NLP utilities (summarization, QA, concept extraction)
├── visualization.py       # Visualization utilities
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── data/                 # Dataset directory (create if using real data)
│   └── arxiv-metadata-oai-snapshot.json
│
└── venv/                 # Virtual environment (created during setup)
```

## 🔧 Configuration

### Adjusting Dataset Size

In `app.py`, modify the `subset_size` parameter:

```python
chatbot.initialize(subset_size=1000)  # Use 1000 papers
```

### Changing Models

In `nlp_processor.py`, you can swap models:

```python
# For faster but less accurate summarization
self.summarizer = pipeline("summarization", model="t5-small")

# For better quality (requires more memory)
self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
```

### Search Parameters

In `chatbot.py`, adjust search results:

```python
results = self.data_processor.search_papers(search_terms, top_k=10)  # Get 10 results
```

## 🐛 Troubleshooting

### Issue: "Out of Memory" Error

**Solution**: Reduce the subset size or use smaller models
```python
# In app.py
chatbot.initialize(subset_size=100)  # Smaller dataset

# In nlp_processor.py
self.summarizer = pipeline("summarization", model="t5-small")
```

### Issue: Slow Loading

**Solution**: 
- First-time model downloads are slow (normal)
- Reduce dataset size
- Use CPU-optimized models

### Issue: NLTK Data Not Found

**Solution**:
```bash
python -c "import nltk; nltk.download('all')"
```

### Issue: Port Already in Use

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

## 📊 Performance Optimization

### For Better Speed:
1. Use smaller models (t5-small instead of bart-large)
2. Reduce subset_size to 500-1000 papers
3. Use faiss-gpu instead of faiss-cpu (if GPU available)

### For Better Quality:
1. Use larger models (bart-large-cnn)
2. Increase subset_size to 10000+ papers
3. Fine-tune models on domain-specific data

## 🔬 Advanced Features

### Custom Domain Training

To train on a specific CS subdomain:

```python
# In data_processor.py, modify the filter
cs_papers = self.papers_df[
    self.papers_df['categories'].str.contains('cs.AI', na=False)  # Only AI papers
].head(subset_size)
```

### Adding New Visualizations

Extend `visualization.py`:

```python
def create_custom_viz(self, data):
    # Your custom visualization logic
    fig = px.scatter(...)
    return fig
```

## 📝 Testing

### Quick Test Commands

```bash
# Test data processor
python -c "from data_processor import ArxivDataProcessor; dp = ArxivDataProcessor(); dp.load_arxiv_data(100); print('Data loaded successfully')"

# Test NLP processor
python -c "from nlp_processor import NLPProcessor; nlp = NLPProcessor(); print('NLP models loaded')"

# Test chatbot
python -c "from chatbot import DomainExpertChatbot; bot = DomainExpertChatbot(); bot.initialize(100); print('Chatbot ready')"
```

## 🤝 Contributing

To extend this project:

1. Add new NLP capabilities in `nlp_processor.py`
2. Implement additional visualizations in `visualization.py`
3. Enhance conversation logic in `chatbot.py`
4. Improve UI/UX in `app.py`

## 📄 License

This project is for educational purposes. The arXiv dataset is provided by Cornell University under their terms of use.

## 🙏 Acknowledgments

- **arXiv** for providing the research paper dataset
- **Hugging Face** for transformer models
- **Streamlit** for the web framework
- **Facebook AI** for FAISS vector search

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with sample data first before using full dataset

## 🎯 Future Enhancements

- [ ] Multi-domain support (Physics, Math, Biology)
- [ ] Paper recommendation system
- [ ] Citation network analysis
- [ ] Export conversation history
- [ ] Fine-tuned domain-specific models
- [ ] Real-time arXiv API integration
- [ ] User authentication and saved sessions

---

Happy Researching! 🚀

Activate the Env after that run commend streamlit run app.py 
