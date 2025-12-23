import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss

class ArxivDataProcessor:
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.papers_df = None
        self.embeddings = None
        self.index = None
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def load_arxiv_data(self, subset_size: int = 10000) -> pd.DataFrame:
        """Load and preprocess arXiv dataset"""
        if self.data_path:
            # Load from local file
            self.papers_df = pd.read_json(self.data_path, lines=True)
        else:
            # Create sample data for demonstration
            self.papers_df = self._create_sample_data()
        
        # Filter for computer science papers
        cs_papers = self.papers_df[
            self.papers_df['categories'].str.contains('cs.', na=False)
        ].head(subset_size)
        
        # Clean and preprocess
        cs_papers = self._preprocess_papers(cs_papers)
        self.papers_df = cs_papers
        
        return self.papers_df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample arXiv-like data for demonstration"""
        sample_papers = [
            {
                'id': 'cs.AI/2023.001',
                'title': 'Advanced Neural Networks for Natural Language Processing',
                'abstract': 'This paper presents novel approaches to neural network architectures for NLP tasks. We introduce transformer-based models with improved attention mechanisms that achieve state-of-the-art results on multiple benchmarks.',
                'categories': 'cs.AI cs.CL',
                'authors': 'John Smith, Jane Doe',
                'update_date': '2023-01-15'
            },
            {
                'id': 'cs.CV/2023.002',
                'title': 'Computer Vision Applications in Medical Imaging',
                'abstract': 'We explore deep learning techniques for medical image analysis, focusing on convolutional neural networks for disease detection and diagnosis. Our approach shows significant improvements in accuracy.',
                'categories': 'cs.CV cs.LG',
                'authors': 'Alice Johnson, Bob Wilson',
                'update_date': '2023-02-20'
            },
            {
                'id': 'cs.LG/2023.003',
                'title': 'Reinforcement Learning for Autonomous Systems',
                'abstract': 'This work investigates reinforcement learning algorithms for autonomous vehicle navigation. We propose a novel reward function that improves learning efficiency and safety.',
                'categories': 'cs.LG cs.RO',
                'authors': 'Charlie Brown, Diana Prince',
                'update_date': '2023-03-10'
            }
        ]
        
        return pd.DataFrame(sample_papers * 100)  # Replicate for demo
    
    def _preprocess_papers(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess paper data"""
        papers_df = papers_df.copy()
        
        # Clean text
        papers_df['clean_abstract'] = papers_df['abstract'].apply(self._clean_text)
        papers_df['clean_title'] = papers_df['title'].apply(self._clean_text)
        
        # Combine title and abstract for better search
        papers_df['combined_text'] = papers_df['clean_title'] + ' ' + papers_df['clean_abstract']
        
        return papers_df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        
        return text
    
    def create_embeddings(self):
        """Create sentence embeddings for papers"""
        if self.papers_df is None:
            raise ValueError("Load data first using load_arxiv_data()")
        
        texts = self.papers_df['combined_text'].tolist()
        self.embeddings = self.sentence_model.encode(texts)
        
        # Create FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def search_papers(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant papers using semantic similarity"""
        if self.index is None:
            raise ValueError("Create embeddings first using create_embeddings()")
        
        # Encode query
        query_embedding = self.sentence_model.encode([query])
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            paper = self.papers_df.iloc[idx]
            results.append({
                'rank': i + 1,
                'score': float(score),
                'id': paper['id'],
                'title': paper['title'],
                'abstract': paper['abstract'],
                'authors': paper['authors'],
                'categories': paper['categories']
            })
        
        return results
    
    def get_paper_by_id(self, paper_id: str) -> Dict:
        """Get specific paper by ID"""
        paper = self.papers_df[self.papers_df['id'] == paper_id]
        if paper.empty:
            return None
        
        return paper.iloc[0].to_dict()