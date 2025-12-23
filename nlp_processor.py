import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.summarizer = None
        self.qa_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models for summarization and QA"""
        try:
            # Load summarization model
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn"
            )
            
            # Load QA model
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to simpler models
            self.summarizer = pipeline("summarization", model="t5-small")
            self.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Generate summary of given text"""
        if len(text.split()) < 50:
            return text
        
        try:
            # Truncate text if too long
            max_input_length = 512
            if len(text.split()) > max_input_length:
                text = ' '.join(text.split()[:max_input_length])
            
            # Adjust max_length based on input length
            input_length = len(text.split())
            adjusted_max_length = min(max_length, max(30, input_length // 2))
            
            summary = self.summarizer(
                text,
                max_length=adjusted_max_length,
                min_length=20,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            # Fallback to extractive summarization
            return self._extractive_summary(text, max_length)
    
    def _extractive_summary(self, text: str, max_length: int) -> str:
        """Simple extractive summarization as fallback"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text
        
        # Score sentences by word frequency
        word_freq = Counter()
        for sentence in sentences:
            words = [word.lower() for word in word_tokenize(sentence) 
                    if word.lower() not in self.stop_words and word.isalpha()]
            word_freq.update(words)
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            words = [word.lower() for word in word_tokenize(sentence) 
                    if word.lower() not in self.stop_words and word.isalpha()]
            score = sum(word_freq[word] for word in words)
            sentence_scores[sentence] = score
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        summary_sentences = [sent[0] for sent in top_sentences[:3]]
        
        return ' '.join(summary_sentences)
    
    def answer_question(self, question: str, context: str) -> Dict:
        """Answer question based on given context"""
        try:
            result = self.qa_pipeline(question=question, context=context)
            return {
                'answer': result['answer'],
                'confidence': result['score'],
                'start': result['start'],
                'end': result['end']
            }
        except Exception as e:
            print(f"QA error: {e}")
            return {
                'answer': "I couldn't find a specific answer in the context.",
                'confidence': 0.0,
                'start': 0,
                'end': 0
            }
    
    def extract_key_concepts(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract key concepts using TF-IDF"""
        # Preprocess text
        sentences = sent_tokenize(text)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top concepts
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            concepts = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            return concepts
        except Exception as e:
            print(f"Concept extraction error: {e}")
            return []
    
    def explain_concept(self, concept: str, context: str) -> str:
        """Generate explanation for a concept based on context"""
        # Create a question about the concept
        questions = [
            f"What is {concept}?",
            f"How does {concept} work?",
            f"What is the definition of {concept}?"
        ]
        
        best_answer = ""
        best_confidence = 0.0
        
        for question in questions:
            result = self.answer_question(question, context)
            if result['confidence'] > best_confidence:
                best_answer = result['answer']
                best_confidence = result['confidence']
        
        if best_confidence > 0.1:
            return best_answer
        else:
            # Fallback explanation
            return f"{concept} is mentioned in the context but requires further research for detailed explanation."
    
    def generate_follow_up_questions(self, text: str, num_questions: int = 3) -> List[str]:
        """Generate potential follow-up questions based on text"""
        concepts = self.extract_key_concepts(text, top_k=5)
        
        question_templates = [
            "How does {} work in practice?",
            "What are the applications of {}?",
            "What are the advantages and disadvantages of {}?",
            "How does {} compare to other approaches?",
            "What are the future directions for {}?"
        ]
        
        questions = []
        for concept, _ in concepts[:num_questions]:
            template = np.random.choice(question_templates)
            questions.append(template.format(concept))
        
        return questions