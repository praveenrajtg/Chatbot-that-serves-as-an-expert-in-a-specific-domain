from typing import List, Dict, Tuple
import pandas as pd
from data_processor import ArxivDataProcessor
from nlp_processor import NLPProcessor
import json
import os

class DomainExpertChatbot:
    def __init__(self, data_path: str = None):
        self.data_processor = ArxivDataProcessor(data_path)
        self.nlp_processor = NLPProcessor()
        self.conversation_history = []
        self.current_context = ""
        
    def initialize(self, subset_size: int = 1000):
        """Initialize the chatbot with data and embeddings"""
        print("Loading arXiv data...")
        self.data_processor.load_arxiv_data(subset_size)
        
        print("Creating embeddings...")
        self.data_processor.create_embeddings()
        
        print("Chatbot initialized successfully!")
    
    def chat(self, user_input: str) -> Dict:
        """Main chat interface"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Determine intent
        intent = self._classify_intent(user_input)
        
        response = {}
        
        if intent == "search_papers":
            response = self._handle_paper_search(user_input)
        elif intent == "explain_concept":
            response = self._handle_concept_explanation(user_input)
        elif intent == "summarize":
            response = self._handle_summarization(user_input)
        elif intent == "question_answering":
            response = self._handle_question_answering(user_input)
        else:
            response = self._handle_general_query(user_input)
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response["answer"]})
        
        return response
    
    def _classify_intent(self, user_input: str) -> str:
        """Classify user intent based on input"""
        user_input_lower = user_input.lower()
        
        if any(keyword in user_input_lower for keyword in ["search", "find papers", "papers about", "research on"]):
            return "search_papers"
        elif any(keyword in user_input_lower for keyword in ["explain", "what is", "define", "concept"]):
            return "explain_concept"
        elif any(keyword in user_input_lower for keyword in ["summarize", "summary", "abstract"]):
            return "summarize"
        elif "?" in user_input:
            return "question_answering"
        else:
            return "general_query"
    
    def _handle_paper_search(self, query: str) -> Dict:
        """Handle paper search requests"""
        # Extract search terms
        search_terms = query.lower().replace("search", "").replace("find papers", "").replace("papers about", "").strip()
        
        # Search papers
        results = self.data_processor.search_papers(search_terms, top_k=5)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant papers for your query.",
                "papers": [],
                "follow_up_questions": []
            }
        
        # Format response
        answer = f"I found {len(results)} relevant papers:\n\n"
        for i, paper in enumerate(results, 1):
            answer += f"{i}. **{paper['title']}**\n"
            answer += f"   Authors: {paper['authors']}\n"
            answer += f"   Categories: {paper['categories']}\n"
            answer += f"   Relevance Score: {paper['score']:.3f}\n\n"
        
        # Generate follow-up questions
        follow_ups = [
            "Would you like me to summarize any of these papers?",
            "Do you want to know more about specific concepts from these papers?",
            "Should I search for more papers on this topic?"
        ]
        
        return {
            "answer": answer,
            "papers": results,
            "follow_up_questions": follow_ups
        }
    
    def _handle_concept_explanation(self, query: str) -> Dict:
        """Handle concept explanation requests"""
        # Extract concept
        concept = query.lower().replace("explain", "").replace("what is", "").replace("define", "").strip()
        
        # Search for relevant papers
        papers = self.data_processor.search_papers(concept, top_k=3)
        
        if not papers:
            return {
                "answer": f"I don't have enough information about '{concept}' in my knowledge base.",
                "papers": [],
                "follow_up_questions": []
            }
        
        # Combine abstracts for context
        context = " ".join([paper['abstract'] for paper in papers])
        
        # Generate explanation
        explanation = self.nlp_processor.explain_concept(concept, context)
        
        # Get key concepts
        key_concepts = self.nlp_processor.extract_key_concepts(context, top_k=5)
        
        answer = f"**{concept.title()}**\n\n{explanation}\n\n"
        answer += "**Related Concepts:**\n"
        for concept_name, score in key_concepts:
            answer += f"- {concept_name}\n"
        
        # Generate follow-up questions
        follow_ups = self.nlp_processor.generate_follow_up_questions(context, 3)
        
        return {
            "answer": answer,
            "papers": papers,
            "follow_up_questions": follow_ups,
            "key_concepts": key_concepts
        }
    
    def _handle_summarization(self, query: str) -> Dict:
        """Handle summarization requests"""
        # Check if user wants to summarize a specific paper
        if "paper" in query.lower():
            # For demo, summarize the first paper from recent search
            if hasattr(self, 'last_search_results') and self.last_search_results:
                paper = self.last_search_results[0]
                summary = self.nlp_processor.summarize_text(paper['abstract'])
                
                answer = f"**Summary of '{paper['title']}':**\n\n{summary}"
                
                return {
                    "answer": answer,
                    "papers": [paper],
                    "follow_up_questions": ["Would you like to know more about specific aspects of this paper?"]
                }
        
        return {
            "answer": "Please specify which paper you'd like me to summarize, or search for papers first.",
            "papers": [],
            "follow_up_questions": ["What topic would you like me to search papers for?"]
        }
    
    def _handle_question_answering(self, question: str) -> Dict:
        """Handle direct questions"""
        # Use current context or search for relevant papers
        if self.current_context:
            context = self.current_context
        else:
            # Search for relevant papers
            papers = self.data_processor.search_papers(question, top_k=3)
            context = " ".join([paper['abstract'] for paper in papers])
        
        # Answer question
        result = self.nlp_processor.answer_question(question, context)
        
        answer = result['answer']
        if result['confidence'] < 0.3:
            answer += "\n\n*Note: I'm not very confident about this answer. You might want to search for more specific papers.*"
        
        return {
            "answer": answer,
            "confidence": result['confidence'],
            "follow_up_questions": ["Would you like me to search for more specific information on this topic?"]
        }
    
    def _handle_general_query(self, query: str) -> Dict:
        """Handle general queries"""
        # Search for relevant papers
        papers = self.data_processor.search_papers(query, top_k=3)
        
        if papers:
            context = " ".join([paper['abstract'] for paper in papers])
            summary = self.nlp_processor.summarize_text(context)
            
            answer = f"Based on recent research, here's what I found:\n\n{summary}\n\n"
            answer += "This information is based on the following papers:\n"
            for paper in papers:
                answer += f"- {paper['title']}\n"
        else:
            answer = "I don't have specific information about that topic in my current knowledge base."
        
        return {
            "answer": answer,
            "papers": papers,
            "follow_up_questions": ["Would you like me to search for more specific information?"]
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.current_context = ""