import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx
from typing import List, Dict, Tuple
import io
import base64

class VisualizationUtils:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_concept_wordcloud(self, concepts: List[Tuple[str, float]]) -> str:
        """Create word cloud from concepts"""
        if not concepts:
            return None
        
        # Create frequency dictionary
        word_freq = {concept: score for concept, score in concepts}
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate_from_frequencies(word_freq)
        
        # Convert to base64 for Streamlit
        img_buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        
        return img_str
    
    def create_paper_similarity_network(self, papers: List[Dict], similarity_threshold: float = 0.3) -> go.Figure:
        """Create network visualization of paper similarities"""
        if len(papers) < 2:
            return None
        
        # Create similarity matrix (simplified)
        n_papers = len(papers)
        similarity_matrix = np.random.rand(n_papers, n_papers)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, paper in enumerate(papers):
            G.add_node(i, title=paper['title'][:50] + "...", 
                      authors=paper['authors'], 
                      score=paper.get('score', 0))
        
        # Add edges based on similarity
        for i in range(n_papers):
            for j in range(i+1, n_papers):
                if similarity_matrix[i][j] > similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
        
        # Get layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(papers[node]['title'][:30])
            node_info.append(f"Title: {papers[node]['title']}<br>Authors: {papers[node]['authors']}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            hovertext=node_info,
            textposition="middle center",
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Paper Similarity Network',
                           title_font_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Network shows relationships between papers based on content similarity",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def create_concept_importance_chart(self, concepts: List[Tuple[str, float]]) -> go.Figure:
        """Create bar chart of concept importance"""
        if not concepts:
            return None
        
        concepts_df = pd.DataFrame(concepts, columns=['Concept', 'Importance'])
        concepts_df = concepts_df.head(10)  # Top 10 concepts
        
        fig = px.bar(
            concepts_df,
            x='Importance',
            y='Concept',
            orientation='h',
            title='Key Concepts by Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_paper_timeline(self, papers: List[Dict]) -> go.Figure:
        """Create timeline visualization of papers"""
        if not papers:
            return None
        
        # Extract years (simplified - using random years for demo)
        years = np.random.randint(2020, 2024, len(papers))
        
        timeline_data = []
        for i, (paper, year) in enumerate(zip(papers, years)):
            timeline_data.append({
                'Year': year,
                'Title': paper['title'],
                'Authors': paper['authors'],
                'Score': paper.get('score', 0),
                'Index': i
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig = px.scatter(
            timeline_df,
            x='Year',
            y='Score',
            size='Score',
            hover_data=['Title', 'Authors'],
            title='Paper Timeline by Relevance Score',
            color='Score',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_category_distribution(self, papers: List[Dict]) -> go.Figure:
        """Create pie chart of paper categories"""
        if not papers:
            return None
        
        # Extract categories
        all_categories = []
        for paper in papers:
            categories = paper.get('categories', '').split()
            all_categories.extend(categories)
        
        category_counts = Counter(all_categories)
        
        if not category_counts:
            return None
        
        categories, counts = zip(*category_counts.most_common(10))
        
        fig = px.pie(
            values=counts,
            names=categories,
            title='Distribution of Paper Categories'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_search_results_table(self, papers: List[Dict]) -> pd.DataFrame:
        """Create formatted table for search results"""
        if not papers:
            return pd.DataFrame()
        
        table_data = []
        for paper in papers:
            table_data.append({
                'Title': paper['title'],
                'Authors': paper['authors'],
                'Categories': paper['categories'],
                'Relevance Score': f"{paper.get('score', 0):.3f}",
                'Abstract Preview': paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
            })
        
        return pd.DataFrame(table_data)