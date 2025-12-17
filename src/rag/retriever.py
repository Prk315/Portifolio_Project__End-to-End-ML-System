"""Retrieval-Augmented Generation system for response suggestions."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import RAG_CONFIG, MODELS_DIR
from src.utils.logger import log


class RAGRetriever:
    """Retrieval system for finding similar support tickets."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize RAG retriever.

        Args:
            embedding_model: Sentence transformer model name
            top_k: Number of similar documents to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        log.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        self.index = None
        self.documents = None
        self.metadata = None

    def build_index(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        response_column: str = None,
        label_column: str = 'label'
    ):
        """
        Build FAISS index from documents.

        Args:
            df: DataFrame with support tickets
            text_column: Name of text column
            response_column: Name of response column (if available)
            label_column: Name of label column
        """
        log.info(f"Building index from {len(df)} documents...")

        # Store documents
        self.documents = df[text_column].tolist()

        # Store metadata
        self.metadata = []
        for idx, row in df.iterrows():
            meta = {
                'text': row[text_column],
                'label': row[label_column]
            }
            if response_column and response_column in df.columns:
                meta['response'] = row[response_column]
            else:
                # Generate template response based on category
                meta['response'] = self._generate_template_response(row[label_column])

            self.metadata.append(meta)

        # Generate embeddings
        log.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            self.documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings.astype('float32'))

        log.info(f"Index built with {self.index.ntotal} vectors")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        category_filter: str = None
    ) -> List[Dict]:
        """
        Retrieve similar documents for a query.

        Args:
            query: Query text
            top_k: Number of results to return
            category_filter: Filter by category

        Returns:
            List of similar documents with metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if top_k is None:
            top_k = self.top_k

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)

        # Filter and format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < self.similarity_threshold:
                continue

            if category_filter:
                if self.metadata[idx]['label'] != category_filter:
                    continue

            results.append({
                'text': self.metadata[idx]['text'],
                'label': self.metadata[idx]['label'],
                'response': self.metadata[idx]['response'],
                'similarity': float(score)
            })

            if len(results) >= top_k:
                break

        return results

    def suggest_response(
        self,
        query: str,
        predicted_category: str,
        top_k: int = 3
    ) -> Dict:
        """
        Suggest response based on similar tickets.

        Args:
            query: Query text
            predicted_category: Predicted category
            top_k: Number of similar tickets to consider

        Returns:
            Dictionary with suggested response and context
        """
        # Retrieve similar tickets
        similar_tickets = self.retrieve(
            query,
            top_k=top_k,
            category_filter=predicted_category
        )

        if not similar_tickets:
            # Fallback to template
            return {
                'suggested_response': self._generate_template_response(predicted_category),
                'similar_tickets': [],
                'confidence': 0.5
            }

        # Use the most similar ticket's response
        top_response = similar_tickets[0]['response']
        avg_similarity = np.mean([t['similarity'] for t in similar_tickets])

        # Enhance response with context
        enhanced_response = self._enhance_response(
            top_response,
            predicted_category,
            similar_tickets
        )

        return {
            'suggested_response': enhanced_response,
            'similar_tickets': similar_tickets,
            'confidence': float(avg_similarity)
        }

    def _generate_template_response(self, category: str) -> str:
        """Generate template response based on category."""
        templates = {
            'account_issue': "Thank you for contacting us. I understand you're having trouble with your account. Let me help you resolve this. Could you please provide your account email or username so I can look into this for you?",
            'billing': "Thank you for reaching out regarding billing. I'd be happy to help clarify your charges. Could you please provide your order number or account email so I can review your billing details?",
            'technical_support': "I apologize for the technical difficulties you're experiencing. Let's work on resolving this together. Could you provide more details about the error or issue you're encountering?",
            'product_inquiry': "Thank you for your interest in our products! I'd be happy to help answer your questions. What specific information would you like to know?",
            'shipping_delivery': "I understand your concern about your order delivery. Let me check the status for you. Could you please provide your order number?",
            'returns_refunds': "I'd be happy to assist you with your return or refund request. Could you please provide your order number and let me know the reason for the return?",
            'complaint': "I sincerely apologize for your negative experience. Your feedback is very important to us. I want to make this right for you. Could you please provide more details about what happened?",
            'general_inquiry': "Thank you for contacting us! I'm here to help. Could you please provide more details about what you need assistance with?"
        }

        return templates.get(
            category,
            "Thank you for contacting us. How can I assist you today?"
        )

    def _enhance_response(
        self,
        base_response: str,
        category: str,
        similar_tickets: List[Dict]
    ) -> str:
        """Enhance response with additional context."""
        # For now, just return the base response
        # In a full RAG system, you might use an LLM to synthesize a better response
        return base_response

    def save(self, path: str = None):
        """Save index and metadata."""
        if path is None:
            path = MODELS_DIR / "rag_retriever"
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss_index.bin"))

        # Save metadata
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'config': {
                    'embedding_model': self.embedding_model_name,
                    'top_k': self.top_k,
                    'similarity_threshold': self.similarity_threshold
                }
            }, f)

        log.info(f"RAG retriever saved to {path}")

    def load(self, path: str = None):
        """Load index and metadata."""
        if path is None:
            path = MODELS_DIR / "rag_retriever"
        else:
            path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss_index.bin"))

        # Load metadata
        with open(path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)

        self.documents = data['documents']
        self.metadata = data['metadata']

        # Load config
        config = data['config']
        self.embedding_model_name = config['embedding_model']
        self.top_k = config['top_k']
        self.similarity_threshold = config['similarity_threshold']

        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        log.info(f"RAG retriever loaded from {path}")
