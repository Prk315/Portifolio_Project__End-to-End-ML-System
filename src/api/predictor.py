"""Prediction logic with confidence-based routing."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline import BaselineModel
from src.models.transformer import TransformerModel
from src.rag.retriever import RAGRetriever
from src.utils.logger import log


class TicketPredictor:
    """Unified predictor with confidence-based routing."""

    def __init__(
        self,
        baseline_model_path: str = None,
        transformer_model_path: str = None,
        rag_retriever_path: str = None,
        confidence_threshold_high: float = 0.85,
        confidence_threshold_low: float = 0.5
    ):
        """
        Initialize predictor.

        Args:
            baseline_model_path: Path to baseline model
            transformer_model_path: Path to transformer model
            rag_retriever_path: Path to RAG retriever
            confidence_threshold_high: Threshold for automatic routing
            confidence_threshold_low: Threshold for human review
        """
        self.confidence_threshold_high = confidence_threshold_high
        self.confidence_threshold_low = confidence_threshold_low

        self.baseline_model = None
        self.transformer_model = None
        self.rag_retriever = None

        self.model_loaded = False
        self.rag_enabled = False

        # Try to load models
        try:
            if baseline_model_path and Path(baseline_model_path).exists():
                log.info(f"Loading baseline model from {baseline_model_path}")
                self.baseline_model = BaselineModel()
                self.baseline_model.load(baseline_model_path)
                self.model_loaded = True
        except Exception as e:
            log.warning(f"Could not load baseline model: {e}")

        try:
            if transformer_model_path and Path(transformer_model_path).exists():
                log.info(f"Loading transformer model from {transformer_model_path}")
                self.transformer_model = TransformerModel()
                self.transformer_model.load(transformer_model_path)
                self.model_loaded = True
        except Exception as e:
            log.warning(f"Could not load transformer model: {e}")

        try:
            if rag_retriever_path and Path(rag_retriever_path).exists():
                log.info(f"Loading RAG retriever from {rag_retriever_path}")
                self.rag_retriever = RAGRetriever()
                self.rag_retriever.load(rag_retriever_path)
                self.rag_enabled = True
        except Exception as e:
            log.warning(f"Could not load RAG retriever: {e}")

        if not self.model_loaded:
            log.warning("No models loaded. Please train models first.")

    def predict(
        self,
        text: str,
        use_rag: bool = True,
        model_type: str = "transformer"
    ) -> Dict:
        """
        Predict category and suggest response.

        Args:
            text: Input text
            use_rag: Whether to use RAG
            model_type: 'baseline' or 'transformer'

        Returns:
            Dictionary with prediction results
        """
        if not self.model_loaded:
            raise ValueError("No models loaded")

        # Select model
        if model_type == "baseline" and self.baseline_model:
            model = self.baseline_model
        elif model_type == "transformer" and self.transformer_model:
            model = self.transformer_model
        else:
            # Fallback
            model = self.transformer_model or self.baseline_model

        # Get predictions
        predictions = model.predict([text])
        probabilities = model.predict_proba([text])

        predicted_category = predictions[0]
        confidence = float(np.max(probabilities[0]))

        # Get all class probabilities
        all_probs = {}
        for i, class_name in enumerate(model.classes):
            all_probs[class_name] = float(probabilities[0][i])

        # Confidence-based routing
        if confidence >= self.confidence_threshold_high:
            routing_decision = "automatic"
        elif confidence >= self.confidence_threshold_low:
            routing_decision = "human_review_suggested"
        else:
            routing_decision = "human_review_required"

        result = {
            "text": text,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "routing_decision": routing_decision,
            "suggested_response": None,
            "similar_tickets": None
        }

        # RAG response suggestion
        if use_rag and self.rag_enabled:
            try:
                rag_result = self.rag_retriever.suggest_response(
                    text,
                    predicted_category
                )
                result["suggested_response"] = rag_result["suggested_response"]
                result["similar_tickets"] = rag_result["similar_tickets"]
            except Exception as e:
                log.error(f"RAG error: {e}")
                # Fallback to template
                result["suggested_response"] = self._get_template_response(predicted_category)

        else:
            # Template response
            result["suggested_response"] = self._get_template_response(predicted_category)

        return result

    def _get_template_response(self, category: str) -> str:
        """Get template response for category."""
        templates = {
            'account_issue': "Thank you for contacting us. I understand you're having trouble with your account. Let me help you resolve this.",
            'billing': "Thank you for reaching out regarding billing. I'd be happy to help clarify your charges.",
            'technical_support': "I apologize for the technical difficulties you're experiencing. Let's work on resolving this together.",
            'product_inquiry': "Thank you for your interest in our products! I'd be happy to help answer your questions.",
            'shipping_delivery': "I understand your concern about your order delivery. Let me check the status for you.",
            'returns_refunds': "I'd be happy to assist you with your return or refund request.",
            'complaint': "I sincerely apologize for your negative experience. Your feedback is very important to us.",
            'general_inquiry': "Thank you for contacting us! I'm here to help."
        }
        return templates.get(category, "Thank you for contacting us. How can I assist you today?")

    def get_categories(self) -> List[str]:
        """Get list of categories."""
        if self.transformer_model:
            return self.transformer_model.classes
        elif self.baseline_model:
            return self.baseline_model.classes
        else:
            return []
