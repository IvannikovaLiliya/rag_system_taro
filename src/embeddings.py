"""Vector embeddings and database functionality."""
import logging
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector embeddings and database operations."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.vector_db = None
    
    def create_vector_db(self, documents: List):
        """Create vector database from documents."""
        try:
            logger.info("Creating vector database")
            self.embeddings = self.encoder.encode(documents)
            self.vector_db = faiss.IndexFlatL2(self.embeddings.shape[1]) 
            self.vector_db.add(np.array(self.embeddings))
            return self.vector_db
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise