"""Vector embeddings and database functionality."""
import logging
from typing import List, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Кастомный FaissRetriever
class FaissRetriever(BaseRetriever, BaseModel):
    faiss_index: Any = Field(..., description="Faiss index for similarity search.")
    embeddings: Any = Field(..., description="Embeddings of the documents.")
    documents: List[str] = Field(..., description="List of documents.")
    encoder: Any = Field(..., description="Encoder for converting text to embeddings.")

    def __init__(self, **data):
        # Вызываем конструктор BaseModel
        super().__init__(**data)

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Кодируем запрос в вектор
        query_embedding = self.encoder.encode([query])[0]
        query_embedding = np.array(query_embedding).astype('float32')

        # Поиск в faiss
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k=5)

        # Возвращаем документы
        return [Document(page_content=self.documents[i]) for i in indices[0]]

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Асинхронная версия (если нужно)
        return self.get_relevant_documents(query, **kwargs)

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

            # Создаем обертку для faiss
            faiss_retriever = FaissRetriever(
                faiss_index=self.vector_db,
                embeddings=self.embeddings,
                documents=documents,
                encoder=self.encoder
            )

            return faiss_retriever
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise