"""LLM configuration and setup."""
import logging
from vllm import LLM, SamplingParams
import torch
import re
from src.embeddings import VectorStore

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM configuration and prompts."""
    
    def __init__(self, k = 5,
                 model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.llm = LLM(model=model_name, 
                        enforce_eager=True,
                        device=self.device, 
                        dtype="float")
        # Параметры генерации
        self.sampling_params = SamplingParams(temperature=0.7, 
                                                top_p=0.9, 
                                                max_tokens=250)
        
        self.k = k
        self.query =  None
    
    def postprocess_text(self, text):
        """Очистка текста от артефактов."""
        def clean_text(text):
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[\n\t]', ' ', text)
            return text.strip()
        
        def remove_incomplete_sentences(text):
            sentences = text.split('. ')
            if len(sentences) > 1 and not text.endswith('.'):
                text = '. '.join(sentences[:-1]) + '.'
            return text
        
        def remove_duplicates(text):
            sentences = text.split('. ')
            unique_sentences = []
            for sentence in sentences:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
            return '. '.join(unique_sentences)
        
        def remove_special_tokens(text):
            special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
            for token in special_tokens:
                text = text.replace(token, '')
            return text
        
        text = clean_text(text)
        text = remove_incomplete_sentences(text)
        text = remove_duplicates(text)
        text = remove_special_tokens(text)
        return text
    
    def extract_support_text(self, text):

        match = re.search(r'Support:(.*?)##### Input:', text, re.DOTALL)
        if match:
            return match.group(1).strip() 
        return text 
    
    def get_rag_prompt(self, documents, query):
        """Get RAG prompt template."""

        self.query = query
        self.template = """You are an AI language model assistant. Your task is to generate five
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}"""
        
        self.query_embedding = VectorStore().encoder.encode([self.query])
        _, self.indices = VectorStore().vector_db.search(self.query_embedding, self.k)
        self.relevant_chunks = [documents[i] for i in self.indices[0]]
        
        prompt = self.template.format(context=self.template, question=self.query)
        self.outputs = self.llm.generate([prompt], self.sampling_params)
    
        response = self.outputs[0].outputs[0].text
        response = self.postprocess_text(response)

        return response
    
    def get_llm_prompt(self, query):
        """Get llm prompt template."""

        self.query = [query]
        self.outputs = self.llm.generate(self.query, self.sampling_params)

        response = self.outputs[0].outputs[0].text
        response = self.postprocess_text(response)
        response = self.extract_support_text(response)

        return response