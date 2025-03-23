from typing import List, Any
from langchain.llms import BaseLLM
from langchain.schema import LLMResult
from vllm import LLM, SamplingParams
from pydantic import BaseModel, Field
import logging
import torch

logger = logging.getLogger(__name__)

class VLLMWrapper(BaseLLM, BaseModel):
    model_name: str = Field(..., description="The name of the model to use.")
    llm: Any = Field(None, description="The vLLM model instance.")
    sampling_params: Any = Field(None, description="Sampling parameters for generation.")

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = LLM(model=model_name,
                       enforce_eager=True,
                        device=device, 
                        dtype="float")
        
        self.sampling_params = SamplingParams(temperature=0.3, 
                                              top_p=0.9, 
                                              max_tokens=100)

    def _generate(self, prompts: List[str], **kwargs) -> LLMResult:
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        return LLMResult(generations=[[{"text": text}] for text in generated_texts])

    def _llm_type(self) -> str:
        return "vllm"