from src.model.llm import LLM
from src.model.llm_lp import LLMlp


load_model = {
    'llm': LLM,
}
from src.model.llm_lp import LLMlp


load_model_lp = {
    'llm': LLMlp,
}



llama_model_path = {
    'llava_1.5': '/scratch/ys6310/llava-1.5-13b-hf',
    'llava_1.6': '/scratch/ys6310/llava-v1.6-mistral-7b-hf',
    "Llama-3.1-8B":'/scratch/ys6310/Llama-3.1-8B',
    "2-7b":"/scratch/ys6310/Llama-2-7b-hf",
    "2-13b":"/scratch/ys6310/Llama-2-13b-hf"
}
