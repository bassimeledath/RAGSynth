from typing import Optional
import os
from .models import Model, OpenAIModel
from .data_handler import DataHandler
from .qa_generator import QAGenerator


class EvalDataGenerator:

    def __init__(self,
                 path: str,
                 model: Optional[Model] = None):
        self.path = path

        if model is None:
            if "OPENAI_API_KEY" not in os.environ:
                raise Exception("OPENAI_API_KEY is not set in the environment variables. "
                                         "Please set it or provide a custom model.")
            self.model = OpenAIModel()
        else:
            self.model = model
        self.data = DataHandler(self.path)

    def generate(self, k: int = 5):
        top_k_chunks = self.data.get_k_chunks(k)
        qa_generator = QAGenerator(self.model)
        eval_dataset = qa_generator.generate_eval_dataset(top_k_chunks)
        return eval_dataset
    