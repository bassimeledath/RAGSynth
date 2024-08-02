from typing import Optional
from models import Model, OpenAIModel
from data_handler import DataHandler
from qa_generator import QAGenerator


class EvalDataGenerator:

    def __init__(self,
                 path: str,
                 model: Optional[Model] = None,
                 k: int = 5):
        self.path = path
        self.model = model or OpenAIModel()
        self.k = k

    def generate(self):
        data_handler = DataHandler(self.path)
        top_k_chunks = data_handler.get_k_chunks(self.k)
        qa_generator = QAGenerator(self.model)
        eval_dataset = qa_generator.generate_eval_dataset(top_k_chunks)
        return eval_dataset
    