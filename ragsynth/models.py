from typing import Callable
from openai import OpenAI

class Model:
    def get_infer_method(self) -> Callable:
        for infer_method_names in ("predict", "infer", "forward"):
            infer_method = getattr(self, infer_method_names, None)
            if infer_method:
                return infer_method
        raise ValueError(
            f"Model {self} does not have a predict, infer, or forward method."
        )


class OpenAIModel(Model):
    def __init__(self, model_name: str = "gpt-4o"):
        self.openai = OpenAI()
        self.model_name = model_name

    def predict(self, prompt):
        completion = self.openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content