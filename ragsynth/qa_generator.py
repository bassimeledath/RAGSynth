import json
import re
from typing import List, Dict
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from .models import Model

class QAGenerator:
    def __init__(self, model: Model):
        self.model = model
        self.infer_method = self.model.get_infer_method()

    @staticmethod
    def prompt(context: str) -> str:
        few_shot_examples = """
        Context: The Great Wall of China is an ancient series of walls and fortifications located in northern China, built around 500 years ago. Estimates of its length vary from 5,000 to 13,000 miles, but an archaeological survey carried out in 2012 by China's State Administration of Cultural Heritage suggested the wall is more than 13,000 miles long.
        {
            "question": "How long is the Great Wall of China according to a 2012 archaeological survey?",
            "answer": "According to a 2012 archaeological survey by China's State Administration of Cultural Heritage, the Great Wall of China is more than 13,000 miles long."
        }
        Context: Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated.
        {
            "question": "What are neural networks designed to do?",
            "answer": "Neural networks are designed to recognize patterns in data."
        }
        """
        
        instructions = """Given the context above, generate a question and its corresponding answer. The question should be answerable using only the information provided in the context. Output your response as a JSON object with "question" and "answer" keys."""
        
        prompt = f"""
                    {few_shot_examples}\n
                    Context: {context}\n
                    {instructions}\n
                    {{'question': '<question>', 'answer': '<answer>'}}
                """
    
        return prompt
    
    @staticmethod
    def _parse_json(result: str) -> str:
        match = re.search(r"```json\n(.*?)\n```", result, re.DOTALL)
        if match:
            return match.group(1)
        else:
            raise ValueError("Could not find JSON object enclosed in triple backticks.")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def _get_predictions(self, prompt: str) -> Dict[str, str]:
        result = self.infer_method(prompt)
        json_result = self._parse_json(result)
        return json.loads(json_result)

    @staticmethod
    def _parse_result(result: Dict[str, str]) -> Dict[str, str]:
        question = result.get("question", "").strip()
        answer = result.get("answer", "").strip()
        if question and answer:
            return {"question": question, "answer": answer}
        else:
            raise ValueError(f"Model output did not contain valid question and answer: {result}")

    def generate_eval_dataset(self, top_k_chunks: List[str]) -> List[Dict[str, str]]:
        eval_dataset = []
        for context in tqdm(top_k_chunks):
            prompt = self.prompt(context)
            result = self._get_predictions(prompt)
            parsed_result = self._parse_result(result)
            eval_dataset.append(parsed_result)
        return eval_dataset