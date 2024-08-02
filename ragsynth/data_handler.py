
import os
import re
import random
from typing import List, Iterator

class DataHandler:
    def __init__(self, path: str):
        self.path = path
        self.is_directory = os.path.isdir(path)

    def _get_files(self) -> Iterator[str]:
        if self.is_directory:
            for root, _, files in os.walk(self.path):
                for file in files:
                    yield os.path.join(root, file)
        else:
            yield self.path

    def _read_file_chunks(self, file_path: str) -> Iterator[str]:
        with open(file_path, 'r') as file:
            current_chunk = []
            sentence_count = 0
            for line in file:
                sentences = re.split(r'[.!?]+', line)
                for sentence in sentences:
                    if sentence.strip():
                        current_chunk.append(sentence.strip())
                        sentence_count += 1
                        if sentence_count >= random.randint(20, 30):
                            yield ' '.join(current_chunk)
                            current_chunk = []
                            sentence_count = 0
            if current_chunk:
                yield ' '.join(current_chunk)

    def _chunk_generator(self) -> Iterator[str]:
        for file_path in self._get_files():
            yield from self._read_file_chunks(file_path)

    def get_k_chunks(self, k: int = 5) -> List[str]:
        chunks = []
        for chunk in self._chunk_generator():
            chunks.append(chunk)
            if len(chunks) == k:
                return chunks
        if len(chunks) < k:
            raise ValueError(f"Not enough chunks available. Required: {k}, Available: {len(chunks)}")
        return chunks