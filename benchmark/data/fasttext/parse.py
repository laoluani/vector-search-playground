import io

import numpy as np
import tqdm

from pathlib import Path

def load_vectors(path: str, size: int) -> tuple[list[str], np.ndarray]:
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    size = min(n, size)

    vectors = np.empty((size, d), dtype=np.float32)
    words = []

    for i, line in tqdm.tqdm(enumerate(fin)):
        if i == size: 
            break

        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        vectors[i] = list(map(float, tokens[1:]))
    
    return words, vectors

embeddings_file_name = "benchmark/data/fasttext/embeddings.npy"
words_file_name = "benchmark/data/fasttext/words.txt"
fasttext_vectors_path = "benchmark/data/fasttext/fasttext.vec"
size = 100

words, vectors = load_vectors(fasttext_vectors_path, 100)

with Path(words_file_name).open("w") as f:
    for word in words:
        f.write(word + "\n")
    
np.save(embeddings_file_name, vectors)
