import random

import numpy as np
from tqdm import tqdm


def load_glove_file(glove_file):
    print("Loading Glove Model")
    f = open(glove_file, 'r', encoding='utf8')
    model = {}
    for line in tqdm(f):
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


# Replace 'path/to/glove.txt' with the path to your GloVe file
glove_path = r'glove.6B.50d.txt'
glove_embeddings = load_glove_file(glove_path)
print(f"Loaded {len(glove_embeddings)} word vectors.")


# Load GloVe embeddings
def get_glove_embedding(word):
    try:
        return glove_embeddings[word]
    except KeyError:
        return np.array([random.randint(0, 600)] * 50)