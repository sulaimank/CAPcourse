import numpy as np
import re

# pre-selected sample input array
sample_input_array = [
    {"1": "This is a test input."},
    {"2": "Chess, not checkers."},
    {"3": "Algorithms make the world go round."},
    {"4": "Financial Technology is my graduate program."},
    {"5": "UCF is in Florida."}
]

array_input_number = np.random.randint(1, 6)

text_to_tokenize = sample_input_array[array_input_number - 1][str(array_input_number)]

def tokenize(text_to_tokenize):
    # Tokenizes text into words, separating out punctuation since this is a common NLP task
    tokens = re.findall(r'\b\w+\b|\S', text_to_tokenize.lower())
    return tokens

tokens = tokenize(text_to_tokenize)
print('Tokenization:', tokens)