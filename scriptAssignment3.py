# Implement a Python script to calculate self-attention for a given input sequence. 
# The script should compute the self-attention mechanism without using any specialized libraries for neural networks 
# and should be optimized to run efficiently on the CPU.

# Define a Python function that computes self-attention for a given input sequence. (5 Points)
# Implement the self-attention mechanism as described in the lecture, consisting of key, query, and value transformations. (5 Points)
# Ensure that the script is efficient and scaled-down, suitable for running on a CPU. Test the script with various input sequences of
# different lengths to ensure correctness and efficiency. (5 Points)
# Document your code with comments explaining each major step of the self-attention process. (5 Points)

import numpy as np
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# pre-selected sample input array
sample_input_array = [
    {"1": "This is a test input."},
    {"2": "Chess, not checkers."},
    {"3": "Algorithms make the world go round."},
    {"4": "Financial Technology is my graduate program."},
    {"5": "UCF is in Florida."},
    {"6": "I am from Florida. The year is 2024 and I am a graduate student. This team is Team Number 1."},
    {"7": "The only thing we have to fear is fear itself."},
    {"8": "I think, therefore I am."},
    {"9": "To be, or not to be, that is the question."},
    {"10": "A journey of a thousand miles begins with a single step."},
    {"11": "I have a dream."},
    {"12": "Knowledge is power."},
    {"13": "Time is money."},
    {"14": "The pen is mightier than the sword."},
    {"15": "An eye for an eye only ends up making the whole world blind."},
    {"16": "In the end, we will remember not the words of our enemies, but the silence of our friends."}
]

array_input_number = np.random.randint(1, 16)

text_to_tokenize = sample_input_array[array_input_number - 1][str(array_input_number)]

def tokenize(text_to_tokenize):
    # Tokenizes text into words, separating out punctuation since this is a common NLP task
    tokens = re.findall(r'\b\w+\b|\S', text_to_tokenize.lower())
    return tokens

tokens = tokenize(text_to_tokenize)

def count_vectorize(tokens):
    word_counts = {}
    for word in tokens:
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1
    return word_counts

vector = count_vectorize(tokens)

print('Tokenization:', tokens)
print('Count Vectorization:', vector)

def one_hot_encode(tokens):
    # Create a list of unique words
    unique_words = list(set(tokens))

    # Create a dictionary that maps each word to a unique index
    word_to_index = {word: index for index, word in enumerate(unique_words)}

    # Create a one-hot encoding for each word
    one_hot_vectors = np.eye(len(unique_words))

    # Map each word in the input sequence to its one-hot encoding
    embeddings = [one_hot_vectors[word_to_index[word]] for word in tokens]

    return embeddings

def self_attention(input_sequence, embed_dim):
    # Save the current state of the numpy random number generator
    rng_state = np.random.get_state()

    # Set the seed and initialize weight matrices
    np.random.seed(0)
    WQ = np.random.rand(embed_dim, embed_dim)
    WK = np.random.rand(embed_dim, embed_dim)
    WV = np.random.rand(embed_dim, embed_dim)

    # Restore the previous state of the numpy random number generator
    np.random.set_state(rng_state)

    # Calculate Q, K, V
    Q = np.dot(input_sequence, WQ)
    K = np.dot(input_sequence, WK)
    V = np.dot(input_sequence, WV)

    # Calculate attention scores
    attention_scores = np.dot(Q, K.T) / np.sqrt(embed_dim)
    attention_scores = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)

    # Calculate output
    output = np.dot(attention_scores, V)

    return output,attention_scores

# One-hot encode the tokens
embeddings = one_hot_encode(tokens)

# Set embed_dim to the length of the unique words list
embed_dim = len(set(tokens))

# Call the self_attention function
output, attention_scores = self_attention(embeddings, embed_dim)

print('Self-Attention Output:\n', output)

# Get the unique words
unique_words = list(set(tokens))

# Visualize the attention scores
plt.figure(figsize=(10, 10))
plt.imshow(attention_scores, cmap='hot', interpolation='nearest')

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label('Attention Scores')

# Set the tick labels to be the words
plt.xticks(np.arange(len(unique_words)), unique_words, rotation=90)
plt.yticks(np.arange(len(unique_words)), unique_words)

# Add title and axis labels
plt.title('Self-Attention Heatmap')
plt.xlabel('Keys')
plt.ylabel('Queries')

plt.show()

# Call the self_attention function
output, attention_scores = self_attention(embeddings, embed_dim)

# Apply PCA
pca = PCA(n_components=2)  # reduce to 2 dimensions
pca_output = pca.fit_transform(output)

# Print the PCA output
print('PCA Output:\n', pca_output)

# Plot the PCA output
plt.figure(figsize=(10, 10))
plt.scatter(pca_output[:, 0], pca_output[:, 1])

for i, word in enumerate(unique_words):
    plt.annotate(word, (pca_output[i, 0], pca_output[i, 1]))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Self-Attention Output')
plt.show()