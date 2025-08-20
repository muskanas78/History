# Foundational Understanding

# What is Sentence Similarity Check?
# The similarity check using sentence-transformers is about measuring how semantically similar two sentences are, not just word overlap, but meaning.

# Example:
# Sentence A: I love dogs.
# Sentence B: Dogs are amazing!
# Even though the wording is different, the meaning is similar : Cosine similarity will be high (e.g., ~0.8+)

# But:
# Sentence A: I love dogs.
# Sentence B: I hate dogs.
# Despite similar words, the meanings are opposites → Cosine similarity will be much lower.

# What is Cosine Similarity?
# Think of sentence embeddings as points in space, like this:

# Embedding A : [0.1, 0.3, 0.9, ...]
# Embedding B : [0.2, 0.1, 0.7, ...]

# Cosine similarity measures the angle between them:
# 1.0 : exactly same direction (very similar)
# 0.0 : 90° angle (unrelated)
# -1.0 : opposite direction (completely opposite)

# Formula:

# cosine_similarity = (A.B) / ||A||.||B||

# Why We See Only 0 to 1?
# Because the sentence-transformers model outputs normalized vectors (unit length), and they are mostly trained to keep outputs in the positive quadrant, so values typically fall between 0 and 1.

# Although, cosine similarity mathematically ranges from -1 to 1. You just don't see -1 much in practice with these embeddings unless you deliberately choose opposite sentiment.

# Important!
# Cosine similarity ranges from -1 to 1. A cosine similarity of 1 means the vectors are perfectly aligned (no angle between them), indicating maximum similarity, whereas a value of -1 implies they are diametrically opposite, reflecting maximum dissimilarity. Values near zero indicate orthogonality.


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# embedding multiple sentences
sentences = [
    "Technology in China is evolving.",
    "I enjoy reading books.",
    "Machine learning lets computers learn from data."
]

embeddings = model.encode(sentences)

print("Sentence Embeddings:")
print("-" * 40)
i = 1
for sentence in sentences:
    print(f"Sentence {i}: {sentence}")
    i += 1

# show sentence embeddings
print("\nGenerated Embedding Matrix:")
print("-" * 40)
np.set_printoptions(precision=6, suppress=True, edgeitems=2, linewidth=120)
print(embeddings)

print(f"\nEmbedding size = {embeddings.shape} "
      f"({embeddings.shape[0]} sentences × {embeddings.shape[1]} features)\n")


# compare two similar sentences
sentence1 = "I need help with my homework."
sentence2 = "Can you assist me with my homework?"

embedding1 = model.encode([sentence1])
embedding2 = model.encode([sentence2])

similarity = cosine_similarity(embedding1, embedding2)[0][0]

print("Single Pair Similarity:")
print("-" * 40)
print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Cosine Similarity: {similarity:.4f}\n")


# pairwise similarity comparison
pairs = [
    ("I am happy.", "I am joyful."),
    ("I am happy.", "I am sad."),
    ("The cat sleeps a lot.", "The sky is blue."),
    ("I hate this.", "I love this.")
]

print("Pairwise Cosine Similarity:")
print("-" * 40)

for sent1, sent2 in pairs:
    emb1 = model.encode([sent1])
    emb2 = model.encode([sent2])
    sim = cosine_similarity(emb1, emb2)[0][0]
    print(f"Sentence 1: {sent1}")
    print(f"Sentence 2: {sent2}")
    print(f"Cosine Similarity: {sim:.4f}\n")