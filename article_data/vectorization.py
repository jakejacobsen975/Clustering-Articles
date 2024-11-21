import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer


def squared_sum(x):
    """Return 3 rounded square rooted value"""
    return round(sqrt(sum([a * a for a in x])), 3)

def cos_similarity(x, y):
    """Return cosine similarity between two lists"""
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = squared_sum(x) * squared_sum(y)
    return round(numerator / float(denominator), 3)
# Save the heatmap as an image
def create_heatmap(similarity, labels, cmap="YlGnBu"):
    df = pd.DataFrame(similarity)
    df.columns = labels
    df.index = labels
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, cmap=cmap, annot=True, fmt=".2f")
    
    # Save the plot as a PNG file
    plt.savefig("similarity_heatmap.png")
    print("Heatmap saved as similarity_heatmap.png")


# Read JSON data with dynamic key handling
data = []
with open("articles/part-00000") as file:
    for line in file:
        article = json.loads(line)
        # Assume there's only one key-value pair per entry
        if article:
            source_name, text_content = next(iter(article.items()))
            data.append((source_name, text_content))

# Continue with the rest of the processing if data exists
if data:
    source_names, texts = zip(*data)

    # Vectorize and calculate similarities as before
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts).toarray()
    similarity_matrix = [[cos_similarity(embeddings[i], embeddings[j]) for j in range(len(embeddings))] for i in range(len(embeddings))]
    
    create_heatmap(similarity_matrix, source_names)
else:
    print("No valid data to process.")

