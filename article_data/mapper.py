#!/bin/python3

import contractions
import itertools
import json
import nltk
from nltk.corpus import stopwords
import re
import sys

shingle_size = 3

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

sw = stopwords.words('english')

def process_content(content):
    text = content.lower().strip()
    tokens = contractions.fix(text).split()
    cleaned_tokens = [re.sub(r'[^\w\s]', ' ', token).strip() for token in tokens if re.sub(r'[^\w\s]', '', token)]
    compacted_tokens = [word for word in cleaned_tokens if word.strip()]
    filtered_tokens = [word for word in compacted_tokens if word not in sw]
    return ' '.join(filtered_tokens)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        articles = json.loads(line)
    except json.JSONDecodeError:
        sys.stderr.write(f"Error decoding JSON: {line}\n")
        continue
        
    for article in articles:
        if 'url' in article:
            url = article['url']
            content = article['content']
        elif 'link' in article:
            url = article['link']
            content = article['content']
        
        else:
            continue
        
        cleaned_content = process_content(content)
        print(f"{url}:::{cleaned_content}")
