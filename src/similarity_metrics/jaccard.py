import nltk
import re


def get_similarity_score(sentence1, sentence2, n=1):
    # doesn't do anything about frequency of words in a sentence
    if n is 1:
        tokens1 = set(re.findall(r'[\w]+', sentence1.lower()))
        tokens2 = set(re.findall(r'[\w]+', sentence2.lower()))
        return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    elif n is 2:
        bigrams1 = set(nltk.bigrams([word for word in sentence1.lower().split()]))
        bigrams2 = set(nltk.bigrams([word for word in sentence2.lower().split()]))
        return len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2))

    elif n is 3:
        trigrams1 = set(nltk.trigrams([word for word in sentence1.lower().split()]))
        trigrams2 = set(nltk.trigrams([word for word in sentence2.lower().split()]))
        return len(trigrams1.intersection(trigrams2)) / len(trigrams1.union(trigrams2))
