import re


def get_similarity_score(sentence1, sentence2):
    # doesn't do anything about frequency of words in a sentence
    tokens1 = set(re.findall(r'[\w]+', sentence1.lower()))
    tokens2 = set(re.findall(r'[\w]+', sentence2.lower()))
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
