import re


def get_similarity_score(sentence1, sentence2):
    # doesn't do anything about frequency of words in a sentence
    # different from jaccard in the sense that it captures both mutual presence and mutual absence
    # but within 2 sentences there is no mutual absence case essentially.. missing something?
    tokens1 = set(re.findall(r'[\w]+', sentence1.lower()))
    tokens2 = set(re.findall(r'[\w]+', sentence2.lower()))
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
