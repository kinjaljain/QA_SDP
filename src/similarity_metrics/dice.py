import re


def get_similarity_score(sentence1, sentence2):
    sentence1 = re.findall(r'[\w]+', sentence1.lower())
    sentence2 = re.findall(r'[\w]+', sentence2.lower())
    len_sentence1 = len(sentence1)
    len_sentence2 = len(sentence2)

    if not len_sentence1 or not len_sentence2:
        return 0.0

    if sentence1 == sentence2:
        return 1.0

    if len_sentence1 == 1 or len_sentence2 == 1:
        return 0.0

    sentence1_bigrams = [sentence1[i:i + 2] for i in range(len_sentence1 - 1)]
    sentence2_bigrams = [sentence2[i:i + 2] for i in range(len_sentence2 - 1)]

    sentence1_bigrams.sort()
    sentence2_bigrams.sort()

    len_sentence1 = len(sentence1_bigrams)
    len_sentence2 = len(sentence2_bigrams)

    matches = i = j = 0
    while i < len_sentence1 and j < len_sentence2:
        if sentence1_bigrams[i] == sentence2_bigrams[j]:
            matches += 2
            i += 1
            j += 1
        elif sentence1_bigrams[i] < sentence2_bigrams[j]:
            i += 1
        else:
            j += 1

    return (2 * matches) / (len_sentence1 + len_sentence2)
