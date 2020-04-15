import json
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm
# from nltk.corpus import words
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
from dataloaders.load_and_parse import load_all
# from nltk.stem import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
import enchant

d = enchant.Dict("en_US")

lemmatizer = WordNetLemmatizer()



from langdetect import detect, detect_langs, DetectorFactory

DetectorFactory.seed = 0

from similarity_metrics.jaccard import get_similarity_score as j
# from similarity_metrics.dice import get_similarity_score as dc
# from similarity_metrics.word2vec import get_similarity_score as w
import similarity_metrics.tfidf as t
from similarity_metrics.word2vec import build_model
import os.path
import pickle
import random
import numpy as np

DATA_ROOT = '../../data'

dataset = load_all(DATA_ROOT)

first_sentences = set()

ratio_thresh = 0.4

f = open("meaningless_ratios_"+str(ratio_thresh)+"_modularized.txt", "w")
ff = open("alphabet_ratios_"+str(ratio_thresh)+"_modularized.txt", "w")

import re


# check if the data passed is numeric
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Contain more than 20\% of alphabetical single characters
# [7 false, 4 true]
# 4/11 >= 70%

def filter_meaningless(ref_article, i, jargon):
    line = ref_article[i]
    line.replace(";", " ").replace(",", " ")
    # Table caption
    if re.search(r'Table [0-9]+: ', line):
        start_idx = line.index("Table ")
        line = line[start_idx:]

    # Figure caption
    elif re.search(r'Figure [0-9]+: ', line):
        start_idx = line.index("Figure ")
        line = line[start_idx:]

    words = np.array(nltk.word_tokenize(ref_article[i]))
    num_singles = np.array([len(word) < 2 for word in words])
    count = words[num_singles].shape[0]
    ratio = count / len(words)

    if ratio > 0.53:
        # ff.write(ref_article[i])
        return False, None

    is_valid = []
    check = words
    for word in words:
        if d.check(word) or word in jargon:
            is_valid.append(True)
        else:
            is_valid.append(False)
    is_valid = np.array(is_valid)
    check = np.array(check)
    count = check[is_valid].shape[0]
    ratio = count / check.shape[0]
    print(ratio, line)
    if ratio <= 0.72:
        print("Ignoring Meaningless Sentence:", line)
        # f.write(ref_article[i])
        return False, None
    return True, line


# Contain less than 10\% of tokens whose lemmas correspond to known words (compare against WordNet lexicon)
# Contain more than 30\% of tokens whose lemmas do not correspond to known words (compare against WordNet lexicon)

#  Contain less than 70\% of tokens whose lemmas correspond to known words (compare against WordNet lexicon)
def prepare(dataset):
    result = []
    num_articles = 0
    jargon = open("jargon.txt", 'r').readlines()
    jargon = [word[:-1] for word in jargon]
    for data in tqdm(dataset):
        ref_article = data.ref.sentences
        try:
            ref_title = ref_article[0]
        except:
            ref_title = "No title"
        ref_article = data.ref.sentences
        if ref_article[1] in first_sentences:
            continue
        print("At article", num_articles)
        num_articles += 1
        first_sentences.add(ref_article[1])

        for i in range(1, len(ref_article)):
            is_meaningful, line = filter_meaningless(ref_article, i, jargon)
            if is_meaningful:
                result.append(line)
            else:
                f.write(ref_article[i])
    print("total number of articles: ", num_articles)
    return result

result = prepare(dataset)
# print(len(result))
# print(len(result)//4, len(result)//2, 3*len(result)//4)
# print(len(result)//2 - len(result)//4,  3*len(result)//4 - len(result)//2, len(result) - 3*len(result)//4)

f.close()

with open("all_ref_article_sentences_modularized.txt", "w") as f1:
    for sentence in result:
        f1.write(sentence + "\n")
# with open("prepared_data_nsp_actual_1st_qtr.json", "w") as f:
#     json.dump(result[:len(result)//4], f)
#
# with open("prepared_data_nsp_actual_2nd_qtr.json", "w") as f:
#     json.dump(result[len(result)//4:len(result)//2], f)
#
# with open("prepared_data_nsp_actual_3rd_qtr.json", "w") as f:
#     json.dump(result[len(result)//2:3*len(result)//4], f)
#
# with open("prepared_data_nsp_actual_4th_qtr.json", "w") as f:
#     json.dump(result[3*len(result)//4:], f)

