import re
from gensim import corpora, models, similarities
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords

def build_model(sentences):

    corpus = []
    for sentence in sentences:
        words = re.findall(r'[\w]+', sentence.lower())
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if w not in stops]
        corpus.append(meaningful_words)

    dictionary = corpora.Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]
    lsi = models.LsiModel(corpus, id2word=dictionary)
    model_filename = get_tmpfile("lsi_gensim.lsi")
    dict_filename = get_tmpfile("lsi_gensim.dict")
    lsi.save(model_filename)
    dictionary.save(dict_filename)

def get_similarity_score(sentence1, sentence2):
    stops = set(stopwords.words("english"))
    sentence1 = re.findall(r'[\w]+', sentence1.lower())
    sentence1 = [w for w in sentence1 if w not in stops]
    sentence2 = re.findall(r'[\w]+', sentence2.lower())
    sentence2 = [w for w in sentence2 if w not in stops]
    dictionary = corpora.Dictionary.load("lsi_gensim.dict")
    lsi = models.LsiModel.load("lsi_gensim.lsi")
    token1 = lsi[dictionary.doc2bow(sentence1)]
    token2 = lsi[dictionary.doc2bow(sentence2)]

    # index = similarities.MatrixSimilarity(lsi[corpus])
    # sims = index[vec_lsi]  # perform a similarity query against the corpus
    # print(sims)
