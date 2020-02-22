import re
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords


def build_model(sentences):
    corpus = []
    for sentence in sentences:
        words = re.findall(r'[\w]+', sentence.lower())
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if w not in stops]
        corpus.append(meaningful_words)

    model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=4)
    filename = get_tmpfile("sdp_word2vec.kv")
    model.save(filename)
    model.wv.save_word2vec_format('sdp_word2vec.bin', binary=True)


def get_similarity_score(sentence1, sentence2):
    sentence1 = re.findall(r'[\w]+', sentence1.lower())
    sentence2 = re.findall(r'[\w]+', sentence2.lower())
    model = KeyedVectors.load_word2vec_format('sdp_word2vec.bin', binary=True)
    return model.wv.n_similarity(sentence1, sentence2)
