from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

tf_vectorizer = None

def get_similarity_score(sentence1, sentence2):
    global tf_vectorizer
    if tf_vectorizer is None:
        tf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf = tf_vectorizer.fit_transform([sentence1, sentence2])
    return ((tfidf * tfidf.T).A)[0, 1]

'''
# takes too long if the tfidf matrix is built for all sentences 
def build_model(sentences):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tf_vectorizer = tf.fit(sentences)
    pickle.dump(tf_vectorizer, open("tf.pickle", 'wb'))
    
    
def get_similarity_score(sentence1, sentence2):
    global tf_vectorizer
    if tf_vectorizer is None:
        if os.path.exists("tf.pickle"):
            tf_vectorizer = pickle.load(open("tf.pickle", 'rb'))
    tfidf = tf_vectorizer.transform([sentence1, sentence2])
    return ((tfidf * tfidf.T).A)[0, 1]
'''