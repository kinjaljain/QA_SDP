from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
import pickle
import os
from rank_bm25 import BM25Okapi
from summa import summarizer

import Levenshtein as lvstn
from sematch.semantic.similarity import WordNetSimilarity
from tqdm import tqdm_notebook as tqdm
from nltk.stem import PorterStemmer


from tqdm import tqdm as tqdm

Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')

import spacy
import pytextrank


nlp = spacy.load("en_core_web_sm")
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

import nltk
nltk.download('stopwords')


from rake_nltk import Rake
from nltk.corpus import stopwords
import re

stop = set(stopwords.words('english'))

rakey = Rake(max_length=1, ranking_metric=0)


def encode(sentence):
    return sentence


# latest tf-idf


global tf_vectorizer

root_path = "/Users/kai/PycharmProjects/QA_SDP2/src/dataloaders/"


def build_model(cite_paper, ref_paper, n=1):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, n), min_df=0, stop_words='english')
    sentences = list(cite_paper.sentences.values())
    sentences.extend(list(ref_paper.sentences.values()))
    tf_vectorizer = tf.fit(sentences)
    pickle.dump(tf_vectorizer, open(
        "{}tf-idf_cache/tf_cite_{}_ref_{}_n_{}.pickle".format(root_path, cite_paper.id,
                                                              ref_paper.id, n), 'wb'))
    return tf_vectorizer


def encode(sentence):
    return sentence


def get_similarity_score(cite_paper, ref_paper, sentence1, sentence2, n=1, kernel="none"):
    if os.path.exists(
            "{}tf-idf_cache/tf_cite_{}_ref_{}_n_{}.pickle".format(root_path, cite_paper.id,
                                                                  ref_paper.id, n)):
        tf_vectorizer = pickle.load(open(
            "{}tf-idf_cache/tf_cite_{}_ref_{}_n_{}.pickle".format(root_path, cite_paper.id,
                                                                  ref_paper.id, n),
            'rb'))
    else:
        tf_vectorizer = build_model(cite_paper, ref_paper, n)

    tfidf_1 = tf_vectorizer.transform([sentence1])
    tfidf_2 = tf_vectorizer.transform([sentence2])
    if kernel == "none":
        return cosine_similarity(tfidf_1, tfidf_2).item()
    elif kernel == "linear":
        return linear_kernel(tfidf_1, tfidf_2).item()
    elif kernel == "poly_2":
        return polynomial_kernel(tfidf_1, tfidf_2, 2).item()
    elif kernel == "poly_3":
        return polynomial_kernel(tfidf_1, tfidf_2, 3).item()
    elif kernel == "rbf":
        return rbf_kernel(tfidf_1, tfidf_2).item()




wns = WordNetSimilarity()
ps = PorterStemmer()

def get_similarity_score(sentence1, sentence2):
    # doesn't do anything about frequency of words in a sentence
    tokens1 = set(re.findall(r'[\w]+', sentence1.lower()))
    tokens2 = set(re.findall(r'[\w]+', sentence2.lower()))
    # print(sentence1)




    rakey.extract_keywords_from_sentences(sentence1.replace("-", " ").lower().split())

    keys1 = rakey.get_ranked_phrases()
    # keys1 = [x.split() for x in keys1]
    # print(keys1)
    # keys1 = [item for sublist in keys1 for item in sublist]
    # print(keys1)

    rakey.extract_keywords_from_sentences(sentence2.replace("-", " ").lower().split())
    keys2 = rakey.get_ranked_phrases()
    # keys2 = [x.split() for x in keys2]
    # keys2 = [item for sublist in keys2 for item in sublist]

    # keys = set(keys1) and set(keys2)
    keys2 = set(keys2)
    # for word in sentence1.replace("-", " ").lower().split():
    #     if word in keys2:
    #         return 1
    # return 0
    # keys1 = tokens1
    # keys2 = tokens2

    tokens1 = set(keys1) - stop  # - set(ref_keys)
    tokens2 = set(keys2) - stop  # - set(cite_keys)
    # tokens1 = set([ps.stem(token) for token in tokens1])
    # tokens2 = set([ps.stem(token) for token in tokens2])
    # v = 0
    # for token1 in tokens1:
    #     max_sim = 0
    #     for token2 in tokens2:
    #         max_sim = max(max_sim, wns.word_similarity(token1, token2, 'li'))
    #     v += max_sim
    # print(tokens1)
    # print(tokens2)
    # wefwefwef
    # tokens1 = tokens1 - stop
    # tokens2 = tokens2 - stop
    # return v / len(tokens1.union(tokens2))
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('bert-base-nli-mean-tokens')
#



#
# def encode(sentence):
#   return model.encode([sentence])
#
# def get_similarity_score(sentence1, sentence2, kernel = "poly_2"):
#     return cosine_similarity(sentence1, sentence2)
#   #return polynomial_kernel(sentence1, sentence2, 2).item()



import json
accuracy = 0
total = 0
empty_citations = 0
empty_references = 0
import pickle

with open("%sprocessed-data-2019-clean.pkl" % root_path, 'rb') as f:
  dataset = pickle.load(f)

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
  dataset2 = pickle.load(f)

# dataset =  dataset2 + dataset
# dataset = dataset2




ref_counts = {}
for data in dataset2:
    ctr = len(data.offsets.ref)
    if ctr == 9:
        print("ok")
    ref_counts[ctr] = ref_counts.get(ctr, 0) + 1
print(ref_counts)





dataset = dataset2
# dataset = [x for x in dataset if x.ref.id in {"C00-2123",
# "C04-1089",
# "I05-5011",
# "J96-3004",
# "N06-2049",
# "P05-1004",
# "P05-1053"
# "P98-1046",
# "P98-2143",
# "W03-0410"}]



articles = [x[1] for x in dataset]

with open("candall.json") as f:
    cands = json.load(f)

cands = {}


def bias_intro(s, ref_article, num_cites, id2score):
    sid = s[0]
    present = ("intro" in ref_article.sections[sid].lower() or "abstract" in ref_article.sections[
        sid].lower() or "concl" in ref_article.sections[sid].lower() or "summ" in ref_article.sections[sid].lower())
    if not present:
        score = s[1]
    else:
        score = s[1]+min(0.05, (0.01)*(num_cites-1)) + 0.20 * id2score[sid]
        # score = s[1] + (0.02) * (num_cites - 1) + 0.3 * id2score[sid]
    return [sid, score]


memo = {}
def get_id2_score(data):
    global memo
    if data.ref.id in memo:
        return memo[data.ref.id]
    sents = data.ref.sentences.values()
    sents_merged = '\n'.join(sents)
    sent_scores = {x[0]: x[1] for x in summarizer.summarize(sents_merged, ratio=1, split=True, scores=True)}
    id2score = {}
    for id in data.ref.sentences:
        sentence = data.ref.sentences[id]
        if sentence in sent_scores:
            id2score[id] = sent_scores[sentence]
        else:
            closest_sent = sorted([x for x in sent_scores.keys()], key=lambda x: lvstn.distance(x, sentence))[0]
            id2score[id] = sent_scores[closest_sent]
    memo[data.ref.id] = id2score
    return id2score


def map_data(data):
    tp = 0
    fp = 0
    fn = 0
    ref_article = data.ref
    to_break = True
    citing_article = data.cite
    offsets = data.offsets
    citing_sentence_ids = offsets.cite
    true_ref_sentences = offsets.ref
    true_ref_sentence_ids = offsets.ref
    if citing_article.sentences:
      # raku.extract_keywords_from_sentences([citing_article.sentences[x] for x in citing_article.sentences])
      # cite_keys = raku.get_ranked_phrases()[:0]
      # raku.extract_keywords_from_sentences([ref_article.sentences[x] for x in ref_article.sentences])
      # ref_keys = raku.get_ranked_phrases()[:0]

      new_ids = [c for c in citing_sentence_ids]
      for c in citing_sentence_ids:
            # If additional context is reqd
            to_add  = 0
            extra = range(max(1, c-to_add), c)
            new_ids.extend(extra)
            extra = range(c+1, min(len(citing_article.sentences), c + to_add+1))
            new_ids.extend(extra)
      citing_sentence_ids = new_ids
      complete_citing_sentence = " ".join([citing_article.sentences[c] for c in citing_sentence_ids])
      if has_multiple_cites(complete_citing_sentence) > 2:
          print("---------------------")
          print(complete_citing_sentence)
          print([data.ref.sentences[x] for x in data.offsets.ref])
          print("--------------")

      ref_vs = [(x[0],encode(x[1])) for i, x in enumerate(ref_article.sentences.items()) if ( has_multiple_cites(x[1]) < 2)] # or  "intro" in ref_article.sections[x[0]].lower() or "abstract" in ref_article.sections[x[0]].lower() or "concl" in ref_article.sections[x[0]].lower() or "summ" in ref_article.sections[x[0]].lower())]
      similarity_score = {}

      # key_word_refs = []
      # for ref in ref_vs:
      #     rakey.extract_keywords_from_sentences(ref[1].replace("-", " ").lower().split())
      #     key_word_refs.append(rakey.get_ranked_phrases())
      #
      # tokenized_corpus = [key_word_refs[i] for i  in range(len(ref_vs))]
      # bm25 = BM25Okapi(tokenized_corpus)
      # tokenized_query = complete_citing_sentence.split(' ')
      # rakey.extract_keywords_from_sentences(tokenized_query)
      #
      # doc_scores = bm25.get_scores(rakey.get_ranked_phrases())
      # for idx, ref_v in enumerate(ref_vs):
      #     ref_id = ref_v[0]
      #     similarity_score[ref_id] = max(similarity_score.get(ref_id, 0), doc_scores[idx])

      for i in range(1):#c in citing_sentence_ids:
        complete_citing_sentence = ' '.join([citing_article.sentences[x] for x in citing_sentence_ids])
        #complete_citing_sentence = citing_article.sentences[c]
        #complete_citing_sentence = re.sub('\((.+?)\)', '', complete_citing_sentence)
        complete_citing_sentence = encode(complete_citing_sentence)
        if str((ref_article.id, citing_article.id, str(offsets.marker))) in cands:
            top_n = cands[str((ref_article.id, citing_article.id, str(offsets.marker)))]
            similarity_score = {x:1 for x in top_n}
        else:
            #print("Fetching")
            for ref_id, ref_sentence in ref_vs: #(ref_article.sentences.items()):
                try:
                    # similarity_score[ref_id] = max(similarity_score.get(ref_id, 0) , get_similarity_score(citing_article, ref_article, ref_sentence, complete_citing_sentence, 1, "poly_3"))
                    similarity_score[ref_id] = max(similarity_score.get(ref_id, 0) , get_similarity_score(ref_sentence, complete_citing_sentence))
                except Exception as e:
                    #print(e)
                    pass
      if similarity_score:
          id2score = get_id2_score(data)
          similarity_score = {x: similarity_score[x]  + 0.20 * id2score[x] for x in similarity_score} # 0.3
          num_cites = has_multiple_cites(complete_citing_sentence)
          sorted_similarity_score = sorted(similarity_score.items(), key=lambda item: -item[1])
          top_n = [s for s in sorted_similarity_score]
          top_n = [bias_intro(s, ref_article, num_cites, id2score) for s in top_n]

          top_n = [s for s in top_n if s[1]>0.15] # 0.18

          top_n = {x[0]:x[1] for x in top_n[:5]}
          # print("--citing--")
          # print(complete_citing_sentence)
          # print("--candidates--")
          # print('\n'.join(str(x) + '--' + ref_article.sentences[x] for x in top_n))
          # print("--real--")
          # print('\n'.join(str(x) + '--' + ref_article.sentences[x] for x in data.offsets.ref))

          # if num_cites > 1:
          #     print(len([x for x in top_n if
          #                ("intro" in ref_article.sections[x].lower() or "abstract" in ref_article.sections[
          #                    x].lower() or "concl" in ref_article.sections[x].lower() or "summ" in
          #                 ref_article.sections[x].lower())]))

          fp = len(top_n)
          for x in true_ref_sentence_ids:
            if x in top_n:
              fp -= 1
              tp += 1
            else:
              #print(ref_article.sentences[x])
              #print([citing_article.sentences[x] for x in citing_sentence_ids])
              fn += 1
      else:
        top_n = {}
        fn += len(true_ref_sentence_ids)
      ref_offsets = str([str(x) for x in top_n.keys()])
      cite_offsets = str([str(x) for x in offsets.marker])
      result = template_line.replace('####PREDICTION####', ref_offsets).replace("####REF####", ref_article.id).replace("####CITE####", citing_article.id).replace("####CITE_OFF####", cite_offsets)
      return [(ref_article.id, citing_article.id, str(offsets.marker)), result, [x for x in top_n.keys()], (tp, fp, fn)]


template_line = '''Citance Number: 1 | Reference Article:  ####REF####.txt | Citing Article:  ####CITE####.txt | Citation Marker Offset:  ####CITE_OFF#### | Citation Marker:  Green and Manning, 2010 | Citation Offset:  ['39'] | Citation Text:  <S sid ="39" ssid = "13">Joint segmentation and parsing was also investigated for Arabic (Green and Manning, 2010).</S> | Reference Offset:  ####PREDICTION#### | Reference Text:  <S sid ="270" ssid = "63">6 Joint Segmentation and Parsing.</S> | Discourse Facet:  Method_Citation | Annotator:  Ankita Patel |'''

pbar = tqdm(dataset)


import re

def has_multiple_cites(sentence):
    matches = re.findall(r"\D(\d{4})\D", sentence)
    return len(matches)


groups = [("abstract", "intro", "concl", "paper", "summ"), ("",)]

ctr = 0
total_multiple = 0
total = 0
for data in dataset:
    cite_aticle = data.cite
    cite_offset = data.offsets.cite
    sentence = ' '.join([cite_aticle.sentences[c] for c in cite_offset])
    ref_offsets = data.offsets.ref
    ref_sections = [data.ref.sections[c] for c in ref_offsets]
    if has_multiple_cites(sentence) > 3:
        intro_group = groups[0]
        for sec in intro_group:
            for ref_sec in ref_sections:
                if sec in ref_sec.lower():
                    ctr += 1
                    break
        total_multiple += 1
    total += len(data.offsets.ref)

print(ctr, total_multiple, total)
print("here")
































tp = 0
fp = 0
fn = 0
candidates = {}
output_file = open('out.ann.txt', 'w')
total = 0
correct = 0
total_multiple = 0
for ctr, data in enumerate(pbar):
  if tp > 0:
      p = tp/(max(tp+fp,1))
      r = tp/(max(tp+fn,1))
      f1 = 2*p*r/(p+r)
      pbar.set_description("Processing %.3f %.3f %.3f" %(p, r, f1))
  res = map_data(data)
  total += 1
  #print([data.cite.sentences[x] for x in data.offsets.cite])
  candidates[str(res[0])] = res[1]
  #print(data.cite.id, res[2], data.offsets.ref)
  output_file.write(res[1]+'\n')
  tp += res[-1][0]
  fp += res[-1][1]
  fn += res[-1][2]
  if ctr == 10 or ctr % 1000 == 999:
    with open('cand' + str(ctr) +'.json', 'w') as f:
      json.dump(candidates, f)
  cite_aticle = data.cite
  cite_offset = data.offsets.cite
  sentence = ' '.join([cite_aticle.sentences[c] for c in cite_offset])
  ref_offsets = data.offsets.ref
  ref_sections = [data.ref.sections[c] for c in ref_offsets]
  if has_multiple_cites(sentence) > 3:
      intro_group = groups[0]
      for sec in intro_group:
          for ref_sec in ref_sections:
              if sec in ref_sec.lower():
                  ctr += 1
                  if res[-1][0] > 0:
                      correct += res[-1][0]
                  break
      total_multiple += 1

print(total, "correct", correct)
p = tp/(max(tp+fp,1))
r = tp/(max(tp+fn,1))
f1 = 2*p*r/(p+r)
print("f1 is ", f1)
output_file.close()


import multiprocessing as mp

pool = mp.Pool(4)
all_results = []
candidates = {}
ctr = 0
# output_file = open('out.ann.txt', 'w')
# dataset = [x for x in dataset if x.ref.id == "E09-2008"]
# print([(x.cite.id, x.offsets.ref) for x in dataset])
pbar = tqdm(pool.imap_unordered(map_data, dataset), total=len(dataset))
# pbar = tqdm(dataset)
for res in pbar:
    candidates[str(res[0])] = res[2]

    # print(res[0], res[2])
    # output_file.write(res[1]+'\n')
    if tp > 0:
      p = tp/(max(tp+fp,1))
      r = tp/(max(tp+fn,1))
      f1 = 2*p*r/(p+r)
      pbar.set_description("Processing %.3f %.3f %.3f" %(p, r, f1))
    tp += res[-1][0]
    fp += res[-1][1]
    fn += res[-1][2]
    # if ctr == 10 or ctr % 1000 == 999:
    #     with open('cand' + str(ctr) +'.json', 'w') as f:
    #         json.dump(candidates, f)
    ctr += 1
p = tp/(max(tp+fp,1))
r = tp/(max(tp+fn,1))
f1 = 2*p*r/(p+r)
print("f1 is ", f1)
# with open('cand' + "all" + '.json', 'w') as f:
#     json.dump(candidates, f)

# output_file.close()


print("mapped")


