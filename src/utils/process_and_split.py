import re
import nltk
nltk.download('stopwords')
from dataloaders.load_and_parse import load_all
from similarity_metrics.jaccard import get_similarity_score as j
# from similarity_metrics.dice import get_similarity_score as dc
# from similarity_metrics.word2vec import get_similarity_score as w
from tqdm import tqdm
DATA_ROOT = '../../data'
global top_n

def split_citing_sentence(sentence_ids, sentences):
    # keep sentences separate and check if they can further be broken down into simpler sentences
    # intuition: splitting on conjuctions will give simpler and more answerable portions
    citing_sentences = []
    for id in sentence_ids:
        sentence = sentences[id]
        if len(sentence) > 10:
            # conjuctions = re.compile('\sand\s|\sor\s|\sbut\s|\showever\s|\sif\s|\swhile\s|\salthough\s')
            # conjuctions = re.compile('\sand\s|\sor\s')
            conjuctions = re.compile(',|;|\s\s+')
            s = conjuctions.split(sentence)
            # if len(s) >= 2:
            #     # check if splits are at least of length 5
            #     if any([len(x) < 5 for x in s]):
            #         citing_sentences.append(sentence)
            #     else:
            #         citing_sentences.extend(s)
            citing_sentences.extend([x for x in s if len(x) >= 5])
    return citing_sentences


def find_next(similarity_scores, current_ids):
    for key, id in current_ids.items():
        if id == -1:
            current_ids[key] += 1
            sentence_id = similarity_scores[key][current_ids[key]][0]
            if sentence_id in top_n:
                if top_n[sentence_id] < similarity_scores[key][current_ids[key]][1]:
                    top_n[sentence_id] = similarity_scores[key][current_ids[key]][1]
                return
            top_n[sentence_id] = similarity_scores[key][current_ids[key]][1]
            return

    best_i = 0
    for i in similarity_scores.keys():
        if current_ids[i]+1 >= len(similarity_scores[i]):
            continue
        if similarity_scores[i][current_ids[i]+1][1] > similarity_scores[best_i][current_ids[best_i]+1][1]:
            best_i = i

    current_ids[best_i] += 1
    sentence_id = similarity_scores[best_i][current_ids[best_i]][0]

    if sentence_id in top_n:
        if top_n[sentence_id] < similarity_scores[best_i][current_ids[best_i]][1]:
            top_n[sentence_id] = similarity_scores[best_i][current_ids[best_i]][1]
        return

    top_n[sentence_id] = similarity_scores[best_i][current_ids[best_i]][1]
    return


dataset = load_all(DATA_ROOT)
articles = [x[1] for x in dataset]
n = 5

tp = 0
fp = 0
fn = 0
avg_score = 0
all_scores = []

accuracy = 0.
total = 0.
empty_citations = 0
empty_references = 0

pbar = tqdm(dataset)
for data in pbar:
    ref_article = data.ref
    citing_article = data.cite
    offsets = data.offsets
    citing_sentence_ids = offsets.cite
    true_ref_sentences = offsets.ref
    true_ref_sentence_ids = offsets.ref

    if tp > 0:
        p = tp / (max(tp + fp, 1))
        r = tp / (max(tp + fn, 1))
        f1 = 2 * p * r / (p + r)
        pbar.set_description("Processing %.3f %.3f %.3f" % (p, r, f1))

    if citing_article.sentences:

        new_ids = [c for c in citing_sentence_ids]
        for c in citing_sentence_ids:
            # If additional context is reqd
            to_add = 0
            extra = range(max(1, c - to_add), c)
            new_ids.extend(extra)
            extra = range(c + 1, min(len(citing_article.sentences), c + to_add + 1))
            new_ids.extend(extra)
        citing_sentence_ids = new_ids

        # joining the entire set of citing sentences into one big sentence
        complete_citing_sentence = " ".join([citing_article.sentences[c] for c in citing_sentence_ids])
        if len(complete_citing_sentence) > 10:
            citing_sentences = split_citing_sentence(citing_sentence_ids, citing_article.sentences)
        else:
            citing_sentences = complete_citing_sentence
        similarity_score = {}
        for i, citing_sentence in enumerate(citing_sentences):
            similarity_score[i] = {}
            for ref_id, ref_sentence in ref_article.sentences.items():
                try:
                    similarity_score[i][ref_id] = j(ref_sentence, citing_sentence)
                except Exception as e:
                    print(e)

        if similarity_score:
            for i in similarity_score.keys():
                sorted_similarity_score = sorted(similarity_score[i].items(), key=lambda item: -item[1])
                similarity_score[i] = sorted_similarity_score

            top_n = {}
            current_ids = {key: -1 for key in similarity_score.keys()}
            while len(top_n) < 20:
                find_next(similarity_score, current_ids)

            top_n = sorted(top_n.items(), key=lambda item: -item[1])
            for i in range(len(top_n)):
                print(ref_article.sentences[top_n[i][0]])
            print("\n")
            for x in true_ref_sentence_ids:
                print(ref_article.sentences[x])
            print("\n\n")
            fp += len(top_n)
            for x in true_ref_sentence_ids:
                if x in top_n:
                    avg_score = (tp * avg_score + top_n[x]) / (max(tp, 1))
                    all_scores.append(top_n[x])
                    fp -= 1
                    tp += 1
                else:
                    fn += 1
    print(tp, fp, fn)


# Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
# Offsets = namedtuple('Offsets', 'marker cite ref')
# Article = namedtuple('Article', 'xml sentences sections')
# articles = list of articles -> each article has dict of sentences (sentence ID : actual sectence)