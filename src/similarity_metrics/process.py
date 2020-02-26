from dataloaders.load_and_parse import load_all
from similarity_metrics.jaccard import get_similarity_score as j
from similarity_metrics.dice import get_similarity_score as dc
from similarity_metrics.word2vec import get_similarity_score as w
from similarity_metrics.word2vec import build_model

DATA_ROOT = '../../data'

dataset = load_all(DATA_ROOT)
articles = [x[1] for x in dataset]
n = 5

accuracy = 0.
total = 0.
empty_citations = 0
empty_references = 0

do_wordnet = False

if do_wordnet:
    sentences = []
    for article in articles:
        sentences.extend(article.sentences.values())
    build_model(sentences)

for data in dataset:
    ref_article = data.ref
    citing_article = data.cite
    offsets = data.offsets
    citing_sentence_ids = offsets.cite
    true_ref_sentences = offsets.ref
    true_ref_sentence_ids = offsets.ref

    if citing_article.sentences:
        # joining the entire set of citing sentences into one big sentence
        complete_citing_sentence = " ".join([citing_article.sentences[c] for c in citing_sentence_ids])
        # print(complete_citing_sentence)
        similarity_score = {}
        for ref_id, ref_sentence in ref_article.sentences.items():
            try:
                similarity_score[ref_id] = w(ref_sentence, complete_citing_sentence)
            except Exception as e:
                print(e)
        if similarity_score:
            sorted_similarity_score = sorted(similarity_score.items(), key=lambda item: -item[1])
            top_n = [s[0] for s in sorted_similarity_score[:n]]
            for i in true_ref_sentence_ids:
                if i in top_n:
                    accuracy += 1
                    break
        else:
            empty_references += 1
    else:
        empty_citations += 1
    total += 1

accuracy /= total
print("Accuracy: {}, Total DataPoints: {}, Num empty references: {}, Num empty citations: {}".format(
    accuracy, total, empty_references, empty_citations))


# Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
# Offsets = namedtuple('Offsets', 'marker cite ref')
# Article = namedtuple('Article', 'xml sentences sections')
# articles = list of articles -> each article has dict of sentences (sentence ID : actual sectence)