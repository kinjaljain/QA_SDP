import pickle
import tqdm
from collections import namedtuple
from nltk.corpus import stopwords

root_path = "/Users/kinjal/Desktop/Spring2020/11797/QA_SDP/src/dataloaders/"

Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
    dataset = pickle.load(f)

def clean(sentence):
    return sentence.replace("- ", "")

facet_map = {}
facet_count = {}
facet_to_sentences = {}
for data in tqdm.tqdm(dataset):
    citing_article = data.cite
    cite_id = citing_article.id
    facets = data.facet
    ref_article = data.ref
    ref_id = data.ref.id
    offsets = data.offsets
    citing_sentence_ids = offsets.cite
    ref_sentence_ids = offsets.ref
    section = citing_article.sections[citing_sentence_ids[0]]
    new_ids = [x for x in citing_sentence_ids]
    # orig_facet = facet
    facets = [facet.lower().replace("_", "").replace(" ", "").replace("results", "result") for facet in facets]

    if len(facets) == 1:
        if facets[0] in facet_to_sentences:
            # sens = [clean(citing_article.sentences[sen_id]) for sen_id in citing_sentence_ids]
            # for i in sens:
            #     facet_to_sentences[facets[0]].add(i)
            sens = [clean(ref_article.sentences[sen_id]) for sen_id in ref_sentence_ids]
            for i in sens:
                facet_to_sentences[facets[0]].add(i)
        else:
            # facet_to_sentences[facets[0]] = set([clean(citing_article.sentences[sen_id]) for sen_id in citing_sentence_ids])
            facet_to_sentences[facets[0]] = set([clean(ref_article.sentences[sen_id]) for sen_id in ref_sentence_ids])
            # sens = [clean(citing_article.sentences[sen_id]) for sen_id in citing_sentence_ids]
            # for i in sens:
            #     facet_to_sentences[facets[0]].add(i)
    facet_name = "*".join(facets)
    if facet_name not in facet_map:
        facet_map[facet_name] = len(facet_map.keys())
    facet_count[facet_name] = facet_count.get(facet_name, 0) + 1

print("facet count: ", facet_count)
print("facet map: ", facet_map)
print("facet to sentences:", facet_to_sentences)
# with open('task2_dist.txt', 'w') as f:
#     for facet in facet_map:
#         f.write(facet + ":\n")
#         if facet in facet_to_sentences:
#             for sentence in facet_to_sentences[facet]:
#                 f.write(sentence + "\n")

stop_words = set(stopwords.words('english'))
print(stop_words)

facet_word_freq = {}
facet_word_count = {}
for facet in facet_to_sentences:
    if facet not in facet_word_freq:
        facet_word_freq[facet] = {}
        facet_word_count[facet] = 0
    for sentence in facet_to_sentences[facet]:
        words = sentence.replace(".", "").split()
        for word in words:
            if word.lower() in stop_words or len(word) < 4:
                continue
            if word.lower() not in facet_word_freq[facet]:
                facet_word_freq[facet][word.lower()] = 1
            else:
                facet_word_freq[facet][word.lower()] += 1
            facet_word_count[facet] += 1

with open('task2_ref_top15_facet_word_dist.txt', 'w') as f:
    for facet in facet_word_freq:
        for word in facet_word_freq[facet]:
            facet_word_freq[facet][word] /= facet_word_count[facet]
        sorted_facet_words = sorted(facet_word_freq[facet].items(), key=lambda x: x[1], reverse=True)
        f.write(facet + ":\n")
        for word in sorted_facet_words[:15]:
            f.write("{}: {}\n".format(word[0], word[1]))
        f.write("*" * 50 + "\n\n\n")

print("ok")
