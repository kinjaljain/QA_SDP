import pickle
import re
import tqdm
from collections import namedtuple
from nltk.corpus import stopwords
import csv

root_path = "/Users/kinjal/Desktop/Spring2020/11797/QA_SDP/src/dataloaders/"

Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')

dev_ids = ["C00-2123", "C04-1089", "I05-5011", "N06-2049", "P05-1004", "P98-1046", "P98-2143", "W03-0410"]

# with open("%s-data-2018-clean.pkl" % root_path, 'rb') as f:
#     dataset = pickle.load(f)

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
    dataset = pickle.load(f)

facet_map = {}
facet_count = {}
facet_to_sentences = {}
full_facet_to_ref_section_names = {}
full_facet_to_ref_section_names_freq_count = {}
all_sections = set()
all_ref_sections = set()
for data in tqdm.tqdm(dataset):
    citing_article = data.cite
    cite_id = citing_article.id
    facets = data.facet
    ref_article = data.ref
    ref_id = data.ref.id
    if ref_id in dev_ids:
        continue
    offsets = data.offsets
    citing_sentence_ids = offsets.cite
    ref_sentence_ids = offsets.ref
    ref_section = ref_article.sections[ref_sentence_ids[0]].lower()
    all_ref_sections.add(ref_section)
    new_ids = [x for x in citing_sentence_ids]
    facets = [facet.lower().replace("_", "").replace(" ", "").replace("results", "result") for facet in facets]

    if len(facets) == 1:
        if facets[0] in facet_to_sentences:
            sens = [ref_article.sentences[sen_id] for sen_id in ref_sentence_ids]
            for i in sens:
                facet_to_sentences[facets[0]].add(i)
        else:
            facet_to_sentences[facets[0]] = set([ref_article.sentences[sen_id] for sen_id in ref_sentence_ids])
    facet_name = "*".join(facets)
    if facet_name not in facet_map:
        facet_map[facet_name] = len(facet_map.keys())
        if len(facets) == 1:
            full_facet_to_ref_section_names[facet_name] = [ref_section]
    else:
        if len(facets) == 1:
            full_facet_to_ref_section_names[facet_name].append(ref_section)

    facet_count[facet_name] = facet_count.get(facet_name, 0) + 1

for facet_name in full_facet_to_ref_section_names:
    full_facet_to_ref_section_names_freq_count[facet_name] = {}
    for ref_section_name in full_facet_to_ref_section_names[facet_name]:
        full_facet_to_ref_section_names_freq_count[facet_name][ref_section_name] = \
            full_facet_to_ref_section_names_freq_count[facet_name].get(ref_section_name, 0) + 1

for facet in facet_map:
    if facet in full_facet_to_ref_section_names_freq_count:
        sorted_section_names = sorted(full_facet_to_ref_section_names_freq_count[facet],
                                      key=full_facet_to_ref_section_names_freq_count[facet].get, reverse=True)
        sum_for_facet = 0
        for name in sorted_section_names:
            sum_for_facet += full_facet_to_ref_section_names_freq_count[facet][name]
        for section_name in sorted_section_names:
            full_facet_to_ref_section_names_freq_count[facet][section_name] /= sum_for_facet

stop_words = set(stopwords.words('english'))

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

for facet in facet_word_freq:
    for word in facet_word_freq[facet]:
        facet_word_freq[facet][word] /= facet_word_count[facet]
    sorted_facet_words = sorted(facet_word_freq[facet].items(), key=lambda x: x[1], reverse=True)

with open("%straindata.csv" % root_path, "w") as f:
    writer = csv.writer(f)
    for data in tqdm.tqdm(dataset):
        citing_article = data.cite
        cite_id = citing_article.id
        facets = data.facet
        ref_article = data.ref
        ref_id = data.ref.id
        if ref_id in dev_ids:
            continue
        offsets = data.offsets
        citing_sentence_ids = offsets.cite
        ref_sentence_ids = offsets.ref
        section = citing_article.sections[citing_sentence_ids[0]]
        ref_sections = set([ref_article.sections[id].lower() for id in ref_sentence_ids])
        new_ids = [x for x in citing_sentence_ids]
        facets = [facet.lower().replace("_", "").replace(" ", "").replace("results", "result") for facet in facets]

        facet_y = {k: 0 for k in facet_word_freq.keys()}
        for facet in facets:
            facet_y[facet] = 1

        isPercentPresent = False
        isFloatingPointPresent = False

        # features for prob distribution of words in different facets
        facet_prob = {k: 0 for k in facet_word_freq.keys()}
        for sentence in [ref_article.sentences[sen_id] for sen_id in ref_sentence_ids]:
            for word in sentence.split():
                word = word.lower()
                if "%" in word:
                    isPercentPresent = True
                if re.search("([0-9]*[.])?[0-9]+", word):
                    isPercentPresent = True
                if word in stop_words:
                    continue
                for facet in facet_word_freq:
                    if word in facet_word_freq[facet]:
                        facet_prob[facet] += facet_word_freq[facet][word]
        facet_prob = {k: str(v) for k, v in facet_prob.items()}
        # TODO: should divide by number of words/ sentences?

        # features for prob distribution of section in different facets
        facet_section_prob = {k: 0 for k in facet_word_freq.keys()}
        for section in ref_sections:
            for facet in facet_word_freq:
                if section in full_facet_to_ref_section_names_freq_count[facet]:
                    facet_section_prob[facet] += full_facet_to_ref_section_names_freq_count[facet][section]

        facet_prob = {k: str(v) for k, v in facet_prob.items()}
        # TODO: should divide by number of words/ sentences?

        # line ratio for first line in citing sentence
        cite_line_ratio = citing_sentence_ids[0]/len(citing_article.sentences)

        # line ratio for first line in reference sentence
        ref_line_ratio = ref_sentence_ids[0] / len(ref_article.sentences)

        print(cite_id, ref_id, cite_line_ratio, ref_line_ratio, isPercentPresent,
              isFloatingPointPresent, facet_prob["aimcitation"],
              facet_prob["hypothesiscitation"], facet_prob["implicationcitation"],
              facet_prob["methodcitation"], facet_prob["resultcitation"],
              facet_section_prob["aimcitation"], facet_section_prob["hypothesiscitation"],
              facet_section_prob["implicationcitation"], facet_section_prob["methodcitation"],
              facet_section_prob["resultcitation"], facet_y["aimcitation"], facet_y["hypothesiscitation"],
              facet_y["implicationcitation"], facet_y["methodcitation"], facet_y["resultcitation"])
        writer.writerow([cite_id, ref_id, str(cite_line_ratio), str(ref_line_ratio), str(isPercentPresent),
                         str(isFloatingPointPresent), facet_prob["aimcitation"],
                         facet_prob["hypothesiscitation"], facet_prob["implicationcitation"],
                         facet_prob["methodcitation"], facet_prob["resultcitation"],
                         facet_section_prob["aimcitation"], facet_section_prob["hypothesiscitation"],
                         facet_section_prob["implicationcitation"], facet_section_prob["methodcitation"],
                         facet_section_prob["resultcitation"], facet_y["aimcitation"], facet_y["hypothesiscitation"],
                         facet_y["implicationcitation"], facet_y["methodcitation"], facet_y["resultcitation"]])
print("ok")
