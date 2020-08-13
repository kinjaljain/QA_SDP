import pickle
import tqdm
from collections import namedtuple
from collections import defaultdict
from collections import OrderedDict

root_path = "/Users/anjanau/Courses/sem_2/QA/SUBMISSION/QA_SDP/src/dataloaders/"

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
full_facet_to_citing_section_names = {}
full_facet_to_citing_section_names_freq_count = {}
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
    offsets = data.offsets
    ref_sentence_ids = offsets.ref
    citing_sentence_ids = offsets.cite
    section = citing_article.sections[citing_sentence_ids[0]].lower()
    ref_section = ref_article.sections[ref_sentence_ids[0]].lower()
    all_sections.add(section)
    all_ref_sections.add(ref_section)
    new_ids = [x for x in citing_sentence_ids]
    # orig_facet = facet
    facets = [facet.lower().replace("_", "").replace(" ", "").replace("results", "result") for facet in facets]
    # facet = facet.lower().replace("_", "").replace(" ", "").replace("results", "result")
    # Assuming they will fall in same section

    if len(facets) == 1:
        if facets[0] in facet_to_sentences:
            sens = [clean(citing_article.sentences[sen_id]) for sen_id in citing_sentence_ids]
            for i in sens:
                facet_to_sentences[facets[0]].add(i)
        else:
            facet_to_sentences[facets[0]] = set([clean(citing_article.sentences[sen_id]) for sen_id in citing_sentence_ids])

    facet_name = "*".join(facets)
    if facet_name not in facet_map:
        facet_map[facet_name] = len(facet_map.keys())
        if len(facets) == 1:
            full_facet_to_citing_section_names[facet_name] = [section]
            full_facet_to_ref_section_names[facet_name] = [ref_section]
    else:
        if len(facets) == 1:
            full_facet_to_citing_section_names[facet_name].append(section)
            full_facet_to_ref_section_names[facet_name].append(ref_section)
    facet_count[facet_name] = facet_count.get(facet_name, 0) + 1

for facet_name in full_facet_to_citing_section_names:
    full_facet_to_citing_section_names_freq_count[facet_name] = {}
    full_facet_to_ref_section_names_freq_count[facet_name] = {}

    for citing_section_name in full_facet_to_citing_section_names[facet_name]:
        full_facet_to_citing_section_names_freq_count[facet_name][citing_section_name] = \
            full_facet_to_citing_section_names_freq_count[facet_name].get(citing_section_name, 0) + 1

    for ref_section_name in full_facet_to_ref_section_names[facet_name]:
        full_facet_to_ref_section_names_freq_count[facet_name][ref_section_name] = \
            full_facet_to_ref_section_names_freq_count[facet_name].get(ref_section_name, 0) + 1

print("all sections: ", all_sections)
print("count of all sections:", len(all_sections))
print("facet count: ", facet_count)
print("facet map: ", facet_map)
print("full_facet_to_citing_section_names:", full_facet_to_citing_section_names)
print("full_facet_to_citing_section_names_freq_count", full_facet_to_citing_section_names_freq_count)
print("facet to sentences:", facet_to_sentences)
with open('task2_dist.txt', 'w') as f:
    for facet in facet_map:
        f.write(facet + ":\n")
        if facet in facet_to_sentences:
            for sentence in facet_to_sentences[facet]:
                f.write(sentence + "\n")

with open('task2_dist_full_facet2citingsection.txt', 'w') as f:
    for facet in facet_map:
        f.write(facet + ":\n")
        if facet in full_facet_to_citing_section_names:
            for sentence in full_facet_to_citing_section_names[facet]:
                f.write(sentence + "\n")

with open('task2_dist_full_facet_to_citing_section_names_freq_count.txt', 'w') as f:
    for facet in facet_map:
        if facet in full_facet_to_citing_section_names_freq_count:
            f.write(facet + ":\n")
            # print("Facet:", facet)
            sorted_section_names = sorted(full_facet_to_citing_section_names_freq_count[facet], key=full_facet_to_citing_section_names_freq_count[facet].get, reverse=True)
            sum_for_facet = 0
            for name in sorted_section_names:
                sum_for_facet += full_facet_to_citing_section_names_freq_count[facet][name]
            for section_name in sorted_section_names:
                f.write("\t- " + str(section_name) + ": " + str(full_facet_to_citing_section_names_freq_count[facet][section_name]/sum_for_facet) + "\n")


with open('task2_dist_full_facet_to_ref_section_names_freq_count.txt', 'w') as f:
    for facet in facet_map:
        if facet in full_facet_to_ref_section_names_freq_count:
            f.write(facet + ":\n")
            sorted_section_names = sorted(full_facet_to_ref_section_names_freq_count[facet], key=full_facet_to_ref_section_names_freq_count[facet].get, reverse=True)
            sum_for_facet = 0
            for name in sorted_section_names:
                sum_for_facet += full_facet_to_ref_section_names_freq_count[facet][name]
            for section_name in sorted_section_names:
                f.write("\t- " + str(section_name) + ": " + str(full_facet_to_ref_section_names_freq_count[facet][section_name]/sum_for_facet) + "\n")
print("ok")
