import pickle
import re
import tqdm
from collections import namedtuple
from nltk.corpus import stopwords
import csv
import Levenshtein as lvstn
from rake_nltk import Rake
from summa import summarizer
# from similarity_metrics.ir import rakey

root_path = "/Users/kinjal/Desktop/Spring2020/11797/QA_SDP/src/task_2/"

Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')

dev_ids = ["C00-2123", "C04-1089", "I05-5011", "N06-2049", "P05-1004", "P98-1046", "P98-2143", "W03-0410"]

stop_words = set(stopwords.words('english'))
rakey = Rake(max_length=1, ranking_metric=0)
memo_id_2_score = {}

def encode(sentence):
    return sentence

def has_multiple_cites(sentence):
    matches = re.findall(r"\D(\d{4})\D", sentence)
    return len(matches)

def get_similarity_score(sentence1, sentence2):
    tokens1 = set(re.findall(r'[\w]+', sentence1.lower()))
    tokens2 = set(re.findall(r'[\w]+', sentence2.lower()))
    rakey.extract_keywords_from_sentences(sentence1.replace("-", " ").lower().split())
    keys1 = rakey.get_ranked_phrases()
    rakey.extract_keywords_from_sentences(sentence2.replace("-", " ").lower().split())
    keys2 = rakey.get_ranked_phrases()
    keys2 = set(keys2)
    tokens1 = set(keys1) - stop_words
    tokens2 = set(keys2) - stop_words
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

def get_id2_score(ref):
    global memo_id_2_score
    if ref.id in memo_id_2_score:
        return memo_id_2_score[ref.id]
    sents = ref.sentences.values()
    sents_merged = '\n'.join(sents)
    sent_scores = {x[0]: x[1] for x in summarizer.summarize(sents_merged, ratio=1, split=True, scores=True)}
    id2score = {}
    for id in ref.sentences:
        sentence = ref.sentences[id]
        if sentence in sent_scores:
            id2score[id] = sent_scores[sentence]
        else:
            closest_sent = sorted([x for x in sent_scores.keys()], key=lambda x: lvstn.distance(x, sentence))[0]
            id2score[id] = sent_scores[closest_sent]
    memo_id_2_score[ref.id] = id2score
    return id2score

def bias_intro(s, ref_article, num_cites, id2score):
    sid = s[0]
    present = ("intro" in ref_article.sections[sid].lower() or "abstract" in ref_article.sections[
        sid].lower() or "concl" in ref_article.sections[sid].lower() or "summ" in ref_article.sections[sid].lower())
    if not present:
        score = s[1]
    else:
        score = s[1]+min(0.05, (0.01)*(num_cites-1)) + 0.20 * id2score.get(sid, 0)
        # score = s[1] + (0.02) * (num_cites - 1) + 0.3 * id2score[sid]
    return [sid, score]

def sim_score_jugaad(ref_article, similarity_score, complete_citing_sentence):
    id2score = get_id2_score(ref_article)
    similarity_score = {x: similarity_score[x] + 0.30 * id2score.get(x, 0) for x in similarity_score}  # 0.3
    num_cites = has_multiple_cites(complete_citing_sentence)
    sorted_similarity_score = sorted(similarity_score.items(), key=lambda item: -item[1])
    top_n = [s for s in sorted_similarity_score]
    top_n_before_filter = [bias_intro(s, ref_article, num_cites, id2score) for s in top_n]

    top_n = [s for s in top_n_before_filter if s[1] > 0.15]  # 0.18

    top_n = {x[0]: x[1] for x in top_n[:5]}
    if len(top_n) == 0:
        return {x[0]: x[1] for x in top_n_before_filter[:2]}
    return top_n

def get_best_cites(ref_article, complete_citing_sentence):
    ref_vs = [(x[0], encode(x[1])) for i, x in enumerate(ref_article.sentences.items()) if (has_multiple_cites(x[1]) < 2)]  # or  "intro" in ref_article.sections[x[0]].lower() or "abstract" in ref_article.sections[x[0]].lower() or "concl" in ref_article.sections[x[0]].lower() or "summ" in ref_article.sections[x[0]].lower())]
    similarity_score = {}
    complete_citing_sentence_enc = encode(complete_citing_sentence)
    for ref_id, ref_sentence in ref_vs:
        try:
            similarity_score[ref_id] = max(similarity_score.get(ref_id, 0),
                                           get_similarity_score(ref_sentence, complete_citing_sentence_enc))
        except Exception as e:
            # print(e)
            pass
    return sim_score_jugaad(ref_article, similarity_score, complete_citing_sentence)


# with open("%s-data-2018-clean.pkl" % root_path, 'rb') as f:
#     dataset = pickle.load(f)

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
    dataset = pickle.load(f)
# all_results = {}
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
    if ref_id not in dev_ids:
        continue
    offsets = data.offsets
    citing_sentence_ids = offsets.cite
    full_citing_sentence = " ".join([citing_article.sentences[citing_sentence_id] for citing_sentence_id in citing_sentence_ids])
    # ref_sentence_ids = offsets.ref
    ref_sentences_to_score_map = get_best_cites(ref_article, full_citing_sentence)
    ref_sentence_ids = [x for x in ref_sentences_to_score_map.keys()]
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

with open("%stestdata.csv" % root_path, "w") as f:
    writer = csv.writer(f)
    for data in tqdm.tqdm(dataset):
        citing_article = data.cite
        cite_id = citing_article.id
        facets = data.facet
        ref_article = data.ref
        ref_id = data.ref.id
        if ref_id not in dev_ids:
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
