###########IMPORTS##################

import codecs
import json
import os
import csv
import xml.etree.cElementTree as ET

import fasttext
import nltk
import numpy as np
import pandas as pd
import torch
from rake_nltk import Rake
from nltk.corpus import stopwords
import re
from collections import namedtuple
import Levenshtein as lvstn
from sklearn.metrics.pairwise import cosine_similarity
from summa import summarizer
import csv
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--extra_bias_intro_id2score', help='source file for the prediction', type=float, default=0.2)
parser.add_argument('--bias_id2score', help='source database for the prediction', type=float, default=0.30)
parser.add_argument('--threshold', help='predictions by the model', type=float, default=0.15)
parser.add_argument('--top_n_to_keep', type=int, default=5)
parser.add_argument('--output_always', type=bool, default=False)
args = parser.parse_args()
print("Starting : ", args)
# from transformers import BertForSequenceClassification, BertTokenizer
extra_bias_intro_id2score = args.extra_bias_intro_id2score
bias_id2score = args.bias_id2score
threshold = args.threshold
top_n_to_keep = args.top_n_to_keep
output_always = args.output_always
training_set = False

###########GLOBALS##################

root = "./Test-Set-2018"
if training_set:
    root = "./Training-Set-2018"
Article = namedtuple('Article', 'id xml sentences sections')
stop = set(stopwords.words('english'))
rakey = Rake(max_length=1, ranking_metric=0)
memo_id_2_score = {}
id2facet = [ "aimcitation", "hypothesiscitation", "implicationcitation", "methodcitation", "resultcitation" ]
###########SUPPORT FUNCTIONS##################

def has_multiple_cites(sentence):
    matches = re.findall(r"\D(\d{4})\D", sentence)
    return len(matches)

def load_article(filename):
    global paper_load_fail
    try:
        # Ignoring non UTF characters for now
        with codecs.open(filename, mode='r', encoding='utf-8', errors='ignore') as target_file:

            xml = ET.parse(target_file)
            parent_map = {c: p for p in xml.getroot().iter() for c in p}
            sentence_elements = list(xml.getroot().iter('S'))
            sentence_elements = [(x.text,
                                  parent_map[x].attrib['title'] if len(parent_map[x].attrib) > 0 else parent_map[x].tag,
                                  int(x.attrib['sid']) if 'sid' in x.attrib and x.attrib['sid'].isdigit() else 9999)
                                 for x in
                                 sentence_elements]
            # TODO: Check if this is too memory inefficient. Should mostly be okay
            sentence_map = {x[2]: x[0] for i, x in enumerate(sentence_elements)}
            section_map = {x[2]: x[1] for i, x in enumerate(sentence_elements)}
            article = Article("", xml, sentence_map, section_map)
            return article
    except Exception as e:
        paper_load_fail += 1
        return Article("", ET.fromstring("<xml></xml>"), {}, {})


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
        score = s[1] + min(0.05, (0.01)*(num_cites-1)) + extra_bias_intro_id2score * id2score.get(sid, 0)
        # score = s[1] + (0.02) * (num_cites - 1) + 0.3 * id2score[sid]
    return [sid, score]



def newest_file(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


###########REAL FUNCTIONS##################

def encode(sentence):
    return sentence

def get_similarity_score(sentence1, sentence2):
    tokens1 = set(re.findall(r'[\w]+', sentence1.lower()))
    tokens2 = set(re.findall(r'[\w]+', sentence2.lower()))
    rakey.extract_keywords_from_sentences(sentence1.replace("-", " ").lower().split())
    keys1 = rakey.get_ranked_phrases()
    rakey.extract_keywords_from_sentences(sentence2.replace("-", " ").lower().split())
    keys2 = rakey.get_ranked_phrases()
    keys2 = set(keys2)
    tokens1 = set(keys1) - stop
    tokens2 = set(keys2) - stop
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))


# def encode(sentence):
#     return fast_model.get_sentence_vector(sentence)
#
#
# def get_similarity_score(vec1, vec2):
#     return cosine_similarity(vec1, vec2)

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('bert-base-nli-mean-tokens')





# def encode(sentence):
#   return model.encode([sentence])
#
# def get_similarity_score(sentence1, sentence2, kernel = "poly_2"):
#     return cosine_similarity(sentence1, sentence2)
#   #return polynomial_kernel(sentence1, sentence2, 2).item()




# model = BertForSequenceClassification.from_pretrained('../../models/bert_model_2018')
# tokenizer = BertTokenizer.from_pretrained('../../models/bert_model_2018')
# model.config.num_labels = 2
# cuda = torch.cuda.is_available()
# device = torch.device("cpu" if not cuda else "cuda")
# model.to(device)

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()
#
# def get_similarity_score(sentence1, sentence2):
#     ref_sentence = " " + sentence1.lower()
#     citing_sentence = " " + sentence2.lower()
#     pairofstrings = [(citing_sentence, ref_sentence)]
#
#     encoded_batch = tokenizer.batch_encode_plus(pairofstrings, add_special_tokens=True, return_tensors='pt',
#                                                 return_special_tokens_mask=True)
#     attention_mask = (encoded_batch['attention_mask'] - encoded_batch['special_tokens_mask']).to(device)
#     input_ids, token_type_ids = encoded_batch['input_ids'].to(device), encoded_batch['token_type_ids'].to(device)
#     logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
#     logits = torch.nn.functional.softmax(logits, dim=1)
#     logits = logits.cpu().detach().numpy()
#     logits = logits[0]
#     # print("sim score (prpb not):", logits[0])
#     return logits[0]




def sim_score_jugaad(ref_article, similarity_score, complete_citing_sentence):
    id2score = get_id2_score(ref_article)
    similarity_score = {x: similarity_score[x] + bias_id2score * id2score.get(x, 0) for x in similarity_score}  # 0.3
    num_cites = has_multiple_cites(complete_citing_sentence)
    sorted_similarity_score = sorted(similarity_score.items(), key=lambda item: -item[1])
    top_n = [s for s in sorted_similarity_score]
    top_n_before_filter = [bias_intro(s, ref_article, num_cites, id2score) for s in top_n]

    top_n = [s for s in top_n_before_filter if s[1] > threshold]  # 0.18

    top_n = {x[0]: x[1] for x in top_n[:top_n_to_keep]}
    if len(top_n) == 0 and output_always:
        return {x[0]: x[1] for x in top_n_before_filter[:1]}
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


#TODO: Few things
def get_cite_texts_csv(path, ref_id):
    annotation_file = path + "/Citance_XML/citing_sentences.json"
    cite_texts = {}
    with open(annotation_file) as f_ann:
        annotations = json.load(f_ann)
    ann_out_template = path + "/annotation/" + ref_id + ".csv"
    with open(ann_out_template) as f:
        reader = csv.reader(f)
        next(reader)
        uniq = 0
        for row in reader:
            cite_no = str(row[0])
            ref_id = row[1]
            cite_id = row[2]
            marker_offset = row[3]
            marker = row[4]
            citation_offsets = row[5]
            citation_text = row[6]
            citation_text_clean = row[7]
            ref_offsets = ''
            ref_text = ''
            facets = ''
            cite_text = citation_text_clean
            if cite_text == "":
                cite_text = citation_text
            uniq += 1
            d = {'Citance Number': cite_no, 'Reference Article': ref_id, 'Citing Article': cite_id,
                 'Citation Marker Offset': marker_offset, 'Citation Marker': marker,
                 'Citation Offset': citation_offsets,
                 'Citation Text': citation_text, 'Citation Text Clean': citation_text_clean,
                 'Reference Offset': ref_offsets,
                 'Reference Text': ref_text, 'Discourse Facet': facets, 'cite_text': cite_text}
            cite_texts[cite_no + "-" + str(uniq)] = d
    return cite_texts



def get_cite_texts_ann(path, ref_id):
    '''
    Citance Number,Reference Article,Citing Article,Citation Marker Offset,Citation Marker,Citation Offset,Citation Text,
    Citation Text Clean,Reference Offset,Reference Text,Discourse Facet
    '''
    annotation_file = newest_file(path + "/annotation/")
    cite_texts = {}
    uniq = 0
    with open(annotation_file) as f_ann:
        for line in f_ann:
            if len(line.strip()) == 0:
                continue
            try:
                parts = line.split(" | ")
                parts = [part.strip() for part in parts]
                citation_html = parts[6].split(":", maxsplit=1)[1].strip()
                tree = ET.fromstring("<root>" + citation_html + "</root>")
                cite_text = ' '.join([x.text for x in tree.iter('S')])
                cite_no = parts[0].split(":")[1].strip()
                uniq += 1
                ref_article = "-".join(
                    parts[1].split(":")[1].strip().upper().replace("_", "-").replace(".XML", "").replace(".TXT",
                                                                                                         "").split("-")[
                    :2])
                cite_article = "-".join(
                    parts[2].split(":")[1].strip().upper().replace("_", "-").replace(".XML", "").replace(".TXT",
                                                                                                         "").split("-")[
                    :2])
                marker_offset = parts[3].split(":")[1].strip()

                marker = parts[4].split(":")[1].strip()

                citation_offsets = parts[5].split(":")[1].strip()
                citation_text = "dummy" # Hopefully not used
                citation_text_clean = "dummy"  # Hopefully not used

                ref_offsets = parts[7].split(":")[1].strip()
                ref_text = parts[8].split(":")[1].strip()  # Hopefully not used
                facets = parts[9].split(":")[1].strip()
                facets = [facets] if '[' not in facets else eval(facets)
                facets = [facet.lower().replace("_", "").replace(" ", "").replace("results", "result") for facet in
                          facets]

                '''
                Citance Number,Reference Article,Citing Article,Citation Marker Offset,Citation Marker,Citation Offset,Citation Text,
                Citation Text Clean,Reference Offset,Reference Text,Discourse Facet
                '''

                d = {'Citance Number' : cite_no, 'Reference Article' : ref_article, 'Citing Article' : cite_article,
                     'Citation Marker Offset' : marker_offset, 'Citation Marker' : marker, 'Citation Offset' : citation_offsets,
                     'Citation Text' : citation_text, 'Citation Text Clean' : citation_text_clean, 'Reference Offset' : ref_offsets,
                     'Reference Text' : ref_text, 'Discourse Facet' : facets, 'cite_text' : cite_text}
                cite_texts[cite_no + "-" + str(uniq)] = d
            except Exception as e:
                raise e
    return cite_texts
#
# {
#     "ACL-1234" : {
#         "ACL-2345-citance_id" : ["23", "42"]
#     }
# }
#


def write_out_2018_train(path, ref_id, results, ref_article, cite_texts):
    '''
    Citance Number,Reference Article,Citing Article,Citation Marker Offset,Citation Marker,Citation Offset,Citation Text,
    Citation Text Clean,Reference Offset,Reference Text,Discourse Facet
    '''
    with open("./"+path+"/Task1/" + ref_id+".csv", "w") as f:
        keys = ['Citance Number', 'Reference Article' , 'Citing Article',
                     'Citation Marker Offset' , 'Citation Marker' , 'Citation Offset' ,
                     'Citation Text' , 'Citation Text Clean' , 'Reference Offset' ,
                     'Reference Text' , 'Discourse Facet']
        writer = csv.writer(f)
        writer.writerow(keys)
        for cite_uniq in cite_texts:
            cite_data = cite_texts[cite_uniq]
            del cite_data['cite_text']
            if cite_uniq in results and len(results[cite_uniq]) > 0:
                result = results[cite_uniq][0]
                cite_ids = ["'"+str(x)+"'" for x in result]
                cite_ids = ','.join(cite_ids)
            else:
                print("Skipping ", ref_id, " ", cite_uniq)
                cite_ids = ""
                result = []
            # VERY IMP LINES
            cite_data['Reference Offset'] = cite_ids
            sents = ['<S ssid="1" sid="'+str(x)+'">' + ref_article.sentences[x]+'</S>' for x in result]
            cite_data["Reference Text"] = ''.join(sents)
            cite_data['Discourse Facet'] = str(results[cite_uniq][1])

            current_row = [cite_data[key] for key in keys]
            writer.writerow(current_row)
################Strategy-hack################

get_cite_texts = get_cite_texts_csv




# write_out = write_out_2018_test

if training_set:
    get_cite_texts = get_cite_texts_ann

write_out = write_out_2018_train

#############Task 2 stuff############
import pickle
with open('../task_2/results_task2.pkl', 'rb') as f:
    results_task2 = pickle.load(f)

def get_task2_result(ref_id, cite_id, cite_num, results_task2):
    current = results_task2[(str(cite_num), ref_id, cite_id)]
    pred = current.pop(0)
    if max(pred) == 0:
        pred[3] = 1
    return pred

###########Main loop##################





sk = 0



task_a_results = {}
for file in os.listdir(root):
    results = {}
    sk += 1
    ref_id = file
    if sk < -1:
        print("Skipping: ", ref_id)
        continue
    path = root +"/" + file
    ref_article = load_article(path+"/Reference_XML/"+file+".xml")

    print("Doing: ", ref_id)
    cite_texts = get_cite_texts(path, ref_id)
    for cite_id in cite_texts:
        cite_text = cite_texts[cite_id]['cite_text']
        best_cites = get_best_cites(ref_article, cite_text)
        # one_result_task2 = get_task2_result(cite_texts[cite_id]['Reference Article'],
        #                                     cite_texts[cite_id]['Citing Article'],
        #                                     int(cite_texts[cite_id]['Citance Number']), results_task2)
        #facets = [id2facet[i] for i, x in enumerate(one_result_task2) if x == 1]
        facets = []
        results[cite_id] = ([x for x in best_cites.keys()], facets)
    output = "runtest"
    if training_set:
        output = "runtrain"
    task_a_results[ref_id] = (output, ref_id, results, ref_article, cite_texts)
    write_out(output, ref_id, results, ref_article, cite_texts)

best_ids_cached = {}
# all_results = {}
facet_map = {}
facet_count = {}
facet_to_sentences = {}
full_facet_to_ref_section_names_freq_count = {}
all_sections = set()
all_ref_sections = set()
test_data_points = []

with open("../task_2/facet_word_freq.json") as f:
    facet_word_freq = json.load(f)

with open("../task_2/full_facet_whatever.json") as f:
    full_facet_to_ref_section_names_freq_count = json.load(f)

stop_words = set(stopwords.words('english'))
for ref_id in task_a_results:
    path = root + "/" + ref_id
    ref_article = load_article(path+"/Reference_XML/"+ref_id+".xml")

    root_cites = path + "/Citance_XML/"
    all_cite_articles = {}
    for file_in_cite_dir in os.listdir(root_cites):
        if ".xml" not in file_in_cite_dir:
            continue
        cite_path = root_cites + file_in_cite_dir
        cite_article = load_article(cite_path)
        cite_name = "-".join(file_in_cite_dir.upper().replace("_", "-").replace(".XML", "").replace(".TXT","").split("-")[:2])
        all_cite_articles[cite_name] = cite_article

    for cite_text_id in task_a_results[ref_id][4]:
        cite_texts = task_a_results[ref_id][4]
        cite_text = cite_texts[cite_text_id]
        cite_name = "-".join(cite_text["Citing Article"].upper().replace("_", "-").replace(".XML", "").replace(".TXT", "").split("-")[:2])
        print(all_cite_articles.keys())
        cite_id = cite_text["Citing Article"]
        cite_num = cite_text['Citance Number']
        facets = []
        # ref_sentence_ids = offsets.ref
        ref_sentence_ids = task_a_results[ref_id][2][cite_text_id][0]
        ref_section = 'introduction' if len(ref_sentence_ids) == 0 else ref_article.sections[ref_sentence_ids[0]].lower()
        all_ref_sections.add(ref_section)
        ref_sentence_text = ' '.join([ref_article.sentences[x] for x in ref_sentence_ids])
        cite_line_ratio = 1

        # line ratio for first line in reference sentence
        ref_line_ratio = 0.1 if len(ref_sentence_ids) == 0 else ref_sentence_ids[0] / len(ref_article.sentences)
        isPercentPresent = False
        isFloatingPointPresent = False

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
        ref_sections = set([ref_article.sections[id].lower() for id in ref_sentence_ids])
        # features for prob distribution of section in different facets
        facet_section_prob = {k: 0 for k in facet_word_freq.keys()}
        for section in ref_sections:
            for facet in facet_word_freq:
                if section in full_facet_to_ref_section_names_freq_count[facet]:
                    facet_section_prob[facet] += full_facet_to_ref_section_names_freq_count[facet][section]

        facet_prob = {k: str(v) for k, v in facet_prob.items()}



        test_data_points.append([cite_num, cite_id, ref_id, cite_line_ratio, ref_line_ratio, isPercentPresent,
                         isFloatingPointPresent, facet_prob["aimcitation"],
                         facet_prob["hypothesiscitation"], facet_prob["implicationcitation"],
                         facet_prob["methodcitation"], facet_prob["resultcitation"],
                         facet_section_prob["aimcitation"], facet_section_prob["hypothesiscitation"],
                         facet_section_prob["implicationcitation"], facet_section_prob["methodcitation"],
                         facet_section_prob["resultcitation"], 1, 0, 0, 0, 0, ref_sentence_text])



test_columns = ['cite_num', 'cite_id', 'ref_id', 'cite_line_ratio', 'ref_line_ratio', 'isPercentPresent',
                   'isFloatingPointPresent', 'facet_prob_aim', 'facet_prob_hypothesis', 'facet_prob_implication',
                   'facet_prob_method', 'facet_prob_result', 'facet_section_prob_aim', 'facet_section_prob_hypothesis',
                   'facet_section_prob_implication', 'facet_section_prob_method', 'facet_section_prob_result',
                   'is_aimcitation', 'is_hypothesiscitation', 'is_implicationcitation', 'is_methodcitation',
                   'is_resultcitation', 'ref_text']
test = pd.DataFrame(test_data_points,columns=test_columns)
model = fasttext.load_model("/Users/kai/PycharmProjects/QA_SDP2/src/task_2/vecs30.bin")
ids_test = test.iloc[:, :3].to_numpy().tolist()
x_test = test.iloc[:, 3:17]
y_test = [test.iloc[i, 17:22].tolist() for i in range(0, len(x_test))]
x_vecs = test.iloc[:, 22].tolist()
x_vecs = [model.get_sentence_vector(text) for text in x_vecs]
x_vecs = pd.DataFrame(x_vecs)
x_test = pd.concat([x_test, x_vecs], axis=1)

with open("../task_2/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

x_test = scaler.transform(x_test)

with open("../task_2/model.pkl", "rb") as f:
    clf = pickle.load(f)

def make_pred_json(ids, y_pred):
    results_task2 = {}
    for i in range(len(ids)):
        cite_num, cite, ref = ids[i]
        pred = y_pred[i]

        if (cite_num, ref, cite) not in results_task2:
            results_task2[(cite_num, ref, cite)] = [pred]
        else:
            current = results_task2[(cite_num, ref, cite)]
            current.append(pred)
    return results_task2

y_pred = clf.predict(x_test)
results_test = make_pred_json(ids_test, y_pred)



for file in os.listdir(root):
    results = {}
    sk += 1
    ref_id = file
    if sk < -1:
        print("Skipping: ", ref_id)
        continue
    path = root +"/" + file
    ref_article = load_article(path+"/Reference_XML/"+file+".xml")

    print("Doing: ", ref_id)
    cite_texts = get_cite_texts(path, ref_id)
    for cite_id in cite_texts:
        cite_text = cite_texts[cite_id]['cite_text']
        result_cite_id = task_a_results[ref_id][-3][cite_id][0]
        one_result_task2 = get_task2_result(cite_texts[cite_id]['Reference Article'],
                                            cite_texts[cite_id]['Citing Article'],
                                            int(cite_texts[cite_id]['Citance Number']), results_test)
        facets = [id2facet[i] for i, x in enumerate(one_result_task2) if x == 1]
        # facets = []
        results[cite_id] = (result_cite_id, facets)
    output = "runtest"
    if training_set:
        output = "runtrain"
    write_out(output, ref_id, results, ref_article, cite_texts)

# RUN THIS:

'''
rm -rf ./src/for_submission/2018-evaluation-script/eval_dirs/res/Task1/ && cp -r ./src/for_submission/runtrain/Task1 ./src/for_submission/2018-evaluation-script/eval_dirs/res
'''