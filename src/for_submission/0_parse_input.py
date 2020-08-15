###########IMPORTS##################

import codecs
import json
import os
import csv
import xml.etree.cElementTree as ET

import numpy as np
import torch
from rake_nltk import Rake
from nltk.corpus import stopwords
import re
from collections import namedtuple
import Levenshtein as lvstn
from sklearn.metrics.pairwise import cosine_similarity
from summa import summarizer
import csv

from transformers import BertForSequenceClassification, BertTokenizer

training_set = True

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
        score = s[1]+min(0.05, (0.01)*(num_cites-1)) + 0.20 * id2score.get(sid, 0)
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


#TODO: Few things
def get_cite_texts_csv(path, ref_id):
    annotation_file = path + "/Citance_XML/citing_sentences.json"
    cite_texts = {}
    with open(annotation_file) as f_ann:
        annotations = json.load(f_ann)
        # Read annotation
        for annotation in annotations:
            cite_id = annotation["citing_paper_id"]
            cite_text = annotation["clean_text"]
            if cite_text == "":
                cite_text = annotation["raw_text"]
            cite_texts[str(annotation["citance_number"])] = {'cite_text': cite_text}
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
        next(f_ann)
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

def write_out_2018_test(path, ref_id, results, ref_article, cite_texts = None):
    ann_out_template = path +"/annotation/" + ref_id +".csv"
    #TODO: Deal with repeated cite_nos
    with open(ann_out_template) as f:
        new_rows = []
        reader = csv.reader(f)
        new_rows.append(next(reader)) # HEaders
        for line in reader:
            cite_no = line[0]
            new_rows.append(line)
            if cite_no in results and len(results[cite_no]) > 0:
                result = results[cite_no]
                cite_ids = ["'"+str(x)+"'" for x in result]
                cite_ids = ','.join(cite_ids)
            else:
                print("Skipping ", ref_id, " ", cite_no)
                cite_ids = ""
            new_rows[-1][-3] = cite_ids
            sents = ['<S ssid="1" sid="' + str(x) + '">' + ref_article.sentences[x] + '</S>' for x in result]
            new_rows[-1][-2] = ''.join(sents)
            new_rows[-1][-1] = "methodcitation"
    with open("./run1/Task1/" + ref_id+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

def write_out_2018_train(path, ref_id, results, ref_article, cite_texts):
    '''
    Citance Number,Reference Article,Citing Article,Citation Marker Offset,Citation Marker,Citation Offset,Citation Text,
    Citation Text Clean,Reference Offset,Reference Text,Discourse Facet
    '''
    #TODO: Deal with repeated cite_nos
    with open("./runtrain/Task1/" + ref_id+".csv", "w") as f:
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




write_out = write_out_2018_test

if training_set:
    get_cite_texts = get_cite_texts_ann
    write_out = write_out_2018_train

#############Task 2 stuff############
import pickle
with open('../task_2/results_task2.pkl', 'rb') as f:
    results_task2 = pickle.load(f)

def get_task2_result(ref_id, cite_id, cite_num, results_task2):
    current = results_task2[(cite_num, ref_id, cite_id)]
    pred = current.pop()
    return pred

###########Main loop##################





sk = 0




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
        one_result_task2 = get_task2_result(cite_texts[cite_id]['Reference Article'],
                                            cite_texts[cite_id]['Citing Article'],
                                            cite_texts[cite_id]['Citance Number'], results_task2)
        facets = [id2facet[i] for i, x in enumerate(one_result_task2) if x == 1]
        results[cite_id] = ([x for x in best_cites.keys()], facets)
    write_out(path, ref_id, results, ref_article, cite_texts)

# RUN THIS:

'''
rm -rf ./src/for_submission/2018-evaluation-script/eval_dirs/res/Task1/ && cp -r ./src/for_submission/runtrain/Task1 ./src/for_submission/2018-evaluation-script/eval_dirs/res
'''