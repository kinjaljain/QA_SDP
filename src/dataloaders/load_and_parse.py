# Currently only works for 2016 - 2018

import os
import copy
import xml.etree.cElementTree as ET
import logging
import codecs

import enchant
import nltk
from tqdm import tqdm
# from dto.namedtuples import Article, Datum, Offsets
import re
import requests
import numpy as np

from collections import namedtuple
d = enchant.Dict("en_US")

Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')

DATA_ROOT = '../../data'
sep = os.path.sep
ET.XMLParser(encoding="utf-8")

paper_load_fail = 0
annotation_load_fail = 0

l = logging.getLogger('load_parse')
replace_count = 0
bad_count = 0

# Gets names of folders where data is present. ex : "./data/Training-Set-2016/C90-2039_TRAIN"
def get_folders(data_root):
    folders = []
    global folder_skips
    for year_folder in filter(lambda x: os.path.isdir(data_root + sep + x), os.listdir(data_root)):
        year_folder = data_root + sep + year_folder
        for data_folder in os.listdir(year_folder):
            data_folder = year_folder + sep + data_folder
            if not os.path.isdir(data_folder) or '2018' not in data_folder or 'Test-Set-2018' in data_folder:
                continue
                # readme
            folders.append(data_folder)
    return folders


def load_article(filename):
    global paper_load_fail
    if not os.path.splitext(filename)[1] == '.xml':
        l.info("Skipping Non xml : " + filename)
        return
    l.info("parsing :" + filename)
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
        l.error("Error with : " + filename + " with ex : " + str(e))
        paper_load_fail += 1
        return Article("", ET.fromstring("<xml></xml>"), {}, {})


def newest_file(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


author_memo = {}
def load_folder_data(annotation_file, articles, is_test=False):
    global annotation_load_fail
    global author_memo
    data = []
    # try:
    #     year_matches = re.findall(".*Set[-]([0-9]{4}).*", annotation_file)
    #     year = int(year_matches[0]) if len(year_matches) > 0 else 2016 # Dev set
    # except:
    #     print("kk")
    with open(annotation_file, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            try:
                parts = line.split(" | ")
                parts = [part.strip() for part in parts]
                ref_article_name = parts[1].split(":")[1].strip().upper().replace(".XML", "").replace(".TXT", "")
                ref_article = articles[ref_article_name]
                cite_article = articles["-".join(
                    parts[2].split(":")[1].strip().upper().replace("_", "-").replace(".XML", "").replace(".TXT",
                                                                                                         "").split("-")[
                    :2])]
                marker_offset = [int(x) for x in (
                    parts[5].split(":")[1].replace("'", "").replace("[", "").replace("]", "").strip()).replace(",",
                                                                                                               " ").split()]
                citation_offsets = [int(x) for x in (
                    parts[5].split(":")[1].replace("'", "").replace("[", "").replace("]", "").strip()).replace(",",
                                                                                                               " ").split()]
                ref_offsets = [int(x) for x in (
                    parts[7].split(":")[1].replace("'", "").replace("[", "").replace("]", "").strip()).replace(",",
                                                                                                               " ").split()]
                facet = parts[9].split(":")[1].strip()
                facet = [facet] if '[' not in facet else eval(facet)
                if ref_article_name not in author_memo:
                    url = "https://www.aclweb.org/anthology/{}.bib".format(ref_article_name)
                    info = requests.get(url=url).text
                    author_info = info.split('{')[1].split(',')[0].split('-')
                    author = get_formatted_author_info(author_info)
                    author_memo[ref_article_name] = author_info
                else:
                    print("Hit")
                    author_info = author_memo[ref_article_name]
                    author = get_formatted_author_info(author_info)
                year = author_info[-2]
                # author = '' if len(parts) < 10 or ":" not in parts[10] else parts[10].split(":")[1].replace("|",
                #                                                                                             "").strip()
                reference, cite = get_clean_cite_and_ref(ref_article, cite_article, ref_offsets, citation_offsets,
                                                         author, year)
                # reference, cite = ref_article, cite_article
                d = Datum(reference, cite, Offsets(marker_offset, citation_offsets, ref_offsets), author,
                          is_test, facet, year)
                data.append(d)
            except Exception as e:
                l.error(e)
                annotation_load_fail += 1
    return data


def load_folder(folder_root):
    annotation_folder = folder_root + sep + "annotation"
    citance_dir = folder_root + sep + "Citance_XML"
    ref_name = folder_root.split(sep)[-1].replace("_TRAIN", "")
    ref_dir = folder_root + sep + "Reference_XML" + sep + ref_name + ".xml"
    articles = {
        '-'.join(os.path.splitext(x)[0].upper().replace("_", "-").split("-")[:2]): load_article(citance_dir + sep + x)
        for x in os.listdir(citance_dir)}

    assert len(articles) == len(os.listdir(citance_dir))

    ref_article = load_article(ref_dir)
    articles[ref_name.upper()] = ref_article
    # Hack
    for key in articles:
        if articles[key]:
            articles[key] = articles[key]._replace(id=key)

    annotation_file = newest_file(annotation_folder)
    folder_data = load_folder_data(annotation_file, articles)
    return folder_data


def load_all(root):
    print("Loading all folders, skipping 2019 and test set 2018 for now")
    folders = get_folders(root)
    print("Going to load :", len(folders))
    dataset = [load_folder(folder) for folder in tqdm(folders)]
    dataset = [data for datalist in dataset for data in datalist]
    print()
    print("Paper load fails due to xml errors : ", paper_load_fail)
    print("Annotation load fails due to random reasons", annotation_load_fail)
    print("Overall loaded ", len(dataset), " datapoints")
    return dataset


def get_clean_cite_and_ref(ref_article, cite_article, ref_offsets, citation_offsets, author, year):
    global replace_count
    global bad_count
    ref = copy.copy(ref_article)
    cite = copy.copy(cite_article)

    # remove cites in ref and fix ocr issues
    for key, sentence in ref.sentences.items():
        ref.sentences[key] = get_cites(sentence)

    if ref_article.id not in visited:
        for key, sentence in ref.sentences.items():
            meaningful, s = filter_meaningless(ref.sentences, key, jargon)
            if not meaningful and key in ref_offsets:
                if ref.sentences[key] != s:
                    bad_count += 1
                    positives_replaced_file.write(ref.sentences[key] + "\n")
                    positives_replaced_file.write(s + "\n\n")
            ref.sentences[key] = s
        visited.add(ref_article.id)

    cite_sentence = " ".join([cite.sentences[c] for c in citation_offsets])
    cite_sentence = re.sub(r"\D(\d{4})\D", '', cite_sentence)  # regex for removing years
    cite_sentence = re.sub(r"\[[0-9]{1,3}\]", '', cite_sentence)  # regex for removing citation numbers
    translation = {ord(')'): None, ord('('): None, ord('.'): None,  ord(','): None, ord('!'): None}
    cite_sentence = cite_sentence.translate(translation)
    author_info = author.split(" ")
    author_2 = None
    if author_info[-1] in ["et.al.", "etal", "Etal"]:
        author_1 = " ".join(author_info[:-1]) + " et al"
        if len(author_info) >= 2:
            author_2 = author_info[0] + " and " + author_info[1]
    elif len(author_info) >= 2 and author_info[-2] not in ["And", "and"]:
        author_1 = author_info[0] + " & " + author_info[1]
        author_2 = author_info[0] + " and " + author_info[1]
    else:
        author_1 = author_info[-1]

    citing_paper_text = author_1
    old_cite_sentence = cite_sentence
    cite_sentence = cite_sentence.replace(citing_paper_text, "##CITATION##")
    if cite_sentence != old_cite_sentence:
        replace_count += 1
    elif author_2:
        citing_paper_text = author_2
        cite_sentence = cite_sentence.replace(citing_paper_text, "##CITATION##")
        if cite_sentence != old_cite_sentence:
            replace_count += 1
        elif len(author_info) >= 2 and author_info[-2] not in ["And", "and"]:
            author_info.sort()
            author_1 = author_info[0] + " & " + author_info[1]
            author_2 = author_info[0] + " and " + author_info[1]
            citing_paper_text = author_1
            old_cite_sentence = cite_sentence
            cite_sentence = cite_sentence.replace(citing_paper_text, "##CITATION##")
            if cite_sentence != old_cite_sentence:
                replace_count += 1
            elif author_2:
                citing_paper_text = author_2
                cite_sentence = cite_sentence.replace(citing_paper_text, "##CITATION##")
                if cite_sentence != old_cite_sentence:
                    replace_count += 1

    # remove all other cites in cite
    cite.sentences[citation_offsets[0]] = cite_sentence
    for key, sentence in cite.sentences.items():
        cite.sentences[key] = get_cites(sentence)

    if len(citation_offsets) > 1:
        for offset in citation_offsets[1:]:
            cite.sentences[offset] = ""
    print("replace_count: ", replace_count)
    return ref, cite


def get_formatted_author_info(author_info):
    author_details = " ".join(author_info)
    author_info = re.sub(r"[0-9]{4}", '_', author_details).split('_')[0].split()
    author = " ".join([x.capitalize() if x is not "and" else x for x in author_info])
    return author


def get_cites(sentence):
    if sentence:
        regex = r"\(\D*\d{4}(;\D*\d{4})*\)"
        sentence = re.sub(regex, " ", sentence)
    return sentence


def filter_meaningless(ref_article, i, jargon):
    line = ref_article[i]
    # line.replace(";", " ").replace(",", " ")
    enc_line = line.encode('unicode_escape').decode()
    line = re.sub(r"\\u....", "", enc_line)
    # Table caption
    if re.search(r'Table [0-9]+: ', line):
        start_idx = line.index("Table ")
        line = line[start_idx:]
        return True, line

    # Figure caption
    elif re.search(r'Figure [0-9]+: ', line):
        start_idx = line.index("Figure ")
        line = line[start_idx:]
        return True, line

    words = np.array(nltk.word_tokenize(ref_article[i]))
    num_singles = np.array([len(word) < 2 for word in words])
    count = words[num_singles].shape[0]
    ratio = count / len(words)

    if ratio > 0.53:
        alphabet_file.write(ref_article[i] + "\n")
        return False, ""

    is_valid = []
    check = words
    for word in words:
        if d.check(word) or word in jargon:
            is_valid.append(True)
        else:
            is_valid.append(False)
    is_valid = np.array(is_valid)
    check = np.array(check)
    count = check[is_valid].shape[0]
    ratio = count / check.shape[0]
    if ratio <= 0.72:
        meaningless_file.write(ref_article[i] + "\n")
        return False, ""
    return True, line


if __name__ == "__main__":
    logging.basicConfig(filename='example.log', level=logging.ERROR, filemode='w')
    print("Logging to example.log")
    meaningless_file = open("meaningless_sentences.txt", "w")
    alphabet_file = open("alphabet_sentences.txt", "w")
    positives_replaced_file = open("positives_replaced.txt", "w")
    jargon = open("jargon.txt", 'r').readlines()
    jargon = [word[:-1] for word in jargon]
    visited = set()

    dataset = load_all(DATA_ROOT)
    print("Good positives replaced: ", bad_count)

    print("dumping")
    import pickle
    with open('processed-data-2018-clean.pkl', 'wb') as f:
        pickle.dump(dataset, f)
