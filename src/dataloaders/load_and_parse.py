# Currently only works for 2016 - 2018

import os
import xml.etree.cElementTree as ET
import logging
import codecs
from tqdm import tqdm
from dto.namedtuples import Article, Datum, Offsets
import re

DATA_ROOT = '../../data'
sep = os.path.sep
ET.XMLParser(encoding="utf-8")

paper_load_fail = 0
annotation_load_fail = 0

l = logging.getLogger('load_parse')


# Gets names of folders where data is present. ex : "./data/Training-Set-2016/C90-2039_TRAIN"
def get_folders(data_root):
    folders = []
    global folder_skips
    for year_folder in filter(lambda x: os.path.isdir(data_root + sep + x), os.listdir(data_root)):
        year_folder = data_root + sep + year_folder
        for data_folder in os.listdir(year_folder):
            data_folder = year_folder + sep + data_folder
            if not os.path.isdir(data_folder) or '2019' in data_folder or 'Test-Set-2018' in data_folder:
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
                                  parent_map[x].attrib['title'] if len(parent_map[x].attrib) > 0 else parent_map[x].tag)
                                 for x in
                                 sentence_elements]
            # TODO: Check if this is too memory inefficient. Should mostly be okay
            sentence_map = {(i + 1): x[0] for i, x in enumerate(sentence_elements)}
            section_map = {(i + 1): x[1] for i, x in enumerate(sentence_elements)}
            article = Article(xml, sentence_map, section_map)
            return article
    except Exception as e:
        l.error("Error with : " + filename + " with ex : " + str(e))
        paper_load_fail += 1
        return Article(ET.fromstring("<xml></xml>"), {}, {})


def newest_file(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def load_folder_data(annotation_file, articles, is_test=False):
    global annotation_load_fail
    data = []
    try:
        year_matches = re.findall(".*Set[-]([0-9]{4}).*", annotation_file)
        year = int(year_matches[0]) if len(year_matches) > 0 else 2016 # Dev set
    except:
        print("kk")
    with open(annotation_file, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            try:
                parts = line.split(" | ")
                parts = [part.strip() for part in parts]
                ref_article = articles[parts[1].split(":")[1].strip().upper().replace(".XML", "").replace(".TXT", "")]
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
                author = '' if len(parts) < 10 or ":" not in parts[10] else parts[10].split(":")[1].replace("|",
                                                                                                            "").strip()
                d = Datum(ref_article, cite_article, Offsets(marker_offset, citation_offsets, ref_offsets), author,
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


if __name__ == "__main__":
    logging.basicConfig(filename='example.log', level=logging.ERROR, filemode='w')
    print("Logging to example.log")
    load_all(DATA_ROOT)
