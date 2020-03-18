import json
import nltk
from tqdm import tqdm
nltk.download('stopwords')
from dataloaders.load_and_parse import load_all
from similarity_metrics.jaccard import get_similarity_score as j
# from similarity_metrics.dice import get_similarity_score as dc
# from similarity_metrics.word2vec import get_similarity_score as w
import similarity_metrics.tfidf as t
from similarity_metrics.word2vec import build_model
import os.path
import pickle

DATA_ROOT = '../../data'

dataset = load_all(DATA_ROOT)


def prepare(dataset, reference_context):
    result = []
    for data in tqdm(dataset):
        ref_article = data.ref.sentences
        try:
            ref_title = ref_article[0]
        except:
            ref_title = "No title"
        ref_sections = data.ref.sections
        citing_article = data.cite.sentences
        offsets = data.offsets
        positives = offsets.ref
        negatives = ref_article.keys() - set(positives)
        if len(negatives) < 3 or not citing_article:
            continue
        positive_ids = []
        for x in positives:
            positive_ids.extend(
                [i for i in range(max(1, x - reference_context), min(x + reference_context, len(ref_article) - 1) + 1)])
        complete_citing_sentence = " ".join([citing_article[c] for c in offsets.cite])
        negatives = set(negatives) - set(positives)
        negatives = sorted(negatives, key=lambda x: j(ref_article[x], complete_citing_sentence), reverse=True)[:3]
        result.extend([(" ".join([ref_article[i] for i in range(max(1, x - reference_context),
                                                                min(x + reference_context, len(ref_article)-1) + 1)]),
                        complete_citing_sentence, 0, ref_title.strip(),
                        ref_sections[x].strip() if any(c.isalpha() for c in ref_sections[x].strip()) else "##other##") for x in negatives])
        try:
            result.extend([(" ".join([ref_article[i] for i in range(max(1, x - reference_context),
                                                                    min(x + reference_context, len(ref_article)-1) + 1)]),
                            complete_citing_sentence, 1, ref_title.strip(),
                            ref_sections[x].strip() if any(c.isalpha() for c in ref_sections[x].strip()) else "##other##") for x in positives])
        except:
            print("hello")
    return result

reference_context = 2
result = prepare(dataset, reference_context)
with open("prepared_data_with_context_title_section.json", "w") as f:
    json.dump(result, f)

