import nltk
nltk.download('stopwords')
from dataloaders.load_and_parse import load_all

DATA_ROOT = '../../data'

dataset = load_all(DATA_ROOT)
# articles = [x[1] for x in dataset]

print("x")