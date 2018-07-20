import pandas as pd
train = pd.read_csv('./data/labeledTrainData.tsv',delimiter='\t')
test = pd.read_csv('./data/testData.tsv',delimiter='\t')

print(train.head())

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

