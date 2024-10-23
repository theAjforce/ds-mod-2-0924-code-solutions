import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer, TreebankWordTokenizer

nltk.download("punkt")
nltk.download('punkt_tab')

nltk.download("stopwords")
stop_words = stopwords.words("english")
stop_words.remove("i")
idx = stop_words.index("themselves")
stop_words = stop_words[idx+1:]
stop_words.extend([".","!",",","?","(",")",":",";"])

df = pd.read_csv("winemag-data.csv")

tokenized_list = []
for i in df['description']:
   tokenized_list.append(nltk.tokenize.word_tokenize(i))

para_nostop =[]
for list in tokenized_list:
    stop_words_list = []
    for word in list:
        if word not in stop_words:
            stop_words_list.append(word)
    para_nostop.append(stop_words_list)

ps = PorterStemmer()

stemmed_paras = []
for list in para_nostop:
    to_be_stemmed = []
    for word in list:
        to_be_stemmed.append(ps.stem(word))
    stemmed_paras.append(to_be_stemmed)

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('averaged_perceptron_tagger_eng')

lem = WordNetLemmatizer()

nltk.download("averaged_perceptron_tagger")
nltk.download("tagsets")
nltk.download('averaged_perceptron_tagger_eng')

tagged_paras = []
for list in para_nostop:
    tagged = nltk.pos_tag(list)
    tagged_paras.append(tagged)

from nltk.corpus import wordnet as wn

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN
    
results = [[get_wordnet_pos(tag) for word, tag in sentence] for sentence in tagged_paras]

update_tags_paras = [[(word, get_wordnet_pos(tag)) for word, tag in sublist] for sublist in tagged_paras]

lem = WordNetLemmatizer()

lemmatized_list_of_lists = []
for inner_list in update_tags_paras:
    lemmatized_inner_list = [lem.lemmatize(word, get_wordnet_pos(tag)) for word, tag in inner_list]
    lemmatized_list_of_lists.append(lemmatized_inner_list)

df['Cleaned_Stem_Description'] = stemmed_paras
df['Cleaned_Lemma_Description'] = lemmatized_list_of_lists

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join(df['Cleaned_Stem_Description'].astype(str).tolist()) 

cloud = WordCloud().generate(text)
plt.imshow(cloud)