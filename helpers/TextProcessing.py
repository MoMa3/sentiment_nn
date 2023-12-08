import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub('[^\w\s]', '', text)
    text = [word for word in text.split() if word not in stop_words]
    text = [stemmer.stem(w) for w in text]
    return text