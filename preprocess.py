"""
Tokenize and clean input data
"""

import nltk
import string

def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        # On ne télécharge QUE si ce n'est pas trouvé
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)

setup_nltk()

def preprocess(data: str) -> list[str]:
    data.lower()
    data = data.translate(str.maketrans("","",string.punctuation))
    tokens = tokenize(data)
    cleaned_tokens = clean_token(tokens,'english')
    return cleaned_tokens


def tokenize(sentence):
    return nltk.tokenize.word_tokenize(sentence)

def clean_token(tokens,language: str):
    """ 
        IN : tokens -> Each word of a sentence
            language -> Language of the data
        OUT : clean_tokens -> Only meaningful word of a sentence 
    """
    stop_words = set(nltk.corpus.stopwords.words(language))
    cleaned_tokens = [w for w in tokens if w not in stop_words]
    return cleaned_tokens
    
    