"""
Train model and save embedded vector
"""

import gensim
from gensim.models import Word2Vec
import numpy as np

def train_word2Vec(tokens_list: list[list[str]],vector_size,window,min_count,skip_gram: bool=True):
    """
    IN : tokens_list : tokens from each tutor text
    """
    model = Word2Vec(sentences=tokens_list, 
                    vector_size=vector_size,  # Vector Dimension
                    window=window,         # Context Window
                    min_count=min_count,      
                    sg=skip_gram)             # 1 = Skip-Gram, 0 = CBOW
    model.wv.fill_norms()
    # name = "tutorinder_w2v_" + ("sg" if skip_gram else "cbow") + ".model"
    # model.save(name)
    return model

def vectorize_single_profile(model, model_type,tokens: list[str]):
    vector_token = []
    if model_type=='glove':
        for token in tokens:
            if token not in model.keys():
                pass
            else:
                vector_token.append(model[token])
        if not vector_token:
            return np.zeros(model.vector_size)
        return np.array(vector_token).mean(axis=0)
    else:
        for token in tokens:
            if token not in model.wv:
                pass
            else:
                vector_token.append(model.wv[token])
        if not vector_token:
            return np.zeros(model.vector_size)
        return np.array(vector_token).mean(axis=0)


def vectorize_multiple_profile(model, model_type,tokens_list: list[list[str]],save: bool = True):
    list_vector = [] 
    for tokens in tokens_list:
        vector = vectorize_single_profile(model,model_type,tokens)
        list_vector.append(vector)
    array_vector = np.array(list_vector)
    norms = np.linalg.norm(array_vector,axis=1, keepdims=True)
    array_vector_normalized = np.divide(array_vector,norms,out=np.zeros_like(array_vector), where=norms!=0)
    # if save:
    #     np.save("tutor_profiles.npy",array_vector_normalized)
    return array_vector_normalized


def find_similar_words(model, word):
    return model.wv.most_similar(word, topn=10) 