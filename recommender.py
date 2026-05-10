import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from abc import ABC,abstractmethod
import pandas as pd 
from scipy.sparse.linalg import svds
import os
import pickle

import embedder
import preprocess
from knn import KNN
from user import *
from utils import normalize



class RecommenderSystem(ABC):
    def __init__(self):
        self.model = None
        self.df = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load_ressources(self):
        pass

    @abstractmethod
    def inference(self):
        pass



    
class ContentBasedFiltering(RecommenderSystem):
    def __init__(self):
        super().__init__()
        self.tutor_profiles = None
        self.model_type = None

    def train(self,data_path,vector_size=100,min_count=1,window=5,skip_gram:bool = True):
        self.df = pd.read_csv(data_path)
        bio = self.df["bio"]
        tokens_list=[preprocess.preprocess(b) for b in bio]
        # Train model
        self.model = embedder.train_word2Vec(tokens_list,vector_size,window,min_count,skip_gram)
        self.model_type = "word2vec"
        # Vectorize tutor
        self.tutor_profiles = embedder.vectorize_multiple_profile(self.model,self.model_type,tokens_list)

    def load_ressources(self,model_path,tutor_profiles_path,data_path):
        self.model = Word2Vec.load(model_path)
        self.model_type = "word2vec"
        self.tutor_profiles = np.load(tutor_profiles_path)
        self.df = pd.read_csv(data_path)

    def load_glove(self, data_path, glove_path):
        self.df = pd.read_csv(data_path)

        bio = self.df["bio"]
        tokens_list = [preprocess.preprocess(b) for b in bio]
        self.model = {}

        print(f"Loading GloVe...\n")
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()

                word = values[0]
                try :
                    vector = np.asarray(values[1:], dtype=np.float32)
                    self.model[word] = vector
                except ValueError:
                    continue

        print(f"GloVe loaded with {len(self.model)} words.\n")
        self.model_type = "glove"

        # Vectorize tutor profiles
        self.tutor_profiles = embedder.vectorize_multiple_profile(self.model,self.model_type,tokens_list)

    def inference(self, user: Tutoree, weights, return_scores=False):
        if self.model is None or self.tutor_profiles is None or self.df is None:
            raise ValueError("Resources not loaded.")
        
        user_vector = self.process_query(user.query)
        
        similarities = np.dot(self.tutor_profiles, user_vector)

        # You may use KNN here between tutor_profiles and user_vector
        # knn = KNN(self.tutor_profiles) 
        # top_idx,top_sims = knn.find_nn(user.query,n_neighbours=100)

        raw_prices = self.df['hourly_rate'].values.reshape(-1, 1)
        norm_prices = self.normalize_features(raw_prices, ["hourly_rate"])

        user_needs = self.parse_special_needs(user.special_needs)
        final_scores = np.zeros(len(self.df))

        for i in range(len(self.df)):
            tutor_row = self.df.iloc[i]
            tutor_obj = Tutor(**tutor_row.to_dict())
            
            score = weights["similarity"] * similarities[i]
            
            # Subject match
            if tutor_obj.subject.strip().lower() == user.subject.strip().lower():
                score += weights["subject"]
            
            # Special Needs
            if len(user_needs) > 0:
                tutor_needs = self.parse_special_needs(tutor_obj.special_needs)
                matching = user_needs.intersection(tutor_needs)
                if matching:
                    score += weights["special_needs"] * (len(matching) / len(user_needs))
                else:
                    score -= 0.1

            # Mode 
            if user.preferred_learning_mode.strip().lower() == tutor_obj.preferred_learning_mode.strip().lower():
                score += weights["preferred_learning_mode"]
            
            # Level distance
            level_dist = User.get_level_distance(user, tutor_obj)
            if User.compare_level(user, tutor_obj):
                score += weights["level"] * (1 / (1 + level_dist)) # Bonus dégressif
            else:
                score -= weights["level"] * level_dist

            # Price
            score += weights["hourly_rate"] * norm_prices[i][0]
            
            final_scores[i] = score

        if return_scores:
            return final_scores

        # Rank
        top_indices = np.argsort(final_scores)[::-1][:10]
        return top_indices
    
    def parse_special_needs(self,value):
        if pd.isna(value) or value == "" or value == "none":
            return set()
        return set([item.strip().lower() for item in str(value).split(",")])

    def normalize_features(self,data: np.array, feature_names: list) -> np.array:
        """
        Normalise data (reverse columns where 'lower is better' (ex: price).
        """
        normalized = normalize(data)
        
        for i, name in enumerate(feature_names):
            if "hourly_rate" in name.lower():
                # Lower value -> Higher after normalization
                normalized[:, i] = 1.0 - normalized[:, i]
                
        return normalized
        
    def process_query(self,text):
        cleaned_token = preprocess.preprocess(text)
        user_vector = embedder.vectorize_single_profile(self.model,self.model_type,cleaned_token)
        return user_vector
    



    
class CollaborativeFiltering(RecommenderSystem):
    def __init__(self):
        super().__init__()
        self.user_features = None  
        self.tutor_features = None 
        self.user_mapping = None   
        self.ratings_mean = None  

    def train(self, ratings_path, data_path, n_factors=20):
        """
        IN: ratings_path : CSV [student_id, tutor_id, rating]
            data_path : CSV [id, name, bio, ...] (ton fichier tuteurs)
        """
        ratings_df = pd.read_csv(ratings_path)
        self.df = pd.read_csv(data_path)
        
        pivot_table = ratings_df.pivot(index='student_id', columns='tutor_id', values='rating')
        
        # /!\ IMPORTANT : The columns of the SVD matrix are realigned according to the order of the IDs.
        # of the tutors.csv file so that index 0 of the model is index 0 of the CSV.
        pivot_table = pivot_table.reindex(columns=self.df['id'].values).fillna(0)
        
        self.user_mapping = list(pivot_table.index)

        R = pivot_table.values
        self.ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - self.ratings_mean.reshape(-1, 1)

        # SVD Decomposition
        U, sigma, Vt = svds(R_demeaned, k=n_factors)
        
        self.user_features = np.dot(U, np.diag(sigma))
        self.tutor_features = Vt

    def save_ressources(self, folder_path="models/svd"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        np.save(f"{folder_path}/user_features.npy", self.user_features)
        np.save(f"{folder_path}/tutor_features.npy", self.tutor_features)
        np.save(f"{folder_path}/ratings_mean.npy", self.ratings_mean)
        # mapping is a list, we use pickle
        with open(f"{folder_path}/user_mapping.pkl", "wb") as f:
            pickle.dump(self.user_mapping, f)

    def load_ressources(self, folder_path, data_path):
        self.user_features = np.load(f"{folder_path}/user_features.npy")
        self.tutor_features = np.load(f"{folder_path}/tutor_features.npy")
        self.ratings_mean = np.load(f"{folder_path}/ratings_mean.npy")
        with open(f"{folder_path}/user_mapping.pkl", "rb") as f:
            self.user_mapping = pickle.load(f)
        self.df = pd.read_csv(data_path)

    def inference(self, user_id, n_recommendations=10, return_scores=False):
        if self.user_features is None or self.user_mapping is None:
            raise ValueError("Model components missing.")

        try:
            user_idx = self.user_mapping.index(user_id)
            # Raw predict
            user_prediction = np.dot(self.user_features[user_idx, :], self.tutor_features) + self.ratings_mean[user_idx]
            
            # Normalisation 0-1 for hybrid
            cf_scores = (user_prediction - user_prediction.min()) / (user_prediction.max() - user_prediction.min() + 1e-9)
        except (ValueError, AttributeError):
            cf_scores = np.zeros(len(self.df))

        if return_scores:
            return cf_scores

        sorted_indices = np.argsort(cf_scores)[::-1]
        return Tutor.get_tutor(self.df, sorted_indices[:n_recommendations])
    




class HybridRecommender:
    def __init__(self, cb_model, cf_model):
        """
        IN: 
        cb_model : Instance of ContentBasedFiltering already trained/loaded
        cf_model : Instance of CollaborativeFiltering already trained/loaded
        """
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.df = cb_model.df

    def inference(self, student_user, weights={'cb': 0.6, 'cf': 0.4}, 
                  cb_weights={"similarity": 0.4, "subject": 0.4, "level": 0.05, "hourly_rate": 0.1, "special_needs": 0.05, "preferred_learning_mode": 0.0}, 
                  n_recommendations=10):

        # 1) CB 
        cb_scores = self.cb_model.inference(student_user, weights=cb_weights, return_scores=True)

        # 2) CF
        cf_scores = self.cf_model.inference(student_user.id, return_scores=True)

        # 3) Hybrid
        final_scores = (weights['cb'] * cb_scores) + (weights['cf'] * cf_scores)

        # 4) Tri et renvoi
        top_indices = np.argsort(final_scores)[::-1][:n_recommendations]
        return Tutor.get_tutor(self.df, top_indices)