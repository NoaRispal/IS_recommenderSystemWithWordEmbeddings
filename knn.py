from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class KNN:
    def __init__(self,base_vector):
        self.base_vector = base_vector

    def find_nn(self,v,n_neighbours):
        vector_array = np.array(v)
        sims = np.dot(self.base_vector, vector_array)
        top_indices = np.argsort(sims)[::-1][:n_neighbours]
        top_sims = sims[top_indices]
        
        return top_indices, top_sims
    
    def find_nn_threshold(self,v,threshold):
        ## To Add: Save idx to find the vector in the profile corpus
        sims = []
        for vector in self.base_vector:
            sim = cosine_similarity(v,vector)
            if sim <= threshold: 
                sims.append(sim)
        return sims