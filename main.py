#! /usr/bin/env python3

import pandas as pd
import numpy as np
from recommender import ContentBasedFiltering, CollaborativeFiltering, HybridRecommender
from user import Tutoree
import os
import sys


def collect_user_data():
    print("--- Tutor form ---")

    full_name = input("Your Name : ")
    email = input("Your email (Since it's a demo you can enter nothing): ")
    location = input("Your location (Since it's a demo you can enter nothing): ")

    query = input("Your query : ")

    subjects = ["Computer Science", "Mathematics", "Physics", "Humanities", "Geopolitics", "Biology", "Business", "All"]
    subject = ""
    while subject not in subjects:
        print(f"Subjects : {', '.join(subjects)}")
        subject = input("Please choose a subject : ")

    price = -1
    while price < 0:
        try:
            price = float(input("Price ($/h) : "))
        except ValueError:
            print("Please enter a valid number.")

    necessities_list = ["ADHD", "Dyslexia", "Autism"]
    selected_necessities = []
    print(f"Special needs : {', '.join(necessities_list)}")
    user_ne = input("Enter special needs you may separated by a comma (e.g.: ADHD, Autism) : ")
    selected_necessities = [n.strip() for n in user_ne.split(",") if n.strip() in necessities_list]
    selected_necessities = ",".join(selected_necessities)

    modes = ["Online", "Physical", "Both"]
    mode = ""
    while mode not in modes:
        mode = input(f"Preferred learning mode ({'/'.join(modes)}) : ")

    levels = ["highschool", "undergraduate", "graduate", "phd"]
    level = ""
    while level not in levels:
        print(f"Level : {', '.join(levels)}")
        level = input("Your level : ").lower()

    print("\n--- Summary of your request ---")
    data = {
        "fullname" : full_name,
        "email" : email,
        "location" : location,
        "query": query,
        "subject": subject,
        "price": f"{price}$/h",
        "needs": selected_necessities,
        "mode": mode,
        "level": level
    }
    
    for key, value in data.items():
        print(f"{key}: {value}")
    print("\n-----------------------------")
    return data

def main(data,weights={'cb': 0.9, 'cf': 0.1},glove=False):
    DATA_DIR = "data/training"
    TUTORS_CSV = f"{DATA_DIR}/tutors.csv"
    RATINGS_CSV = f"{DATA_DIR}/ratings.csv"
    
    MODEL_W2V = "models/word2vec/tutor_w2v.model"
    PROFILES_NPY = "models/word2vec/tutor_profiles.npy"
    
    # print("--- Initializing Content-Based Filtering ---")
    cbf = ContentBasedFiltering()
    
    if not os.path.exists(TUTORS_CSV):
        print(f"Error : {TUTORS_CSV} not found.")
        return

    os.makedirs("models", exist_ok=True)
    if glove:
        if glove_path:
            cbf.load_glove(TUTORS_CSV,glove_path)
        else: 
            cbf.load_glove(TUTORS_CSV,"models/glove/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt")
        np.save("models/glove/tutor_profiles_glove", cbf.tutor_profiles)
    else: 
        cbf.train(TUTORS_CSV)
        ## optional
        cbf.model.save(MODEL_W2V)
        np.save(PROFILES_NPY, cbf.tutor_profiles)
        ##


    # print("\n--- Initializing Collaborative Filtering (SVD) ---")
    cf = CollaborativeFiltering()
    
    if os.path.exists(RATINGS_CSV):
        cf.train("data/training/ratings.csv", "data/training/tutors.csv")
        cf.save_ressources("models/svd")
    else:
        print(f"Attention : {RATINGS_CSV} not found. CF will not be able to work.")

    hybrid_system = HybridRecommender(cbf, cf)

    test_user = Tutoree(10,data["fullname"],data["price"],data["subject"],data["level"],data["mode"],data["needs"],data["location"],data["email"], data["query"])

    print(f"\n--- Testing Recommendation for User : {test_user.fullname} ---")
    print(f"Query: '{test_user.query}'\n")

    # CB
    # print("Results [Content-Based Only]:")
    weights_cb = {
        "similarity": 0.5,    # bio
        "subject": 0.35,       
        "level": 0.025,         
        "hourly_rate": 0.05,  
        "special_needs": 0.025,
        "preferred_learning_mode": 0.05
    }
    # print("---------------------------------------------")
    # print("Weights for ContentBasedFiltering : \n", weights_cb)
    # print("---------------------------------------------")
    # cb_results_indices = cbf.inference(test_user, weights_cb)
    # for i in cb_results_indices[:3]:
    #     tutor = cbf.df.iloc[i]
    #     print(f"- {tutor['fullname']} | Domain: {tutor['subject']} | Bio: {tutor['bio'][:50]}...")
    # Hybrid (CB + CF)
    print(f"\nResults [Hybrid CB {weights['cb']*100}% / CF {weights['cf']*100}%]:")
    hybrid_results = hybrid_system.inference(test_user, weights,cb_weights=weights_cb, n_recommendations=5)
    
    for tutor_obj in hybrid_results:
        print(f"- {tutor_obj.fullname} \n\t -> Domain: {tutor_obj.subject} \n\t -> Precisely: {tutor_obj.precise_domain} \n\t -> Bio: {tutor_obj.bio} \n\t -> Price: {tutor_obj.hourly_rate}  \n\t -> Cover needs: {tutor_obj.special_needs}" )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        glove_path = sys.argv[1]
    load_data = input("Do you want to create your persona or to load one ? [create/load] : ")
    glove = input("Do you want use embedder GloVe (better but longer) ? [yes/no] : ")
    glove = True if glove=="yes" else False
    
    if load_data == "create":
        data = collect_user_data()
    else : 
        data = {
        "fullname" : "Noa Rispal",
        "email" : "None",
        "location" : "None",
        "query": "I want to learn optimizer for Neural Networks such as Adam Optimizer",
        "subject": "Computer Science",
        "price": 60,
        "needs": "",
        "mode": "Online",
        "level": "undergraduate"
        }
        print("\n--- Summary of your request ---")
        for key, value in data.items():
            print(f"{key}: {value}")
        print("\n-------------------------------")
    print(f"\nData collected successfully !\n")
    print(f"Now here's the model prediction : ")
    main(data,glove=glove)
    swipe = input("\nRetry with more serendipity ? [y/n] : ")
    if swipe=='y': 
        weights={'cb': 0.7, 'cf': 0.3}
        main(data,weights,glove=glove)
        swipe = input("\nRetry again with more serendipity ? [y/n] : ")
        if swipe=='y': 
            weights={'cb': 0.5, 'cf': 0.5}
            main(data,weights,glove=glove)
            if swipe=='y': 
                swipe = input("\nOne more time with more serendipity ? [y/n] : ")
                weights={'cb': 0.3, 'cf': 0.7}
                main(data,weights,glove=glove)
