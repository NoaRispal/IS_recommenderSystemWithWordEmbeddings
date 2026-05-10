# Tutorinder

Tutorinder is a hybrid tutor recommendation system combining **Word Embeddings**, **Semantic Search**, and **Recommender Systems** techniques to create personalized tutor recommendations for students.

The project was developed as part of a research work on:

* Natural Language Processing (NLP)
* Word Embeddings
* Recommender Systems
* Hybrid Recommendation Architectures

The main goal of Tutorinder is to improve tutor discovery by moving beyond simple keyword filtering and instead using semantic understanding and collaborative recommendation methods.

---

# Getting Started

Make sure to have all the requirements, run:
```bash
pip install -r requirements.txt
```

To start the project, run:

```bash
python main.py glove_path
```

When launching the application, the user has two possible choices:

* Create a new persona
* Load an existing persona

## Create a Persona

To create a new persona, type:

```text
create
```

A series of questions will then appear.
For each question, the user must write the desired answer among the proposed choices.

The system will use these answers to build a personalized user profile.

N.B. : This project is not intended to fully simulate real-world production environments. Therefore, user inputs are not necessarily validated or corrected.
If the input differs from the proposed choices (spelling mistakes, uppercase/lowercase differences, formatting errors, etc.), the script may fail.

---

## Load an Existing Persona

To load an already existing persona, type:

```text
load
```

The application will then retrieve the saved persona and use it directly inside the recommendation pipeline.

---

## Choose embedder model

The user also needs to choose the model to embed words : 

* hand-made Word2Vec model trained on bio of tutors datasets
* Use [GloVe](https://nlp.stanford.edu/projects/glove/) an unsupervised learning algorithm for obtaining vector representations for words. In this case, you need to download [glove.2024.wikigiga.100d.zip](https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip) from their website and to place it in :
```text
./models/glove/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt
```
Or give the path to the file with **glove_path** parameter when running the command

# Problem Statement

Modern educational platforms contain hundreds of tutor profiles with different:

* subjects,
* teaching styles,
* prices,
* educational levels,
* and specializations.

Finding the correct tutor can therefore become difficult for students.
Traditional search systems rely mostly on exact keyword matching and filtering, which creates several problems:

* semantic mismatch between student queries and tutor profiles,
* inability to understand context,
* poor personalization,
* and cold start issues.

Tutorinder was designed to solve these limitations using:

* Word Embeddings for semantic understanding,
* Matrix Factorization for collaborative filtering,
* and Hybrid Scoring for ranking.

---

# Main Features

## Semantic Tutor Search

Tutorinder uses Word Embeddings to transform tutor biographies and student queries into dense vectors.

This allows the system to:

* understand semantic similarity,
* detect related meanings,
* retrieve tutors even when different words are used,
* and improve recommendation quality.

Example:

A student searching for:

> "Need help in advanced mathematics and machine learning"

may still retrieve tutors mentioning:

* artificial intelligence,
* deep learning,
* statistics,
* or data science.

---

## Hybrid Recommendation System

Tutorinder combines:

### Content-Based Filtering (CBF)

Using:

* tutor biographies,
* embeddings,
* cosine similarity,
* semantic search.

### Collaborative Filtering (CF)

Using:

* user-tutor interactions,
* ratings,
* matrix factorization,
* latent features.

The final recommendation score combines both systems.

---

## Two-Stage Recommendation Pipeline

The recommendation process is divided into two phases:

### 1. Candidate Generation

A fast retrieval stage using:

* embeddings,
* cosine similarity,
* K-Nearest Neighbors (KNN).

This stage retrieves only the most relevant tutors.

### 2. Ranking

A second scoring stage combining:

* semantic similarity,
* collaborative filtering scores,
* tutor metadata,
* weighted scoring.

This architecture improves:

* scalability,
* recommendation quality,
* computational efficiency.

---

# System Architecture

```text
Student Query
      ↓
Preprocessing
(Tokenization + Cleaning)
      ↓
Word Embeddings
      ↓
Mean Pooling
      ↓
Sentence Vector
      ↓
KNN Candidate Generation
      ↓
Hybrid Scoring
(CBF + CF)
      ↓
Ranking
      ↓
Recommended Tutors
```

---

# Technologies Used

## Natural Language Processing

* Word2Vec (Gensim)
* GloVe
* Tokenization (nltk)
* Stopword Removal (nltk)
* Mean Pooling (numpy)
* Cosine Similarity (scipy)

## Machine Learning

* K-Nearest Neighbors (KNN)
* Singular Value Decomposition (SVD)
* Hybrid Recommender Systems

## Programming

* Python
* NumPy
* Pandas
* Scikit-learn
* Gensim


# Hybrid Recommendation Formula

Tutorinder combines semantic similarity and collaborative filtering scores.

```text
FinalScore = x × ContentBasedScore + y × CollaborativeScore
```

(x,y) are the weights of the hybrid model. Usually, you will start with a high x because you want the results to perfectly match the student query but the more he swipes the more we increase serendipity (i.e. y)

# Cold Start Problem

One of the main challenges in recommender systems is the Cold Start Problem.

New users and new tutors often lack interaction data.

Tutorinder mitigates this issue using semantic embeddings:
even without ratings, tutor profiles can still be matched semantically by only using CBF score.

This is one of the main advantages of the hybrid architecture.

---

# Authors

Developed as an academic project on for Intelligent Systems class (HCMUT) by Noa RISPAL:


