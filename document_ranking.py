import os
from dotenv import load_dotenv
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
doc_filepath = os.environ['PATH_TO_CLEAN_DOCS']

# load the vectorizer and the vectorized documents
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)


# find the top n most relavent documents based on an input query - i.e., a question from Brisa
def get_top_n_docs(query: str, n: int = 3):

    # store cleaned doc filenames in a list
    doc_filenames = [f for f in os.listdir(doc_filepath) if os.path.isfile(os.path.join(doc_filepath, f))]

    top_n_docs = []

    # compute the TF-IDF score for the query
    query_vector = vectorizer.transform([query])

    # compute the cosine similarity between the query and each document
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # rank the documents based on their cosine similarity scores
    ranking = np.argsort(cosine_similarities[0])[::-1][:n]

    # get the top n documents
    for i in range(n):
        idx = ranking[i]
        file_path = os.path.join(doc_filepath, doc_filenames[idx])
        with open(file_path, 'r') as file:
            content = file.read()
            top_n_docs.append(content)
    
    return top_n_docs


# top_docs = get_top_n_docs('what is asthma?')
# print(len(top_docs))
# print(type(top_docs))
# print(top_docs[0])