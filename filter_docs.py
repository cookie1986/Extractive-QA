from gensim import corpora, models
import os
from dotenv import load_dotenv
import spacy

load_dotenv()

clean_docs_filepath = os.environ['PATH_TO_CLEAN_DOCS']
filtered_docs_filepath = os.environ['PATH_TO_FILTERED_DOCS']

# load the trained model, dictionary, and corpus
lda_model = models.LdaModel.load('lda_model.gensim')
dictionary = corpora.Dictionary.load('lda_dictionary.dict')
corpus = corpora.MmCorpus('corpus.mm')

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    # Tokenize, remove stop words and lemmatize the text
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]


# Determine which topics are relevant
relevant_topics = {0, 1, 3}

filtered_docs = []
for file in os.listdir(clean_docs_filepath):
    with open(os.path.join(clean_docs_filepath, file)) as f:
        doc = f.read()
    
    # preprocess the document
    doc_bow = dictionary.doc2bow(preprocess(doc))

    # get the topics for this document
    doc_topics = lda_model.get_document_topics(doc_bow)

    # check if any of the document's main topics are in the relevant_topics set
    doc_main_topics = {t[0] for t in doc_topics if t[1] > 0.2}  # adjust the threshold as needed
    if doc_main_topics & relevant_topics:
        # add the document to the list of filtered documents
        filtered_docs.append(doc)
    

# write filtered documents to the output directory
for i, doc in enumerate(filtered_docs):
    with open(os.path.join(filtered_docs_filepath, f'doc_{i}.md'), 'w') as f:
        f.write(doc)