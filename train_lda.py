from gensim import corpora, models
from dotenv import load_dotenv
import os
import spacy

load_dotenv()

# initialise spacy
nlp = spacy.load("en_core_web_sm")

clean_docs_filepath = os.environ['PATH_TO_CLEAN_DOCS']

def preprocess_and_tokenize(doc_text: str):
    # Tokenize, lemmatize, and remove stop words
    doc_nlp = nlp(doc_text)
    token_list = [
        token.lemma_.lower() for token in doc_nlp
        if not token.is_stop
        and token.is_alpha  # Only keep alphanumeric characters
        and len(token.orth_) > 2  # Only keep words of 3 or more characters
        and nlp.vocab[token.orth_].is_stop == False  # Remove common stop words
    ]
    return token_list


# create LDA dict
dictionary = corpora.Dictionary()

preprocessed_data = []

# Preprocessing and dictionary creation
for file in os.listdir(clean_docs_filepath):
    with open(os.path.join(clean_docs_filepath, file)) as f:
        # read and preprocess doc
        doc_token_list = preprocess_and_tokenize(f.read())
        preprocessed_data.append(doc_token_list)
        # add to dictionary
        dictionary.add_documents([doc_token_list])

# save dictionary
dictionary.save('lda_dictionary.dict')

# Create the Bag-of-Words model for each document i.e for each document we create a dictionary reporting how many
# words and how many times those words appear. Save this to 'corpus'
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_data]

# save the corpus
corpora.MmCorpus.serialize('corpus.mm', corpus)

# train lda
lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, passes=10)

# save the model
lda_model.save('lda_model.gensim')

# print topics for visual inspection
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)
