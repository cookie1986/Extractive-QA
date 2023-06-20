import os
import re
import nltk
from dotenv import load_dotenv
import enchant
from nltk.tokenize import word_tokenize
import chardet

load_dotenv()

dictionary = enchant.Dict("en_GB")

english_words = set(nltk.corpus.words.words())

documents_filepath = os.environ['PATH_TO_DOCS']
cleaned_documents_filepath = os.environ['PATH_TO_CLEAN_DOCS']
markdown_files = [f for f in os.listdir(documents_filepath) if f.endswith('.md')]
cleaned_markdown_filepath = os.environ['PATH_TO_CLEAN_MD']
bad_files_filepath = os.environ['PATH_TO_BAD_FILES']

# lines with any of the below are deleted
remove_list = ['donate', 'cookies', 'forbidden', 'search', 'browse', 'page', 'www', 'http', 
               'copyright', 'registered charity', 'vat number', 'menu', 'obj', 'javascript',
               'sign up','united kingdom','newsletter','email us','contact us', '@', 'hide',
               'news','freedom of information','licence', 'call', 'helpline','whatsapp'] 

remove_pattern = '|'.join([f'.*{word}.*\n' for word in remove_list])

# documents without one of these keywords are filtered out
corpus_keywords = ['asthma','lung','inhaler','breath','trigger','air','airway','allergic',
                   'allergy','allergen','peak flow', 'bronchodilator','bronchial provocation',
                   'respiratory', 'saba', 'wheeze','wheezing','pulmonary','steroids','nebulizer',
                   'corticosteriod','albuterol','attack','copd','exercise','seasonal','asthmatic',
                   'prevention','pollution','leukotriene','theophylline','immunotherapy',
                   'montelukast','exacerbation','spirometry','methacholine','eosinophils',
                   'ige antibodies','treatment','shortness','lungs']

def majority_english_words(line):
    '''
    checks the proportion of words in a line that are english language words.

    returns:
        boolean: 
            TRUE, if english words are 50% of all words
            FALSE, otherwise
    '''

    tokens = word_tokenize(line)
    english_words = [token for token in tokens if dictionary.check(token)]
    
    return len(english_words)/len(tokens) >= 0.8


def filter_document(document):
    '''
    splits document into lines and removes lines where english words are less than a 
    predefined threshold (i.e., 0.8)

    returns: document as a list of sentences
    '''
    filtered_lines = []
    for line in document:
        if majority_english_words(line):
            filtered_lines.append(line)
    return filtered_lines

# checks if document contains a keyword
def contains_keyword(doc, keywords=corpus_keywords):
    for keyword in keywords:
        if keyword in doc:
            print(f'KEYWORD: {keyword}')
            return True
    return False

# # empty list for storing processed docs
# processed_docs = []

for file in markdown_files:

    with open(f"{documents_filepath}/{file}", 'r') as f:
        md_text = f.read()
    # predict file encoding
    with open(f"{documents_filepath}/{file}", 'rb') as f:
        # predict encoding and filter out low confidence files
        result = chardet.detect(f.read())
    
    # write the bad file to a seperate location for review
    if result['encoding'] is None or result['confidence'] < 0.8:
        with open(f"{bad_files_filepath}/{file}", "w", encoding='utf-8') as f:
            f.write(md_text)
            continue

    print(file)

    # remove HTML tags and URLs
    clean_text = re.sub(r'\[.*?\]', '', md_text)
    clean_text = re.sub(r'\(.*?\)', '', clean_text)

    # header/footer removal
    clean_text = clean_text.replace('Main menu', '')

    # irrelevant text removal
    clean_text = re.sub(remove_pattern, '', clean_text, flags = re.I)
    # lowercasing
    clean_text = clean_text.lower()
    # special characters removal
    clean_text = re.sub(r'[^\w\s\t.,!?@]', ' ', clean_text)

    # remove excessive newlines
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    # remove short and irrelevant lines
    lines = clean_text.split("\n")  # split the text into lines
    lines = [line for line in lines if len(line.split()) > 3]
    lines = filter_document(lines)
    clean_text = "\n".join(lines)

    # tidy excess whitespace
    clean_text = re.sub('\s+',' ', clean_text)

    # skip blank files
    if len(clean_text) < 1:
        continue

    # check for keyword
    if not contains_keyword:
        continue
    
    with open(f"{cleaned_documents_filepath}/{file}", "w", encoding='utf-8') as f:
        f.write(clean_text)
