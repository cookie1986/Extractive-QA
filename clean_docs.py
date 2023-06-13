import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
documents_filepath = os.environ['PATH_TO_DOCS']
cleaned_documents_filepath = os.environ['PATH_TO_CLEAN_DOCS']
markdown_files = [f for f in os.listdir(documents_filepath) if f.endswith('.md')]

# lines with any of the below are deleted
remove_list = ['donate', 'cookies', 'forbidden', 'search', 'browse', 'page', 'www', 'http', 
               'copyright', 'registered charity', 'vat number', 'menu'] 
remove_pattern = '|'.join([f'.*{word}.*\n' for word in remove_list])

for file in markdown_files:

    with open(f"{documents_filepath}/{file}", 'r') as f:
        md_text = f.read()
    print(file)

    # remove HTML tags and URLs
    clean_text = re.sub(r'\[.*?\]', '', md_text)
    clean_text = re.sub(r'\(.*?\)', '', clean_text)

    # header/Footer removal - depends on what you see as headers and footers.
    clean_text = clean_text.replace('Main menu', '')

    # irrelevant text removal
    clean_text = re.sub(remove_pattern, '', clean_text, flags = re.I)
    # lowercasing
    clean_text = clean_text.lower()
    # special characters removal
    clean_text = re.sub(r'[^\w\s]', '', clean_text)

    # remove excessive newlines
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    # remove short lines
    lines = clean_text.split("\n")  # split the text into lines
    lines = [line for line in lines if len(line.split()) > 2]
    clean_text = "\n".join(lines)

    # remove .md file extension
    file_no_md = file[:-3]
    with open(f"{cleaned_documents_filepath}/{file_no_md}.txt", "w") as f:
        f.write(clean_text)