from dotenv import load_dotenv
import os

load_dotenv()

clean_docs_filepath = os.environ['PATH_TO_CLEAN_DOCS']

docs = []

for file in os.listdir(clean_docs_filepath)[:200]:
    with open(os.path.join(clean_docs_filepath, file), encoding="utf-8") as f:
        doc = f.read()
        print(f'----------------{file}----------------')
        print(doc[:20])
        print(' ')
        print(' ')