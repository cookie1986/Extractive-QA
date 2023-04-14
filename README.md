# Extractive-Question Answering for Asthma

This is an extractive Question-Answering (QA) package that performs document ranking using TF-IDF to select the top N most relevant documents, and then performs extractive QA to generate potential answers for the input query. The answer with the highest confidence is returned as the final output.

## Features

* Document ranking using TF-IDF to find the top N relevant documents
* Extractive QA to generate potential answers from each of the top N documents
* Selection of the answer with the highest confidence

## Installation

To install the necessary dependencies, run:

```
pip install -r requirements.txt
```

## Usage

First, prepare a corpus of documents as plain text or Markdown files and place them in a suitable directory.

To use the package, follow these steps:

1. Run the document ranking step to find the top N relevant documents based on the input query:

```
from document_ranking import get_top_n_docs

query = "What is asthma?"
top_n = 3

top_n_docs = get_top_n_docs(query, n=top_n)

```

2. Perform extractive QA on the top N documents to generate potential answers. This will output the answer with the highest confidence score:

```
from question_answer import extract_answer

answers = extract_answer(query, top_n_docs)
```

## Contributing

If you have any suggestions, improvements, or bug reports, please feel free to submit an issue or create a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.