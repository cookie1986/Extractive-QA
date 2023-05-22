# Extractive-Question Answering for Asthma
Generating Answers from the Pre-trained Language Model

This notebook uses a pretrained LM, fine-tuned on our annotated asthma data, and uses it to answer a question that has not formed part of the training (i.e., it is out-of-sample).
To generate an answer, it needs the question, and a corpus of unlabelled documents. Using Information Retrievel (IR), it will rank order the documents, and search for an answer within the top K most relevant, where K is an integer that is pre-specified by the user.Generating Answers from the Pre-trained Language Model

This notebook uses a pretrained LM, fine-tuned on our annotated asthma data, and uses it to answer a question that has not formed part of the training (i.e., it is out-of-sample).
To generate an answer, it needs the question, and a corpus of unlabelled documents. Using Information Retrievel (IR), it will rank order the documents, and search for an answer within the top N most relevant, where N is an integer that is pre-specified by the user.

The extracted documents are then concatenated into a single document that is used as the 'context' for the trained model. Extractive QA works by finding the answer within the most relevant segment of text in the context document. This is done in several stages:

1. The model encodes the question, which captures its semantics, named entities, and dependency relations.
2. Next, the model encodes the document, breaking it up into chunks if it exceeds a certain size (with overlap to prevent losing information).
3. The model then aligns the question with each context chunk, calculating the probability that each word token is the start or end of the answer.
4. This step is repeated over all chunks, with the start/end of the answer determined by the tokens with the highest probability.
5. The answer is then extracted based on these start/end tokens, and returned to the user (with a certain confidence level)
6. As an optional step, a threshold can be set that enables the model to abstain if the associated confidence score is low.

Some things to be aware of:
* This approach does not perform any post-processing on the extracted answer. Post-processing steps such as answer verification or adjusting politeness and tone can be used to improve the suitability of the answer.
* Increasing the value of K during IR, using particularly (non-distilled) LMs, and having a large overlap during the chunking phase will all impact on runtime.
* This approach can be run without fine-tuning the model (i.e., zero-shot), but it will likely not perform as well.

Measuring the quality of answers can be done via objective metrics (i.e., F1 where answers are known), or using subjective measures (i.e., Likert scores).

## Summary
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