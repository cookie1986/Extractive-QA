from typing import List
import numpy as np
from transformers import pipeline

model_checkpoint = "distilbert-base-cased-distilled-squad"
question_answer = pipeline("question-answering", model=model_checkpoint)

def extract_answer(question: str, context_docs: List):

    # extract answers for each document in context list
    answers = []
    for doc in context_docs:
        answers.append(question_answer(question=question, context=doc))
    # print(answers)
    answer_confidence_scores = [ans['score'] for ans in answers]
    
    # find the answer with the highest score
    top_answer_index = np.argsort(answer_confidence_scores)[::-1][0]
    
    # extract the answer to the doc with the highest score
    answer = answers[top_answer_index]['answer']

    return answer


# contexts = ['asthma is a disease', 'red is my favourite colour', 'tomorrow is saturday']

# print(extract_answer('what is asthma?', contexts))