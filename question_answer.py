from typing import List
import numpy as np
from transformers import pipeline
import paraphrase

model_checkpoint = "deepset/tinyroberta-squad2"
question_answer = pipeline("question-answering", model=model_checkpoint)

def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

def extract_answer(question: str, context_docs: List):

    # extract answers for each document in context list
    answers = []
    for doc in context_docs:
        answers.append(question_answer(question=question, context=doc))
    answer_confidence_scores = [ans['score'] for ans in answers]
    
    # normalise the scores so they're probabilities
    normalized_scores = softmax(answer_confidence_scores)

    # set a threshold
    threshold = 0.4

    # check if the highest score is above the threshold
    max_score = np.max(normalized_scores)
    print(f"Confidence: {max_score}")
    if max_score <= threshold:
        answer = paraphrase.extract_and_paraphrase_answers(question, context_docs)
        return answer
    
    # find the answer with the highest score
    top_answer_index = np.argsort(answer_confidence_scores)[::-1][0]
    
    # extract the answer to the doc with the highest score
    answer = answers[top_answer_index]['answer']
    
    # post-processing - check if the answer is part of a sentence.
    start_idx = context_docs[top_answer_index].find(answer)

    # Find the beginning of the sentence
    sentence_start = context_docs[top_answer_index].rfind('.', 0, start_idx) + 2
    # Adjust if at the beginning of the document
    if sentence_start == 1:
        sentence_start = 0
    # find the end of the sentence
    sentence_end = context_docs[top_answer_index].find('.', start_idx) + 1

    # extract the full sentence containing the answer
    full_sentence = context_docs[top_answer_index][sentence_start:sentence_end].strip()

    return full_sentence