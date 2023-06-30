from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np
from typing import List

# Initialize the QA pipeline
model_checkpoint = "deepset/tinyroberta-squad2"
question_answer = pipeline("question-answering", model=model_checkpoint)

# Initialize the paraphrasing model
paraphrase_model_name = 'tuner007/pegasus_paraphrase'
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name)
paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)

def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

def extract_and_paraphrase_answers(question: str, context_docs: List):

    # extract answers for each document in context list
    answers = []
    for doc in context_docs:
        answers.append(question_answer(question=question, context=doc))
    answer_confidence_scores = [ans['score'] for ans in answers]
    
    # normalize the scores so they're probabilities
    normalized_scores = softmax(answer_confidence_scores)

    # set a threshold
    threshold = 0.3  # Lower the threshold to allow multiple answers

    # find indices of answers with scores above the threshold
    top_answer_indices = np.where(normalized_scores > threshold)[0]
    
    # Extract the sentences containing the answers
    extracted_sentences = []
    for idx in top_answer_indices:
        answer = answers[idx]['answer']
        start_idx = context_docs[idx].find(answer)

        # Find the beginning of the sentence
        sentence_start = context_docs[idx].rfind('.', 0, start_idx) + 2
        # Adjust if at the beginning of the document
        if sentence_start == 1:
            sentence_start = 0
        # find the end of the sentence
        sentence_end = context_docs[idx].find('.', start_idx) + 1

        # extract the full sentence containing the answer
        full_sentence = context_docs[idx][sentence_start:sentence_end].strip()
        extracted_sentences.append(full_sentence)
    
    # Paraphrase the extracted sentences
    paraphrased_sentences = []
    max_length = paraphrase_model.config.max_position_embeddings
    for sentence in extracted_sentences:
        inputs = paraphrase_tokenizer.encode("paraphrase: " + sentence, return_tensors="pt", max_length=max_length, truncation=True)
        outputs = paraphrase_model.generate(inputs, max_length=150, num_return_sequences=1, num_beams=10)
        paraphrased_sentence = paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)
    
    paraphrase = ' '.join(paraphrased_sentences)
    
    return paraphrase