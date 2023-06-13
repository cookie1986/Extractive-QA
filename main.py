import document_ranking
import question_answer

import os
from dotenv import load_dotenv
import openai

load_dotenv()

OPENAI_KEY = os.environ['OPENAI_KEY']
openai.api_key = OPENAI_KEY

# test_query = 'What is asthma?' # ANSWER: Lung condition
# test_query = 'How serious is asthma?' # ANSWER: Life
test_query = 'What can I do to prevent an asthma attack?' # ANSWER: Tests and treatments
# test_query = 'What do i do if I have an asthma attack?' # ANSWER: Find your MLA (Medical Licence Assessment)
# test_query = 'What inhaler do I use to prevent an asthma attack?' # ANSWER: Hansard (???)
# test_query = 'How can I improve my asthma symptoms?' # ANSWER: campaigning for better lung health
# test_query = 'What do I do if I lose my inhaler?' # ANSWER: Locate your MLA
# test_query = 'What symptoms make asthma worse?' # ANSWER: symptoms-tests-treatments


top_docs = document_ranking.get_top_n_docs(test_query, n=3)

answer = question_answer.extract_answer(test_query, top_docs)

prompt = f'given the question "{test_query}" and the answer "{answer}", give me a more conversationally friendly answer.'


processed_answer = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=35,
    n=1,
    stop=None,
    temperature=0.5,
)

print(processed_answer["choices"][0]["text"])
