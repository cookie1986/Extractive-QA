import document_ranking
import question_answer

test_query = 'What is asthma?'

top_docs = document_ranking.get_top_n_docs(test_query, n=3)

print(question_answer.extract_answer(test_query, top_docs))