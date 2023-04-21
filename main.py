import document_ranking
import question_answer

test_query = 'what is asthma?'

top_docs = document_ranking.get_top_n_docs(test_query, n=2)

print(question_answer.extract_answer(test_query, top_docs))