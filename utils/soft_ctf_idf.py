import numpy as np
import scipy.sparse as spr


def soft_ctf_idf(
    doc_topic_matrix: np.ndarray, doc_term_matrix: spr.csr_array
) -> np.ndarray:
    term_importance = doc_topic_matrix.T @ doc_term_matrix
    overall_in_topic = term_importance.sum(axis=1)
    n_docs = len(doc_topic_matrix)
    tf = (term_importance.T / overall_in_topic).T
    idf = np.log(n_docs / (np.abs(term_importance).sum(axis=0)))
    ctf_idf = tf * idf
    return ctf_idf
