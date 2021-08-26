from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def similarity(sent1, sent2):
    def cos_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
        similarity = dot_product / l2_norm     
        return similarity

    text = [sent1, sent2]
    tfidf_vect_simple = TfidfVectorizer()
    feature_vect_simple = tfidf_vect_simple.fit_transform(text)
    feature_vect_dense = feature_vect_simple.todense()
    vect1 = np.array(feature_vect_dense[0]).reshape(-1,)
    vect2 = np.array(feature_vect_dense[1]).reshape(-1,)
    similarity_simple = cos_similarity(vect1, vect2)
    return similarity_simple