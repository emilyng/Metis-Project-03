import numpy as np
from sklearn.linear_model import LogisticRegression
from TD_IDF_Vectorizer import vectorize_test, normalize_test
from imb_undersample import imb_undersample

def optimal_c(X_text_tr, y_text_tr):
    train_scores = []
    val_scores = []
    text_vec_lens = []

    X_text_tr_rs, y_text_tr_rs, _ = imb_undersample(X_text_tr, y_text_tr)
    X_tr_vector, X_val_vector, df = vectorize_test(X_text_tr_rs, X_text_val)
    X_tr_vector_scaled, X_val_vector_scaled = normalize_test(X_tr_vector, X_val_vector)

    c_ranges = np.arange(0.1, 1, 0.05)

    for c in c_ranges:


        lr = LogisticRegression(penalty='l1', solver='liblinear', C=c)
        lr.fit(X_tr_vector_scaled, y_text_tr_rs)
        y_preds = lr.predict(X_val_vector_scaled)

        train_score = lr.score(X_tr_vector_scaled, y_text_tr_rs)
        val_score = lr.score(X_val_vector_scaled, y_text_val)

        train_scores.append((c, train_score))
        val_scores.append((c, val_score))
        text_vec_lens.append(len(np.where(lr.coef_!=0)[1]))

    best_c_index = np.where([lens<35 for lens in text_vec_lens])[0][-1]
    best_c = c_ranges[best_c_index]
    return best_c
