import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tfidfVectorizer import vectorize_test, normalize_test

def rf_select_best_words(X_text_tr_rs, text_val):
    '''
    Selection of best word predictors using RandomForestClassifier
    Input: imblearn sampled training X_text, validation text set
    Output: df1 dataframe of train set words, df2 dataframe of validation set words,
    fitted vectorizer, fitted scaler, idx indices of words
    '''
    X_tr_vector, X_val_vector, df1, df2, vectorizer = vectorize_test(X_text_tr_rs, text_val)
    X_tr_vector_scaled, X_val_vector_scaled, scaler = normalize_test(X_tr_vector, X_val_vector)

    rf = RandomForestClassifier(max_depth=7)
    rf.fit(X_tr_vector_scaled, y_text_tr_rs)

    #pick out the indexes of the 25 words with the highest feature importances
    enum_list = []
    for i, feat_imp in enumerate(rf.feature_importances_):
        enum_list.append((feat_imp, i))

    idx = []
    for feat_ind_pair in sorted(enum_list, reverse=True)[:25]:
        idx.append(feat_ind_pair[1])

    df1 = df1.iloc[:, idx]
    df2 = df2.iloc[:, idx]

    return df1, df2, vectorizer, scaler, idx

def lr_select_best_words(X_text_tr_rs, text_val):
    '''
    Selection of best word predictors using LogisticRegression
    Input: imblearn sampled training X_text, validation text set
    Output: df1 dataframe of train set words, df2 dataframe of validation set words,
    fitted vectorizer, fitted scaler, idx indices of words
    '''
    X_tr_vector, X_val_vector, df1, df2, vectorizer = vectorize_test(X_text_tr_rs, text_val)
    X_tr_vector_scaled, X_val_vector_scaled, scaler = normalize_test(X_tr_vector, X_val_vector)


    lr = LogisticRegression(penalty='l1', solver='liblinear', C=.2)
    lr.fit(X_tr_vector_scaled, y_text_tr_rs)

    idx = np.where(lr.coef_!=0)[1]
    word_coefs = lr.coef_[idx]
    df1 = df1.iloc[:, idx]
    df2 = df2.iloc[:, idx]

    return df1, df2, word_coefs, vectorizer, scaler, idx
