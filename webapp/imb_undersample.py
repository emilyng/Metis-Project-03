import numpy as np
import imblearn.under_sampling

def imb_undersample(X_tr, y_tr):
    '''
    Imbalance undersampling approach
    Input: training X, training y
    Output: X_tr_rs (resampled training X), y_tr_rs (resampled training y),
    RUS.sample_indices_ (indices of samples taken)
    '''
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    ratio = {1 : n_pos , 0 : n_neg//9}

    RUS = imblearn.under_sampling.RandomUnderSampler(sampling_strategy = ratio, random_state=42)

    X_tr_rs, y_tr_rs = RUS.fit_sample(X_tr, y_tr)

    return X_tr_rs, y_tr_rs, RUS.sample_indices_
