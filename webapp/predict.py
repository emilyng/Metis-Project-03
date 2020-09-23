import pickle
import numpy as np
from text_process import get_title_data, get_desc_data, prep_desc

my_model = pickle.load(open("lr_rf_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
idx = pickle.load(open("idx.pkl", "rb"))

def vectorize_desc(description):
    '''
    TF-IDF Vectorize data for model prediction
    Input: description string
    Output: normalized tf-idf sparse vector
    '''
    prepped_desc = prep_desc(description)

    desc_vector = vectorizer.transform([prepped_desc])
    scaled_desc_vector = scaler.transform(desc_vector).todense()[:,idx]
    return scaled_desc_vector

def make_genre_vector(genre):
    '''
    Generate genre vector based on user input
    Input: user inputted genre (drop down selection)
    Output: genre vector
    '''
    genres = ['Fantasy', 'Fiction', 'Historical Fiction', 'Mystery', 'Nonfiction',
              'Other', 'Romance', 'Science Fiction', 'Young Adult']
    genre_vector = np.zeros(len(genres))
    for i, gen in enumerate(genres):
        if gen == genre:
            genre_vector[i] = 1
        else:
            genre_vector[i] = 0
    return genre_vector

def compile_data(number_of_pages, title, description, author, publisher, series, genre):
    '''
    Process and compile all neccessary data for model
    Input: user inputted number_of_pages, title string, description string, author selection
    publisher selection, series selection, genre selection
    Output: full_data ready for modeling
    '''
    title_semantic, title_word_count = get_title_data(title)
    desc_semantic, word_count, desc_len, num_unique_words, \
    avg_word_len, syllable_count, lexicon_count, sentence_count, \
    flesch_reading_ease = get_desc_data(description)

    genre_vector = make_genre_vector(genre)
    numerical_data1 = np.array([number_of_pages, series])
    numerical_data2 = np.array([publisher, author, title_semantic, desc_semantic,
                        title_word_count, desc_len, num_unique_words, avg_word_len,
                        syllable_count, lexicon_count, sentence_count, flesch_reading_ease])

    data0 = np.concatenate((numerical_data1, genre_vector, numerical_data2))
    data0 = data0.reshape(1,-1)

    scaled_desc_vector = vectorize_desc(description)

    full_data = np.concatenate((data0, scaled_desc_vector[0]), axis=1)

    return full_data

def make_prediction(data, threshold, model=my_model):
    '''
    Make prediction given new data
    Input: data, predefined model
    Output: message string associated with prediction[0,1]
    '''
    prob = model.predict_proba(data)
    threshold = threshold/100
    if prob[0][1] < threshold:
        prediction = 1
    else:
        prediction = 0
    message_array = ["You have a Best Seller! :smiley:",
                      "It is unlikely to get on the best seller list. :disappointed:"]
    return message_array[prediction]
