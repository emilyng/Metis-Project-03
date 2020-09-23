import pickle
import streamlit as st
import numpy as np
import pandas as pd
from predict import compile_data, make_prediction
from text_process import get_title_data, get_desc_data, prep_desc
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import Normalizer

def str2bool(v):
    if str(v).lower() in ("yes", "true", "t", "1"):
        return 1
    else:
        return 0

st.image('books.jpg', width=700)

st.title('New York Times Best Seller Predictor')
st.write('Will your book become the next best seller?')

number_of_pages = st.text_input("Number of Pages", 22)
try:
    float(number_of_pages)
except:
    st.text('Number of pages needs to be a number!')

title = st.text_input("Title")
if len(title) == 0:
    st.text('Please Enter Book Title.')

description = st.text_area("Description", height=50)
if len(description) == 0:
    st.text('Please Enter Book Description.')

is_best_selling_author = ['Yes', 'No']
author = str2bool(st.selectbox(
    'Has the author been on the list before?',
     is_best_selling_author))

is_top_publisher = ['Yes', 'No']
publisher = str2bool(st.selectbox(
    'Is it published under one of the Big 5 publishers? \n(Penguin & Random, HarperCollins, Simon & Schuster, Hachette, or MacMillan)',
     is_top_publisher))

is_series = ['Yes', 'No']
series = str2bool(st.selectbox(
    'Is it part of a series?',
    is_series))

genres = ['Fantasy', 'Fiction', 'Historical Fiction', 'Mystery', 'Nonfiction',
          'Other', 'Romance', 'Science Fiction', 'Young Adult']
genre = st.selectbox(
    'Pick the genre that best describes the book.',
     genres)

st.sidebar.markdown('Adjust the minimum probability to consider a book becoming a Best Seller.')
threshold = st.sidebar.slider('Threshold', 0, 100, 50, 1)
#st.sidebar.info(value)

if len(title) == 0 or len(description) == 0:
    pass
else:
    data = compile_data(number_of_pages, title, description, author, publisher, series, genre)
    st.header(make_prediction(data, threshold))
