# New York Times Best Seller Predictor
Model predicts whether a book will become a New York times bestseller. The features used for the model are:

- Number of Pages
- Title
- Description
- Whether the author has been on the list before
- If the book is published under one of the Big 5 publishers
- If the book is part of a series
- Book Genre

--

### System Requirements:
Some of the packages used in this project include:

- `numpy`
- `pandas`
- `sklearn`
- `nltk`
- `TextBlob`
- `textstats`
- `imblearn`
- `streamlit`

--

Most of the project code is contained in the notebooks folder.

### NYT API
Data on NYT best sellers is gathered using the [NYT API](https://developer.nytimes.com/).
Code for how this data is retrieved is located in the `NYT API requests.ipynb` notebook.
It makes requests to the API and saves the data into a json file.

### Creating NYT Best Seller Datasets
Reading the NYT best sellers .json files into pandas dataframes occurs in  the
`Creating NYT Best Seller Datasets.ipynb` notebook.

### Merging Datasets
The self retrieved data from the NYT API,
another NYT best sellers dataset from Kaggle, and a general books dataset from Kaggle
which was scrapped from goodreads.com. I choose to include the second NYT best sellers
dataset in an effort to be able to include more NYT best sellers books and increase the chances
of intersecting with the goodreads dataset. This is done with the intention of helping reduce
class imbalance.

The merging of these different datasets is conducted in the `Merging Datasets.ipynb` notebook.

### Further Cleaning , Some EDA + Models
The bulk of the project is done in the `Further Cleaning, Some EDA + Models.ipynb`
notebook. This is where data cleansing, text processing, EDA and modeling is done.
It starts off with a baseline model with and iteratively test models after each added set of features.

NLTK was used for text preprocessing.
TextBlob was used to measuring Title and Description polarity.
texstats was used for features: Description length, average word length, Flesch Reading Ease, and more.

Models tested includes:
- Dummy Classifier
- Logistic Regression
- k Nearest Neighbors
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- SVC
- Decision Tree
- Random Forest

### Results
The best performing model is a Random Forest Classifier with a preliminary Lasso Logistic
Regression model to pick the words in the book description that serve as best predictors.

### Streamlit Webapp
The deployment of the model is implemented through a streamlit webapp. Source code for
this is provided in the `webapp` folder. Folder contains:

#### `text_process.py`
Code for text preprocessing of Book title and description that would be inputted into
the predictor.

#### `imb_undersample.py`
Code to implement undersampling of the 0th class (non best seller) to account for large
class imbalance.

#### `OptimalC.py`
Code to find the optimal regularization C parameter for the Lasso Logistic regression.
The goal is to find the words in books descriptions that would serve as best predictors
of a NYT best seller book without having these words outnumber the amount of book features
such as number of pages, semantics, genre, etc. The limit for the number of words picked from this
model is 25 words. Optimal C = 0.2.

#### `selectBestWords.py`
Code to perform the Lasso Logistic regression to pick out the word features for the
final model.

### To run Webapp: 
`streamlit run nyt_webapp.py`
