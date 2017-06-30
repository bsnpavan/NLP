import os
import numpy as np
import pandas as pd

#Set Path 
os.chdir('E:\AVChennaiMeetup')

#List Files in the Directory
os.listdir()

# Import data (Ensure Encoding as below to avoid unexpected format issues)
test = pd.read_csv('test.csv')
test_tweets = pd.read_csv('test_tweets.csv', encoding="ISO-8859-1")
train = pd.read_csv('train.csv')
train_tweets = pd.read_csv('train_tweets.csv', encoding="ISO-8859-1")

# Join train data on ID
train_data = pd.merge(train, train_tweets, on='ID', how='outer')
test_data = pd.merge(test, test_tweets, on='ID', how='outer')

#Remove old train data frames to free up memory
del train
del train_tweets

from bs4 import BeautifulSoup  

# Initialize the BeautifulSoup object on a single tweet i.e. first record 
example1 = BeautifulSoup(train_data['Tweet'][0])  

# Use regular expressions to do a find-and-replace
import re
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print(letters_only)

#Convert to lower Case
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words

# Import Natural language Toolkit
import nltk
nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
print(stopwords.words("english")) 

# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print(words)

###There are many other things we could do to the data - For example, Porter Stemming and Lemmatizing (both available in NLTK) would allow us to treat "messages", "message", and "messaging" as the same word, which could certainly be useful. However, for simplicity, we can stop here
## TO CONTRUCT A SINGLE FUNCTION TO APPLY ON THE COMPLETE DATA
def tweets_to_text(raw_tweet):
    # Function to convert a raw tweet to a string of words
    # The input is a single string (a raw tweet), and 
    # the output is a single string (a preprocessed raw tweet)
    #
    # 1. Remove HTML
    raw_tweet = BeautifulSoup(raw_tweet).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_tweet) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

##CAll the function for clean tweets
check_one_tweet = tweets_to_text(train_data['Tweet'][0])
print(check_one_tweet)

# Total number of Tweets in training set
num_train_tweets = len(train_data.index)

# Initialize an empty list to hold the clean tweets
clean_train_tweets = []

# Loop over each tweet; 
for i in range( 0, num_train_tweets ):
    # Call our function for each one, and add the result to the list of
    # clean tweets
    clean_train_tweets.append(tweets_to_text(train_data['Tweet'][i]))
    
### Calculate the frequency of the words
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
data_train_features = vectorizer.fit_transform(clean_train_tweets)

# Numpy arrays are easy to work with, so convert the result to an 
# array
data_train_features = data_train_features.toarray()

# Take a look at the words in the vocabulary i.e. the features generated
features = vectorizer.get_feature_names()
print(features)

# Sum up the counts of each vocabulary word
count_of_words_train = np.sum(data_train_features, axis=0)


# For each, print the vocabulary word and the number of times it 
# appears in the training set
for word, count in zip(features, count_of_words_train):
    print(count, word)
## Consolidating train and test sets before classification
X_train = data_train_features
Y_train = train_data['Sentiment'] 

print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( X_train, Y_train)

# Create an empty list and append the clean reviews one by one
num_test_tweets = len(test.index)
clean_test_tweets = [] 

print("Cleaning and parsing the test set tweets...\n")
for i in range(0,num_test_tweets):
    clean_test_tweets.append(tweets_to_text(test_data["Tweet"][i]))

# Get a bag of words for the test set, and convert to a numpy array
data_test_features = vectorizer.transform(clean_test_tweets)
data_test_features = data_test_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(data_test_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test_data["ID"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "AV_ChennaiMeetup_output.csv", index=False, quoting=3 )







