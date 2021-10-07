import pandas as pd
import numpy as np
from pprint import pprint
import re

#####################################################################################################################
# Import Relevant files for training and pre-processing
#####################################################################################################################
with open('data/hotelposT-train.txt', 'r') as f:
    positive_reviews_raw = f.readlines()

with open('data/hotelnegT-train.txt', 'r') as f:
    negative_reviews_raw = f.readlines()

with open('data/positive-words.txt', 'r') as f:
    positive_words_raw = f.readlines()

with open('data/negative-words.txt', 'r') as f:
    negative_words_raw = f.readlines()

# file constants
positive_words = [word.strip().rstrip() for word in positive_words_raw]
negative_words = [word.strip().rstrip() for word in negative_words_raw]
pronouns = ["I", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]

#####################################################################################################################
#  define necessary preprocessing and feature extraction definitions
#####################################################################################################################
def text_regex_preprocessing(text):
    """
    Takes the raw review and removes any unecessary features from the review

    1. removes any new line symbols
    2. removes any apostrophies
    3. removes text that have money references (e.g. $100)
    4. removes digits -- they do not count as words
    5. removes any paraentheses since we want the words inside of them, not the symbol itself. 
    
    """
    text = text.replace('\n', '')
    text = text.strip().rstrip()
    text = re.sub('\'', '', text)
    text = re.sub('^$[0-9]', '', text)
    text = re.sub('\d', '', text)
    text = re.sub("\(|\)|\'", '', text)
    return text

def extract_id(text):
    """
    Takes the raw review from the text file and extracts the ID number from the start of the raw review
    and the unprocessed text. The text will be processed in another function.
    """
    idnum = re.findall('^ID-[0-9]*\t', text)[0].rstrip().strip()
    review = re.sub('^ID-[0-9]*\t', '', text)
    return idnum, review

def log_word_count(review):
    """
    Takes the natural log of the entire review, not just the unique words
    """
    return round(np.log(len(review.split(' '))), 2)

def review_contains_no(review):
    """
    Checks to see if the review contains any instance of the word 'no'
    """
    review_words = [word.lower() for word in review.split(' ')]
    if 'no' in  review_words:
        return 1
    else:
        return 0

def review_contains_exclimation_point(review):
    """
    Checks to see if the review contains any instance of a '!'
    """
    review_words = [word.lower() for word in review.split(' ')]
    if '!' in  review_words:
        return 1
    else:
        return 0

def word_intersections(review):
    """
    Checks to see if the review contains positive or negative words 

    Takes the set of words from pre-processed review and intersection with negative and positive words
    dictionaries. From there, take the length that set and return for the model feature. 

    TODO: Check to see if assignment wants unqiue count or total count --> does the count include two instance of the same positive word?
    """
    review_words = {word.lower() for word in review.split(' ')}

    n_positive_words = len(review_words.intersection(positive_words))
    n_negative_words = len(review_words.intersection(negative_words))

    return n_negative_words, n_positive_words

def pronouns_used(text, pronouns):
    review_words = {word for word in text.split(' ')}
    n_pronouns_used = len(review_words.intersection(pronouns))

    return n_pronouns_used


def create_review_corpus(reviews, dictionary_to_append_to, sentiment_class=None,):
    for review in reviews:
        idnum, review_text = extract_id(review)
        dictionary_to_append_to[idnum]={
            'review_text':text_regex_preprocessing(review_text),
            'sentiment_class': sentiment_class
        }

        dictionary_to_append_to[idnum]['log_word_count'] = log_word_count(dictionary_to_append_to[idnum]['review_text'])
        dictionary_to_append_to[idnum]['contains_no'] = review_contains_no(dictionary_to_append_to[idnum]['review_text'])
        dictionary_to_append_to[idnum]['contains_exclimation_point'] = review_contains_exclimation_point(dictionary_to_append_to[idnum]['review_text'])

        n_positives, n_negatives = word_intersections(review)
        dictionary_to_append_to[idnum]['n_positive_words'] = n_positives
        dictionary_to_append_to[idnum]['n_negative_words'] = n_negatives

        dictionary_to_append_to[idnum]['n_pronouns_used'] = pronouns_used(dictionary_to_append_to[idnum]['review_text'], pronouns)

reviews = {}

create_review_corpus(positive_reviews_raw, sentiment_class=1, dictionary_to_append_to=reviews)
create_review_corpus(negative_reviews_raw, sentiment_class=0, dictionary_to_append_to=reviews)

review_csv = pd.DataFrame().from_dict(reviews, orient='index').reset_index().rename(columns={'index':'id'}).drop('review_text', axis=1)
review_csv.to_csv('curran-thomas-assgn2-part1.csv', index=False, header=False)


with open('data/HW2-testset.txt', 'r') as f:
    testset_raw = f.readlines()

testset_reviews = {}

create_review_corpus(testset_raw, dictionary_to_append_to=testset_reviews)

testset_df = pd.DataFrame().from_dict(testset_reviews).T.reset_index().drop('review_text', axis=1)
testset_df.to_csv('assgn2-testset-reviews.csv', header=False, index=False)