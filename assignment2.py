import json
import time
import math
import csv
import nltk
import sys
from string import punctuation
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import defaultdict

# if len(sys.argv) != 4:
#     print("Usage: python script_name.py <topics_1.json> <topics_2.json> <answers.json>")
#     sys.exit(1)

# set of stopwords
stop_words = set(stopwords.words('english')) # Made into a set in case the list has doubles

# list of special characters
special_chars = set(punctuation)

# # loading json files into local variables
# with open(sys.argv[1], 'r', encoding = 'utf-8') as json1:
#     topic_one_file = json.load(json1)
# with open(sys.argv[2], 'r', encoding = 'utf-8') as json2:
#     topic_two_file = json.load(json2)
# with open(sys.argv[3], 'r', encoding='utf-8') as json3:
#     answers_file = json.load(json3)



## PARSING THROUGH JSON FILES ##
''' 
HTML tags, stopwords, then special characters are removed in that order for 
the topics files "Body" text and answer file's "Text" . The order is the same for any other text with no HTML 
tags. The text is also turned to lower case at the end.
'''
# This function will be used to remove html tags from the body.
def remove_tags(soup):
    for data in soup(['style', 'script']):
        #remove tags
        data.decompose()
    # returns a string
    return ' '.join(soup.stripped_strings)

# This function will be used to remove stop words and special characters from any text
# Also turns the cleaned text to lower-case
def remove_stopwords(text, list_stopwords, list_special_chars):
    # turn the text into a list
    words = text.split()
    # remove special characters
    for word in words:
        for char in list_special_chars:
            word = word.replace(char,"")
    # remove stopwords from the text
    filtered_text = [w for w in words if not w.lower() in list_stopwords]
    # remove whitespace in words
    clean_text = ' '.join(filtered_text).strip()
    clean_text = clean_text.lower()
    # returns a string
    return clean_text

# This function will remove htlm tags first and then remove the stopwords
def parsing_topics_strings(json_file):
    # iterate through the queries in topics file
    for query in json_file:
        title = query['Title']
        title = remove_stopwords(title, stop_words, special_chars)
        query['Title'] = title # title has been parsed
        
        body = query['Body']
        body = remove_tags(BeautifulSoup(body, "html.parser"))
        body = remove_stopwords(body, stop_words, special_chars)
        query['Body'] = body # body has been parsed
    return json_file # return the json file

# This function is made to remove htlm tags and stopwords from answer texts
def parsing_answer_text(answer_file):
    # iterate through documents and cleans them
    for answer in answer_file:
        text = answer['Text']
        text = remove_tags(BeautifulSoup(text, "lxml"))
        text = remove_stopwords(text, stop_words, special_chars)
        answer['Text'] = text # text has been parsed
    return answer_file # return the answers document

## MAKING TF-IDF DICTIONARY ##
'''
In this section I make:
- answer_tf_dict: {word : {answer_id : tf}}
- answer_idf_dict: {word: idf}
- a useless function for scoring
- answer_tf_idf_matrix: {word: {answer_id : tf-idf value}} #I don't know if matrix is the write word
- answer_vector_dict: {answer_id : {word: tf_idf}}
- query_tf_dict: {word: {query_id : tf}}
- query_tf_idf_matrix: {word: {query_id : tf_idf}}
- query_vector_dict: {query_id : {word: tf_idf}}
'''


# This method aims to make a term frequency dictionary 
# term frequency = number of times term appears in document / total number of terms
def make_tf_dict(answer_file):
    tf_dict = defaultdict(lambda: defaultdict(float)) # {word : {answer_id : tf}}
    all_ids = [answer['Id'] for answer in answer_file]
    for answer in answer_file:
        text = answer['Text'].split() # list of words
        id = answer['Id'] # answer id
        text_length = len(text) # total terms in the answer document
        word_count = {} # {word : count}
        # get the number of times every word appears in the document
        for word in text:
            word_count[word] = word_count.get(word, 0) + 1
        # iterate through the word_count dict and get the tf value for each word
        for word, count in tqdm(word_count.items()):
            if word not in tf_dict:
                # initialize every id to zero for that word
                tf_dict[word] = {id: 0.0 for id in all_ids}
            tf_dict[word][id] = count / text_length
    return tf_dict
# this function makes an idf dict that is used for measuring the TFxIDF of both answers and documents
def make_idf_dict(tf_dict, answer_file):
    total_documents = len(answer_file)
    idf_dict = {} # (word: idf)
    for word, doc_dict in tf_dict.items():
        # Count how many documents contain the word (TF > 0)
        number_of_docs_with_term = sum(1 for tf in doc_dict.values() if tf > 0)
        # Calculate IDF
        idf_dict[word] = math.log10(total_documents / number_of_docs_with_term)
    return idf_dict
    

def make_tf_idf_matrix(tf_dict, idf_dict):
    tf_idf_matrix = {} # {word: {id : tf-idf value}}
    # Iterate through every word
    for word, doc_dict in tf_dict.items():
        # Get the IDF value of the word
        idf_value = idf_dict[word]
        # Initialize the inner dictionary for the word
        tf_idf_matrix[word] = {}
        # Calculate TF-IDF for each document and store it
        for doc_id, tf_value in doc_dict.items():
            tf_idf_matrix[word][doc_id] = tf_value * idf_value  # Compute TF-IDF
    return tf_idf_matrix


# this function makes a vector for every answer document and stores it in a dictionary
def make_answers_vectors(tf_idf_matrix):
    answers_vector = {} # {id: {word : tf-idf value}}
    # Iterate through the tf-idf matrix to build the answers vector
    for word, doc_dict in tf_idf_matrix.items():
        for doc_id, tf_idf_value in doc_dict.items():
            if doc_id not in answers_vector:
                answers_vector[doc_id] = {}  # Initialize the dictionary for this ID
            answers_vector[doc_id][word] = tf_idf_value  # Assign the tf-idf value
    return answers_vector

# this function iterates through every query from the topics files and obtains the tf value of every word from the body and title
def make_query_tf_dict(topic_file):
    query_tf_dict = {} # {word: {q_id: tf}}
    all_ids = [query['Id'] for query in topic_file]
    # iterate through the topic_file
    for query in topic_file:
        title = query['Title'].split()
        body = query['Body'].split()
        q_id = query['Id']
        combined_text = title + body
        text_length = len(combined_text)
        word_count = {} # {word: count}

        # Count the number of times the word appears in the query text
        for word in combined_text:
            word_count[word] = word_count.get(word, 0) + 1

         # Calculate TF for each word
        for word, count in word_count.items():
            if word not in query_tf_dict:
                query_tf_dict[word] = {}  # Initialize the word entry
            query_tf_dict[word][q_id] = count / text_length  # Assign the TF value
    # Calculate for missing id's
    for word in query_tf_dict:
        for id in all_ids:
            if id not in query_tf_dict[word]:
                query_tf_dict[word][id] = 0.0
    return query_tf_dict

def make_query_tf_idf_matrix(query_tf_dict, idf_dict):
    query_tf_idf_matrix = {} #{word : {q_id: tf_idf}}
    for word, tf_dict in query_tf_dict.items():
        query_tf_idf_matrix[word] = {}  # Initialize the word entry
        idf_value = idf_dict.get(word, 0.0)  # Get IDF value or default to 0.0
        # Calculate TF-IDF for each ID
        for q_id in tf_dict:
            query_tf_idf_matrix[word][q_id] = tf_dict[q_id] * idf_value
    return query_tf_idf_matrix

def make_query_vectors(query_tf_idf_matrix):
    query_vectors = {} # {q_id {word: tf_idf}}
    for word, id_dict in query_tf_idf_matrix.items():
        for q_id, tf_idf_value in id_dict.items():
            if q_id not in query_vectors:
                query_vectors[q_id] = {}  # Initialize the inner dictionary for this ID
            query_vectors[q_id][word] = tf_idf_value  # Assign the tf-idf value
    return query_vectors

## COSINE SIMILARITY ##
# THIS IS THE EQUATION FOR COSINE SIMILARITY AND I HOPE ITS CORRECT
def cosine_similarity(query_vector,answer_vector):
    dot_product = 0
     # get the dot product of the query and answer vectors
    for word in query_vector:
        if word in answer_vector:  # Check if the word exists in answer_vector
            dot_product += query_vector[word] * answer_vector[word]

    #query length
    query_vector_length = sum(value ** 2 for value in query_vector.values()) ** 0.5

    #answer length
    answer_vector_length = sum(value ** 2 for value in answer_vector.values()) ** 0.5

    #avoid division by zero
    if query_vector_length == 0 or answer_vector_length == 0:
        return 0.0
    else:
        return dot_product / (query_vector_length * answer_vector_length)

def get_results(query_vector_dict, answer_vector_dict):
    query_answer_results = {} # {query_id: {answer_id: cosine similarity}}
     # Iterate through queries
    for query_id in query_vector_dict:
        # The dict will store all the similarities for one query and then be added to the results in the end
        similarity_dict = {}  # {answer_id: cosine similarity}
        query_vector = query_vector_dict[query_id]  # {word: tf_idf}

        for answer_id in answer_vector_dict:
            answer_vector = answer_vector_dict[answer_id]  # {word: tf_idf}
            similarity = cosine_similarity(query_vector, answer_vector)
            similarity_dict[answer_id] = similarity

        # Sort the results in descending order
        sorted_answers_similarities = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)

        # Adding the results to the final dict
        query_answer_results[query_id] = sorted_answers_similarities
    return query_answer_results

def make_results_tsv_file(topic_results_dict, name_of_file):
    with open(name_of_file, 'w') as tsv_file:
        # Iterate through queries
        for queryID in topic_results_dict:
            rank = 1
            # Iterate through the results of the queries, only taking the top 100
            for answer, score in topic_results_dict[queryID][:100]:  # Unpack answer and score
                # Determine score based on the similarity threshold
                if score >= 0.70:
                    result_score = 2
                elif 0.70 > score >= 0.45:
                    result_score = 1
                else:
                    result_score = 0  # Add a case for scores below 0.45 if needed
                # Write to TSV file
                tsv_file.write(f"{queryID}\tQ0\t{answer}\t{rank}\t{result_score}\tbroken_system\n")  # Use 'answer' instead of 'id'
                rank += 1
        
                
                    

# loading json files into local variables
with open('topics_1.json', 'r', encoding = 'utf-8') as json1:
    topic_one_file = json.load(json1)
with open('topics_2.json', 'r', encoding = 'utf-8') as json2:
    topic_two_file = json.load(json2)
with open('Answers.json', 'r', encoding='utf-8') as json3:
    answers_file = json.load(json3)

# true_start = time.time()
# start_time = time.time()
# # cleaning the text
# print("Parsing stuff...")
# topic_one_file = parsing_topics_strings(topic_one_file)
# topic_two_file = parsing_topics_strings(topic_two_file)
# answers_file = parsing_answer_text(answers_file)
# end_time = time.time()
# print(f"Time it took to parse files: {round(end_time - start_time, 3)} seconds ")
# # answers
# start_time = time.time()
# print("Creating TF Dict...")
# answers_tf_dict = make_tf_dict(answers_file)
# print("created tf dict")
# print("Creating idf dict")
# answers_idf_dict = make_idf_dict(answers_tf_dict,answers_file)
# print("Created idf dict")
# answers_tf_idf_matrix = make_tf_idf_matrix(answers_tf_dict,answers_idf_dict)
# answers_vectors_dict = make_answers_vectors(answers_tf_idf_matrix)
# end_time = time.time()
# print(f"Time it took to make answers_vectors_dict: {round(end_time - start_time, 3)} seconds ")
# # topic 1
# start_time = time.time()
# topic_one_tf_dict = make_query_tf_dict (topic_one_file)

# topic_one_tf_idf_matrix = make_query_tf_idf_matrix(topic_one_tf_dict,answers_idf_dict)
# topic_one_vectors_dict = make_query_vectors(topic_one_tf_idf_matrix)
# topic_one_results = get_results(topic_one_vectors_dict,answers_vectors_dict)
# make_results_tsv_file(topic_one_results, 'result_tfidf_1.tsv')
# end_time = time.time()
# print(f"Time to get results for topic 1 and make tsv file: {round(end_time - start_time, 3)} seconds ")
# # topic 2
# start_time = time.time()
# topic_two_tf_dict = make_query_tf_dict (topic_two_file)
# topic_two_tf_idf_matrix = make_query_tf_idf_matrix(topic_two_tf_dict,answers_idf_dict)
# topic_two_vectors_dict = make_query_vectors(topic_two_tf_idf_matrix)
# topic_two_results = get_results(topic_two_vectors_dict,answers_vectors_dict)
# make_results_tsv_file(topic_two_results, 'result_tfidf_2.tsv')
# end_time = time.time()
# print(f"Time to get results for topic 2 and make tsv file: {round(end_time - start_time, 3)} seconds ")

# true_end = time.time()
# print(f"Time it took to finish everything: {round(true_end - true_start, 3)} seconds ")



