import pandas as pd
import re
from collections import defaultdict
import math

# reads the dataset 
def read_file(file_path = 'IMDB Top 250 Movies.csv'):
    try:
            data_table = pd.read_csv(file_path)
    except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None

    #create one column for all things we want to test for
    data_table['test_features'] =  data_table['genre'] + ' ' + data_table['tagline'] + \
        data_table['casts'] + ' ' + data_table['directors'] + ' ' + data_table['writers']
    
    documents = pd.Series(data_table['test_features'].values, index=data_table['name']).to_dict()
    return documents

#put all words in the text in the same case form 
def tokenize(text):
      if not isinstance(text, str):
            return []
      return re.findall(r'\b\w+\b', text.lower())

# makes a dictionary of all the words then append movie titles to the words that contain them 
def build_inverted_index(documents):
    inverted_index = defaultdict(list)
# loop the documetns contents
    for movie_title, test_features in documents.items():
            # loop the words we are testing for 
            for token in tokenize(test_features):
                  # check if movie title is already assigned to that word if not append it 
                  if movie_title not in inverted_index[token]:
                        inverted_index[token].append(movie_title)
    
    return inverted_index

#calculates the term frequency and the inverse document frequency
def calculate_tf_idf_vectors(documents, inverted_index):
    num_doc = len(documents)
    vocabulary = list(inverted_index.keys())
    vocab_size = len(vocabulary)

    idf = {}
    for term,doc_ids in inverted_index.items():
            idf[term] = math.log(num_doc/ (len(doc_ids) + 1))

    tfidf_vectors = {}
    tokenized_docs = {doc_id: tokenize(text) for doc_id,text in documents.items()}

    for doc_id, tokens in tokenized_docs.items():
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
        
        total_tokens = len(tokens)
        tf = {token: count / total_tokens for token, count in term_counts.items()}
        vector = [0] * vocab_size
        for i, term in enumerate(vocabulary):
            tf_score = tf.get(term, 0)
            idf_score = idf.get(term, 0)
            vector[i] = tf_score * idf_score

        tfidf_vectors[doc_id] = vector
    return tfidf_vectors, vocabulary

#calculates consine similarity between movies 
def cosine_similarity(vector_1, vector_2):
    dot_product = sum(x * y for x,y in zip(vector_1,vector_2))
    magnitude_vector1 =  math.sqrt(sum(x**2 for x in vector_1))
    magnitude_vector2 = math.sqrt(sum(x**2 for x in vector_2))

    if magnitude_vector1 == 0 or magnitude_vector2 == 0:
        return 0
    
    return dot_product / (magnitude_vector1 * magnitude_vector2)

#give the recommendations for the movie you liked  shows only top 5 
def get_recommendations(target_title, documents, tfidf_vectors, top_n = 5):
    if target_title not in documents:
          return f"Error: Movie '{target_title}' not found in the dataset."

    target_vector = tfidf_vectors[target_title] 
    similarities = []

    for movie_title,vector in tfidf_vectors.items():
        if movie_title != target_title:
            sim_score = cosine_similarity(target_vector, vector)
            similarities.append((movie_title, sim_score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]
    

if __name__ == "__main__":
    # --- Step 1: Setup (runs only once at the start) ---
    print("Loading and preparing movie data...")
    movie_documents = read_file()
    if movie_documents:
        inverted_index = build_inverted_index(movie_documents)
        tfidf_vectors, vocabulary = calculate_tf_idf_vectors(movie_documents, inverted_index)
        print("Movie recommendation engine is ready.")
        print("-" * 40)

        # --- Step 2: Interactive Loop ---
        while True:
            # Get input from the user
            user_input = input("Enter a movie title to get recommendations (or type 'exit' to quit): ")
            
            # Check if the user wants to quit
            if user_input.lower() == 'exit':
                break
            
            # Get and display the recommendations
            recommendations = get_recommendations(user_input, movie_documents, tfidf_vectors)
            
            if isinstance(recommendations, str):
                # Handle the case where the movie is not found
                print(recommendations)
            else:
                print(f"Top 5 recommendations for '{user_input}':")
                for title, score in recommendations:
                    print(f"- {title} (Similarity: {score:.3f})")
            
            print("-" * 40)