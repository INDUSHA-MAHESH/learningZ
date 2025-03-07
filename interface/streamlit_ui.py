import streamlit as st
import pickle
import numpy as np
import math


def predict_sentiment_single(text, pos_prob, neg_prob, pos_likelihoods, neg_likelihoods, should_tokenize=True):
    tokens = text.split() if should_tokenize else text
    pos_log_prob = math.log(pos_prob) 
    neg_log_prob = math.log(neg_prob)  
    
    smoothing_factor = 1e-6  
    
    for word in tokens:
        pos_prob_word = pos_likelihoods.get(word, smoothing_factor)
        neg_prob_word = neg_likelihoods.get(word, smoothing_factor)

        pos_log_prob += math.log(pos_prob_word+1e-6)  
        neg_log_prob += math.log(neg_prob_word+1e-6)  

    if pos_log_prob > neg_log_prob:
        return "positive"
    else:
        return "negative"

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def predict_word_embeddings(word, W1, vocab_to_int, int_to_vocab, top_n=10):
    if word not in vocab_to_int:
        print(f"Word '{word}' not in vocabulary.")
        return []

    embedding = W1[vocab_to_int[word]]
    similarities = {}
    for i in range(len(int_to_vocab)):
        if int_to_vocab[i] != word:  # Exclude the input word itself
            similarities[int_to_vocab[i]] = cosine_similarity(embedding, W1[i])

    # Sort by similarity and return the top N words
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities[:top_n]



def viterbi_decode(words, transition_probabilities, emission_probabilities, initial_probabilities, word2idx, tag2idx, idx2tag):
    """optimize finding of likely seq"""
    words = words.split()
    n_words = len(word2idx)
    n_tags = len(tag2idx)
    T = len(words)

    viterbi = np.zeros((n_tags, T)) #matrix
    backpointers = np.zeros((n_tags, T), dtype=int)

    for s in range(n_tags): 
        word = words[0].lower() 

        if word in word2idx:
            viterbi[s, 0] = initial_probabilities[s] * emission_probabilities[s, word2idx[word]]
        else:
            viterbi[s, 0] = initial_probabilities[s] * 0.00001 # small prob

    for t in range(1, T):
        word = words[t].lower() 
        for s in range(n_tags): 
            if word in word2idx:
                emission_prob = emission_probabilities[s, word2idx[word]]
            else:
                emission_prob = 0.00001

            viterbi[s, t] = np.max(viterbi[:, t-1] * transition_probabilities[:, s] * emission_prob)
            backpointers[s, t] = np.argmax(viterbi[:, t-1] * transition_probabilities[:, s]) #stores the best possible path at any pt of time

    best_path = [None] * T
    best_path_pointer = np.argmax(viterbi[:, T-1])
    best_path[T-1] = idx2tag[best_path_pointer]

    for t in range(T-2, -1, -1):
        best_path_pointer = backpointers[best_path_pointer, t+1]
        best_path[t] = idx2tag[best_path_pointer]

    return best_path


def predict(tab_no,mod_input,no=0):
    if tab_no=="one":
        file = "word_embeddings.pkl"
        with open(file,'rb') as f:
            data = pickle.load(f)
            W1_loaded,vocab_to_int_loaded = data["embeddings"],data["vocab_to_int"]
        int_to_vocab_loaded = {i: word for word, i in vocab_to_int_loaded.items()}
        similar_words = predict_word_embeddings(mod_input, W1_loaded, vocab_to_int_loaded, int_to_vocab_loaded, top_n=no)
        return similar_words

    elif tab_no=="two":
        file = "naive_bayes_model_scratch.pkl"
        with open(file, 'rb') as f:
            data = pickle.load(f)
            pos_prob, neg_prob, pos_likelihoods, neg_likelihoods = data["pos_prob"],data["neg_prob"],data["pos_likelihoods"],data["neg_likelihoods"]
        prediction = predict_sentiment_single(mod_input, pos_prob, neg_prob, pos_likelihoods, neg_likelihoods)
        return prediction
    
    elif tab_no=="three":
        file = "ner_hmm_model_scratch.pkl"
        with open(file,'rb') as f:
            data = pickle.load(f)
            transition_probabilities, emission_probabilities, initial_probabilities, word2idx, tag2idx, idx2tag=data["transition_probabilities"], data['emission_probabilities'], data['initial_probabilities'], data['word2idx'], data['tag2idx'], data['idx2tag']
        predicted_tags = viterbi_decode(mod_input, transition_probabilities, emission_probabilities, initial_probabilities, word2idx, tag2idx, idx2tag)
        return predicted_tags
    
    
tab1,tab2,tab3 = st.tabs(["Skip Gram", "Naive Bayes", "NER-HMM"])

with tab1:
    st.header("Skip Gram Word Embeddings")
    st.subheader("Over the IMDB Dataset")
    inp = st.text_input("Enter a word related to movies(reviews) to get it's related words")
    no = st.slider("Set the number of related words to return",1,20,5)
    # st.button("Predict")
    if st.button('Predict',key=3):
        result = predict("one", inp,no)
        st.write("Similar Words to "+inp)
        for i in result:
            st.write(i[0])

with tab2:
    st.header("Naive Bayes Sentiment Analysis")
    st.subheader("Over the IMDB Dataset")
    inp= st.text_input("Enter a review to predict it's sentiment")
    st.warning("It's currently not accurate much")
    #st.button("Predict",on_click=predict(2,inp))
    if st.button('Predict',key=1):
        result = predict("two", inp)
        if result=="positive":
            st.success("Sentiment of the review is positive")
        else:
            st.error("Sentiment of the review is negative")

with tab3:
    st.header("Named Entity Recognition with HMM")
    inp = st.text_input("Enter a small sentence to get it's named entities")
    st.warning("It's currently not accurate much")
    #st.button("Predict",on_click=predict(3,inp))
    if st.button('Predict',key=2):
        result = predict("three", inp)
        st.write("The Named Entity Labels are")
        st.write(result)

