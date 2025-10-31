import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load the necessary files
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('trained_model\words.pkl', 'rb'))
classes = pickle.load(open('trained_model\classes.pkl', 'rb'))
model = load_model('trained_model\chatbot_model.h5')

# Function to clean up a sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class/intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a response from the bot
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I don't understand. Can you please rephrase?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = "I'm sorry, I don't have a response for that." # Default response
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# --- Streamlit App Interface ---

st.set_page_config(page_title="Healthcare Chatbot", layout="centered")

st.title("🩺 Healthcare Chatbot")
st.write("Welcome! I'm here to provide information based on your symptoms. Please note, I am not a doctor. This is not a medical diagnosis.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How are you feeling today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.spinner('Thinking....'):
        ints = predict_class(prompt)
        res = get_response(ints, intents)

    # Display bot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(res)
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": res})