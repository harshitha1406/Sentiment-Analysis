import streamlit as st
import sklearn
import pickle
from sklearn.feature_extraction.text import CountVectorizer
new_model = pickle.load(open('Reviews.pkl', 'rb'))
vect = pickle.load(open('vect.pkl', 'rb'))
st.title('Sentiment Analysis')
input = st.text_input('Enter your message : ')
input = vect.transform([input])
new_y_pred = new_model.predict(input.toarray())

if st.button('Predict'):
  st.title([new_y_pred[0]])
