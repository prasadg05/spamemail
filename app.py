import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample trained data (simulate training)
data = {'text': ['Win a lottery now', 'Hi, how are you?', 'Lowest price on medicines', 'Let's catch up later', 'Congratulations, you won!'],
        'label': [1, 0, 1, 0, 1]}
df = pd.DataFrame(data)

# Vectorize and train
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
model = MultinomialNB()
model.fit(X, df['label'])

st.title("📧 Spam Email Classifier")

user_input = st.text_area("Enter the email content:")

if st.button("Check"):
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)[0]
    st.write("**Spam**" if prediction == 1 else "**Not Spam**")