import streamlit as st

# Title of the app
st.title('My First Streamlit App')

# Taking user input
user_input = st.text_input("Enter some text")

# Displaying user input
st.text("You entered: " + user_input)