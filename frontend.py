import streamlit as st
from medical_rag_backend import get_medical_response

st.set_page_config(page_title="Medical Assistant", page_icon="🩺")
st.title("🩺 Medical Assistant Chatbot")
st.markdown(
    "Ask health-related questions. I provide information from trusted medical documents."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input("You:", "")

if user_input:
    answer = get_medical_response(user_input)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Assistant", answer))

# Display chat history
for sender, message in st.session_state.messages:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")
