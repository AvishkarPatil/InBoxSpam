import streamlit as st  # type: ignore
import pickle

st.markdown(
    """
    <style>
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    .stTextArea {
        text-align: center;
    }
    .stButton button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Error loading model or vectorizer. Please ensure the 'spam.pkl' and 'vectorizer.pkl' files are present.")
    st.stop()

st.title("📧 Email Spam Classification Application")
st.write(
    """
    Welcome to the Email Spam Classifier!
    This application uses Machine Learning to determine whether an email is **Spam** or **Not Spam (Ham)**.
    """
)

st.subheader("Classification")
user_input = st.text_area("Enter the email text below for classification:", height=150)

if st.button("Classify"):
    if user_input.strip():
        try:
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)

            if result[0] == 0:
                st.success("✅ This is NOT a Spam Email!")
            else:
                st.error("🚨 This is a SPAM Email!")

            st.experimental_rerun()
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
    else:
        st.warning("⚠️ Please enter email text before classifying.")