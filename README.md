<h2 align="center">InBoxSpam</h2>

<h3 align="center">Spam Email Classification</h3><br>

<div align="center">
<img src="https://cdn-icons-png.flaticon.com/512/10733/10733107.png" width="150">
</div>
<br>

<p align="center">This is a <b>Machine Learning application</b> built with Python and Streamlit that classifies emails as <b>spam</b> or <b>ham (not spam)</b>. It leverages a trained ML model to analyze text input and predict whether the email is spam or not.</p>
<br>

### ◈ Features
- **Interactive User Interface**: Enter an email and instantly classify it as spam or not.
- **Streamlit-Powered**: Provides a seamless and responsive web interface.
- **Pre-Trained ML Model**: Uses a Naive Bayes classifier for efficient and accurate predictions.
- **Custom Vectorization**: Email text is preprocessed with a TF-IDF vectorizer to feed the model.

<br>

### ◈ Documentation

#### ▪ How to Run Locally
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/AvishkarPatil/InBoxSpam.git
    cd InBoxSpam
    ```
2. **Install Dependencies**:
    Ensure you have Python installed. Run the following to install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. **Start the Application**:
    Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4. **Classify Emails**:
    - Enter email text in the text box provided.
    - Click **Classify** to check if the email is spam or not.


#### ▪ How to Host on Streamlit
1. **Create a Streamlit Account**:
    - Go to [Streamlit](https://streamlit.io/) and sign up for an account.
2. **Deploy the Application**:
    - In your Streamlit dashboard, click on **New app**.
    - Connect your GitHub repository.
    - Select the branch and the main file (`app.py`).
    - Click **Deploy**.
<br>

----- 

<h4 align='center'>© 2024 Avishkar Patil</h4>
