A simple yet effective web application that classifies SMS messages as Spam or Not Spam using a trained machine learning model. Built with Python, Streamlit, and NLP techniques.

Features
Text Preprocessing: Cleans and stems text using NLTK.
ML Prediction: Uses a trained model to classify SMS as spam or not.
Interactive UI: Built with Streamlit for ease of use.
Spam Probability: Displays probabilities for both spam and not spam.
Preloaded Messages: Quickly test the model with example texts.

 Model Info
Vectorizer: TF-IDF (vectorizer.pkl)
Model: Trained classifier (likely Logistic Regression or Naive Bayes) saved as model.pkl.
Training Notebook: Included as sms-classifier.ipynb and sms-classifier-checkpoint.ipynb.

 File Structure
├── app2.py                    # Main Streamlit app
├── model.pkl                  # Trained ML model
├── vectorizer.pkl             # TF-IDF vectorizer
├── sms-classifier.ipynb       # Training and evaluation notebook
├── sms-classifier-checkpoint.ipynb  # Jupyter checkpoint

Setup Instructions
1. Clone the repo or download the files.
2. Install dependencies:
   pip install streamlit scikit-learn nltk
3. Run the app
   streamlit run app2.py
4. Navigate to http://localhost:8501 in your browser.

How It Works
  Text is cleaned:
  Lowercased
  Tokenized
  Stopwords and punctuation removed
  Stemmed using PorterStemmer
  Preprocessed text is vectorized using TF-IDF.
  The vector is passed into the trained model to predict spam likelihood.





