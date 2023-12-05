from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import torch.nn as nn
import torch
import numpy as np

# Create a Flask application
app = Flask(__name__)

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize result_text variables
    input_sentence = None
    binary_classification_b_e= None
    binary_classification_b_m= None
    binary_classification_b_t= None
    binary_classification_e_m= None
    binary_classification_t_e= None
    binary_classification_t_m= None
    multi_class_classification = None
    category_results = []

    
    # Check if the form is submitted
    if request.method == 'POST':
        # Get the text entered in the input field
        user_text = request.form['user_text']

        # Input sentence
        input_sentence = f"Input sentence: {user_text}"

        # Load the saved model
        with open('models/tasktwo_b_e.pkl', 'rb') as model_file:
            loaded_classifier = pickle.load(model_file)
        with open('vectorizer/tasktwo_b_e.pkl', 'rb') as vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(vectorizer_file)
        X_new_tfidf = loaded_tfidf_vectorizer.transform([user_text])
        binary_classification_b_e = loaded_classifier.predict(X_new_tfidf)


        with open('models/tasktwo_b_m.pkl', 'rb') as model_file:
            loaded_classifier = pickle.load(model_file)
        with open('vectorizer/tasktwo_b_m.pkl', 'rb') as vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(vectorizer_file)
        X_new_tfidf = loaded_tfidf_vectorizer.transform([user_text])
        binary_classification_b_m = loaded_classifier.predict(X_new_tfidf)


        with open('models/tasktwo_b_t.pkl', 'rb') as model_file:
            loaded_classifier = pickle.load(model_file)
        with open('vectorizer/tasktwo_b_t.pkl', 'rb') as vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(vectorizer_file)
        X_new_tfidf = loaded_tfidf_vectorizer.transform([user_text])
        binary_classification_b_t = loaded_classifier.predict(X_new_tfidf)


        with open('models/tasktwo_e_m.pkl', 'rb') as model_file:
            loaded_classifier = pickle.load(model_file)
        with open('vectorizer/tasktwo_e_m.pkl', 'rb') as vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(vectorizer_file)
        X_new_tfidf = loaded_tfidf_vectorizer.transform([user_text])
        binary_classification_e_m = loaded_classifier.predict(X_new_tfidf)


        with open('models/tasktwo_t_e.pkl', 'rb') as model_file:
            loaded_classifier = pickle.load(model_file)
        with open('vectorizer/tasktwo_t_e.pkl', 'rb') as vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(vectorizer_file)
        X_new_tfidf = loaded_tfidf_vectorizer.transform([user_text])
        binary_classification_t_e = loaded_classifier.predict(X_new_tfidf)


        with open('models/tasktwo_t_m.pkl', 'rb') as model_file:
            loaded_classifier = pickle.load(model_file)
        with open('vectorizer/tasktwo_t_m.pkl', 'rb') as vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(vectorizer_file)
        X_new_tfidf = loaded_tfidf_vectorizer.transform([user_text])
        binary_classification_t_m = loaded_classifier.predict(X_new_tfidf) 

        # Process the original text
        binary_classification_b_e = f"Binary Classification b vs e: {'b' if binary_classification_b_e[0] == 1 else 'e'}"
        binary_classification_b_m = f"Binary Classification b vs m: {'b' if binary_classification_b_m[0] == 1 else 'm'}"
        binary_classification_b_t = f"Binary Classification b vs t: {'b' if binary_classification_b_t[0] == 1 else 't'}"
        binary_classification_e_m = f"Binary Classification e vs m: {'e' if binary_classification_e_m[0] == 1 else 'm'}"
        binary_classification_t_e = f"Binary Classification t vs e: {'t' if binary_classification_t_e[0] == 1 else 'e'}"
        binary_classification_t_m = f"Binary Classification t vs m: {'t' if binary_classification_t_m[0] == 1 else 'm'}"


        ##################################################################################################################
        # Task Three
        ##################################################################################################################

        # this does not work as we could not figure out how to vectorize our input string in time. We always had different tensor sizes and got an error
        if False:
            class NewsNN(nn.Module):
                def __init__(self, input_size, hidden_size, num_classes):
                    super(NewsNN, self).__init__()
                    self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
                    self.relu = nn.ReLU()
                    self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
                    self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

                def forward(self, x):
                    out = self.layer_1(x)
                    out = self.relu(out)
                    out = self.layer_2(out)
                    out = self.relu(out)
                    out = self.output_layer(out)
                    return out
            
            hidden_size = 100  # 1st layer and 2nd layer number of feature
            num_classes = 4

            vocabulary = {char.lower(): idx for idx, char in enumerate(set(user_text.lower()))}
            input_size = len(vocabulary)
            input_size = 135402

            # Convert user text to one-hot encoded tensor
            user_tensor = torch.tensor([vocabulary.get(char.lower(), 0) for char in user_text.lower()], dtype=torch.long)
            user_tensor = torch.nn.functional.one_hot(user_tensor, num_classes=len(vocabulary)).float()

            # Add an extra dimension to simulate the batch (even if there is only one text)
            user_tensor = user_tensor.unsqueeze(0)

            model = NewsNN(input_size, hidden_size, num_classes)
            model.load_state_dict(torch.load("./models/taskthree.pth"))
            model.eval()

            # Make predictions
            with torch.no_grad():
                output = model(user_tensor)



            # user_text = torch.FloatTensor([user_text]).to(device)
            # output = model(user_tensor)
            prediction = int(torch.max(output.data, 1)[1].numpy())
            # Process the reversed text
            multi_class_classification = f"Multi-Class Classification: {prediction}"


        
    # Render the home page template
    return render_template('index.html', input_sentence=input_sentence, binary_classification_b_e=binary_classification_b_e, binary_classification_b_m=binary_classification_b_m, binary_classification_b_t=binary_classification_b_t, binary_classification_e_m=binary_classification_e_m, binary_classification_t_e=binary_classification_t_e, binary_classification_t_m=binary_classification_t_m, category_results=category_results)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
