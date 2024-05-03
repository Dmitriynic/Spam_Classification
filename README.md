# Spam Classification using TF-IDF Logistic regression and PyTorch

This project aims to classify text data into predefined categories using various natural language processing techniques, including lemmatization, stemming, and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. The classification is performed using both traditional machine learning algorithms like logistic regression and deep neural networks implemented using PyTorch.

## Data Preprocessing

1. **Tokenization:** The text data is tokenized into individual words or tokens.
2. **Lemmatization:** Words are lemmatized to convert them into their base or root form.
3. **Stemming:** The text is stemmed to reduce words to their stem or root form.
4. **TF-IDF Vectorization:** TF-IDF is used to convert text data into numerical vectors, representing the importance of each word in the document.

## Modeling

1. **Logistic Regression:** Logistic regression is applied as a baseline model for classification.
2. **Deep Neural Network (DNN):** A simple deep neural network is implemented using PyTorch to capture complex patterns in the text data.

## Training and Evaluation

1. **Cross-Validation:** The data is split into training and validation sets using k-fold cross-validation to evaluate model performance.
2. **Grid Search:** Grid search is employed to tune hyperparameters of the models for improved performance.

## Libraries Used

- NLTK: For text preprocessing tasks such as tokenization, lemmatization, and stemming.
- Scikit-learn: For implementing TF-IDF vectorization, logistic regression, cross-validation, and grid search.
- PyTorch: For building and training deep neural network models.

## Usage

1. Install the required libraries listed in the `requirements.txt` file.
2. Preprocess the text data using techniques like tokenization, lemmatization, and stemming.
3. Convert the preprocessed text data into TF-IDF vectors using Scikit-learn.
4. Train a logistic regression model on the TF-IDF vectors.
5. Implement a deep neural network using PyTorch for text classification.
6. Evaluate the performance of both models using cross-validation and grid search.
7. Choose the model with the best performance for classification tasks.

## References

- NLTK Documentation: [Link](https://www.nltk.org/)
- Scikit-learn Documentation: [Link](https://scikit-learn.org/stable/documentation.html)
- PyTorch Documentation: [Link](https://pytorch.org/docs/stable/index.html)

**Note:**
- This README provides an overview of the project and its components. For detailed implementation instructions, refer to the project documentation and source code.
- Additional preprocessing techniques and model architectures can be explored to further improve classification accuracy.

