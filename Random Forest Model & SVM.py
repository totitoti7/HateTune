#Random Forest Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset from CSV file
data = pd.read_csv('/Users/fatmaalzhraemadmohamed/Documents/New_Clean_dataset.csv')

# Handling missing values by replacing NaN with an empty string
data['hatespeech'] = data['hatespeech'].fillna('')

####################### Statistics #####################

# Basic statistics about the dataset
num_samples = len(data)
num_hate_speech = (data['index'] == 1).sum()
num_neutral = (data['index'] == 0).sum()

print("Total samples:", num_samples)
print("Number of hate speech samples:", num_hate_speech)
print("Number of neutral samples:", num_neutral)

# Descriptive statistics for text lengths
data['text_length'] = data['hatespeech'].apply(len)
mean_length = np.mean(data['text_length'])
max_length = np.max(data['text_length'])
min_length = np.min(data['text_length'])

print("Mean text length:", mean_length)
print("Maximum text length:", max_length)
print("Minimum text length:", min_length)

# Average sentence length
data['Sentence Length'] = data['hatespeech'].apply(lambda x: len(x.split()))
avg_sentence_length = data['Sentence Length'].mean()
print(f'Average Sentence Length: {avg_sentence_length:.2f} words')

# Separate hate and neutral comments
hate_comments = data[data['index'] == 1]['hatespeech']
neutral_comments = data[data['index'] == 0]['hatespeech']

# Total Word Count
total_word_count_hate = sum(hate_comments.str.split().apply(len))
total_word_count_neutral = sum(neutral_comments.str.split().apply(len))

# Vocabulary Size
def get_vocabulary_size(text):
    words = [word for word in text.split()]
    return len(set(words))

vocabulary_size_hate = get_vocabulary_size(' '.join(hate_comments))
vocabulary_size_neutral = get_vocabulary_size(' '.join(neutral_comments))

print("Total Word Count (Hate):", total_word_count_hate)
print("Total Word Count (Neutral):", total_word_count_neutral)
print("Vocabulary Size (Hate):", vocabulary_size_hate)
print("Vocabulary Size (Neutral):", vocabulary_size_neutral)

# Distribution of Ten Most Frequent Terms
def get_most_frequent_terms(text, n=10):
    words = [word for word in text.split()]
    word_freq = Counter(words)
    return word_freq.most_common(n)

most_frequent_terms_hate = get_most_frequent_terms(' '.join(hate_comments))
most_frequent_terms_neutral = get_most_frequent_terms(' '.join(neutral_comments))

print("Top 10 Most Frequent Terms (Hate):", most_frequent_terms_hate)
print("Top 10 Most Frequent Terms (Neutral):", most_frequent_terms_neutral)

# Word Cloud
def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Create Word Cloud for Hate
create_word_cloud(' '.join(hate_comments))

# Create Word Cloud for Neutral
create_word_cloud(' '.join(neutral_comments))

####################### Trianing #####################
# # Splitting the data into features (X) and labels (y)
# X = data['hatespeech']
# y = data['index']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##########Random Forest
# # Vectorize the text data using CountVectorizer
# vectorizer = CountVectorizer()
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

# # Train a Random Forest classifier
# random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest.fit(X_train_vectorized, y_train)

# # Make predictions on the test set
# y_pred = random_forest.predict(X_test_vectorized)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Print classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Print confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)


#############SVM 
# # Vectorize the text data using TF-IDF
# vectorizer = TfidfVectorizer()
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

# # Train an SVM classifier
# svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
# svm_classifier.fit(X_train_vectorized, y_train)

# # Make predictions on the test set
# y_pred = svm_classifier.predict(X_test_vectorized)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Print classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Print confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)



