# Import
import pandas as pd
import numpy as np
import re
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Read Dataset
df=pd.read_csv("full complaints.csv")
df.head(5)
df["Date received"]=pd.to_datetime(df["Date received"])

start_date = "2023-07-01"
end_date = "2024-06-30"
df1 = df[(df["Date received"] >= start_date) & (df["Date received"] <= end_date)]
df2 = df1.dropna(subset=['Consumer complaint narrative'])
narrative = df2[["Consumer complaint narrative"]].copy()
narrative = narrative.drop_duplicates(subset=['Consumer complaint narrative'])
print(len(narrative))

# Masked info removal
def remove_masked_info(text):
    # Remove placeholder patterns like 'XXXX', 'XX/XX/XXXX', 'XX', etc.
    text = re.sub(r'X+|XX/XX/XXXX|scrub>|\{\$[\d,\.]+\}', '', text)
    return text.strip()

narrative['cleaned_narrative'] = narrative['Consumer complaint narrative'].apply(remove_masked_info)

# Preprocessing
# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(["would", "please", "also"])

def preprocess_text(text):
    # Convert text to lower case
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and numbers
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

narrative['tokens'] = narrative['cleaned_narrative'].apply(preprocess_text)
narrative['processed_text'] = narrative['tokens'].apply(lambda tokens: ' '.join(tokens))

print(narrative.head(10))

# VADER Sentiment Label
vader=narrative.copy()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentimentAnalyser = SentimentIntensityAnalyzer()

def calculate_sentiment(text):
    scores = sentimentAnalyser.polarity_scores(text)
    compound_score = scores['compound']
    sentiment_label = 'Negative' if compound_score < -0.1 else 'Positive' if compound_score > 0.1 else 'Neutral'
    urgency_label = 'Urgent' if compound_score < 0 else 'Normal'
    return compound_score, sentiment_label, urgency_label

vader[['vader_score', 'vader_label', 'vader_urgency']] = vader['processed_text'].apply(lambda x: pd.Series(calculate_sentiment(x)))

def binary_class(text):
  if text == 'Urgent':
    binary = 1
  else:
    binary = 0
  return binary

vader['vader_class'] = vader['vader_urgency'].apply(binary_class)
print(vader.head(5))

# Calculate length of each complaint narrative
vader['narrative_length1'] = vader['Consumer complaint narrative'].apply(lambda x: len(x.split()))
vader['narrative_length2'] = vader['processed_text'].apply(lambda x: len(x.split()))
vader = vader.dropna(subset=['processed_text'])
vader = vader.drop_duplicates(subset=['processed_text'])

# Descriptive Statistics
numeric_stats = vader.describe()
print(numeric_stats)

# Histogram for continuous variables
vader.hist(figsize=(10, 5), bins=5, color='skyblue', edgecolor='black')
# Show the plot
plt.tight_layout()  # Adjust layout
plt.show()

# Boxplot Wordcount before preprocessing
plt.figure(figsize=(10, 6))
sns.boxplot(data=vader, x='narrative_length1', color='skyblue')
plt.title('Boxplot of Word Count in Consumer Complaints before Preprocessing')
plt.xlabel('Number of Words')
plt.show()

# Boxplot Wordcount after preprocessing
plt.figure(figsize=(10, 6))
sns.boxplot(data=vader, x='narrative_length2', color='skyblue')
plt.title('Boxplot of Word Count in Consumer Complaints After Preprocessing')
plt.xlabel('Number of Words')
plt.show()

# Boxplot Sentiment Score
plt.figure(figsize=(10, 6))
sns.boxplot(data=vader, x='vader_score', color='skyblue')
plt.title('Boxplot of Sentiment Score')
plt.xlabel('Sentiment Score')
plt.show()

# Plot histogram of narrative lengths
plt.figure(figsize=(12,6))
vader['narrative_length1'].plot(kind='hist', bins=100, color='blue', edgecolor='black')
plt.title('Distribution of Raw Complaint Narrative Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xlim(0, 1200)
plt.xticks(range(0, 1201, 50))
plt.show()

# Plot histogram of narrative lengths
plt.figure(figsize=(12,6))
vader['narrative_length2'].plot(kind='hist', bins=100, color='blue', edgecolor='black')
plt.title('Distribution of Processed Complaint Narrative Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xlim(0, 1200)
plt.xticks(range(0, 1201, 50))
plt.show()

# Plot histogram of sentiment scores
plt.figure(figsize=(8,6))
vader['vader_score'].plot(kind='hist', bins=50, color='green', edgecolor='black')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Plot bar chart of urgency
plt.figure(figsize=(8,6))
vader['vader_urgency'].value_counts().plot(kind='bar', color=['red', 'blue'])
plt.title('Urgency of Complaints')
plt.xlabel('Urgency Category')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=vader, x='narrative_length2', y='vader_score', alpha=0.5)

# Add a trend line for better visualization
sns.regplot(data=vader, x='narrative_length2', y='vader_score', scatter=False, color='red')

# Customize the plot
plt.title('Sentiment Score vs. Narrative Length')
plt.xlabel('Narrative Length (Word Count)')
plt.ylabel('Sentiment Score')
plt.xlim(0, 50)  # Focus on shorter narratives, adjust if needed
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=vader, x='narrative_length2', y='vader_score', alpha=0.5)

# Add a trend line for better visualization
sns.regplot(data=vader, x='narrative_length2', y='vader_score', scatter=False, color='red')

# Customize the plot
plt.title('Sentiment Score vs. Narrative Length')
plt.xlabel('Narrative Length (Word Count)')
plt.ylabel('Sentiment Score')
plt.xlim(0, 50)  # Focus on shorter narratives, adjust if needed
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=vader, x='narrative_length2', y='vader_score', alpha=0.5)

# Add a trend line for better visualization
sns.regplot(data=vader, x='narrative_length2', y='vader_score', scatter=False, color='red')

# Customize the plot
plt.title('Sentiment Score vs. Narrative Length')
plt.xlabel('Narrative Length (Word Count)')
plt.ylabel('Sentiment Score')
plt.xlim(0, 50)  # Focus on shorter narratives, adjust if needed
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=vader, x='narrative_length2', y='vader_score', alpha=0.5)

# Add a trend line for better visualization
sns.regplot(data=vader, x='narrative_length2', y='vader_score', scatter=False, color='red')

# Customize the plot
plt.title('Sentiment Score vs. Narrative Length')
plt.xlabel('Narrative Length (Word Count)')
plt.ylabel('Sentiment Score')
plt.xlim(0, 50)  # Focus on shorter narratives, adjust if needed
plt.grid()
plt.show()

# Narrative Length vs VADER Sentiment Score
plt.figure(figsize=(10,6))
plt.scatter(vader['narrative_length2'], vader['vader_score'], alpha=0.5, color='blue')
plt.title('Sentiment Score vs. Complaint Length')
plt.xlabel('Complaint Length (Number of Words)')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.show()
