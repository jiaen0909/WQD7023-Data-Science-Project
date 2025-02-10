import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary

vectorizer = CountVectorizer(max_features=10000, max_df=0.95, min_df=2)

dtm_sparse = vectorizer.fit_transform(model3_df['processed_text'])

# Create LDA dictionary and corpus from DTM
texts = model3_df['processed_text'].apply(lambda x: x.split()).tolist()
dictionary = Dictionary(texts)
corpus = Sparse2Corpus(dtm_sparse, documents_columns=False)

# Function to compute coherence scores and get the best LDA model
def get_best_lda_model(corpus, dictionary, texts, num_topics_list):
    """
    Train LDA models for given topics and return the model with the highest coherence score.
    """
    coherence_scores = {}
    best_model = None
    highest_coherence = -1
    best_num_topics = None

    for num_topics in num_topics_list:
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores[num_topics] = coherence_score
        print(f"Number of Topics: {num_topics}, Coherence Score:{coherence_score:.4f}")

        if coherence_score > highest_coherence:
            highest_coherence = coherence_score
            best_model = lda_model
            best_num_topics = num_topics

    return best_model, highest_coherence, best_num_topics, coherence_scores

# Define number of topics and compute the best model
num_topics_list = [3, 5, 7, 10]
best_lda_model, highest_coherence, best_num_topics, coherence_scores = get_best_lda_model(corpus, dictionary, texts, num_topics_list)

# Print coherence scores and highest coherence
print("Coherence Scores:", coherence_scores)
print(f"Best Number of Topics: {best_num_topics} with Coherence Score: {highest_coherence:.4f}")

def tune_alpha(corpus, dictionary, texts, num_topics, alpha_list, beta='symmetric'):
    """
    Compare different alpha values and return the best alpha with the highest coherence score.
    """
    coherence_scores = {}
    best_model = None
    highest_coherence = -1
    best_alpha = None

    for alpha in alpha_list:
        # Train the LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            random_state=42,
            alpha=alpha,
            eta=beta
        )

        # Compute coherence score
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores[alpha] = coherence_score
        print(f"Alpha: {alpha}, Coherence Score:{coherence_score:.4f}")

        # Update the best model
        if coherence_score > highest_coherence:
            highest_coherence = coherence_score
            best_model = lda_model
            best_alpha = alpha

    return best_model, best_alpha, highest_coherence

alpha_list = ['symmetric', 'asymmetric', 0.01, 0.1, 1.0]
best_model_alpha, best_alpha, coherence_alpha = tune_alpha(
    corpus=corpus,
    dictionary=dictionary,
    texts=texts,
    num_topics=5,
    alpha_list=alpha_list
)
print("Coherence Scores:", coherence_scores)
print(f"Best Alpha: {best_alpha}, Coherence: {coherence_alpha}")

def tune_beta(corpus, dictionary, texts, num_topics, alpha, beta_list):
    """
    Compare different beta values and return the best beta with the highest coherence score.
    """
    coherence_scores = {}
    best_model = None
    highest_coherence = -1
    best_beta = None

    for beta in beta_list:
        # Train the LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            random_state=42,
            alpha=alpha,
            eta=beta
        )

        # Compute coherence score
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores[beta] = coherence_score
        print(f"Beta: {beta}, Coherence Score:{coherence_score:.8f}")

        # Update the best model
        if coherence_score > highest_coherence:
            highest_coherence = coherence_score
            best_model = lda_model
            best_beta = beta

    return best_model, best_beta, highest_coherence

beta_list = ['symmetric', 0.01, 0.1, 1.0]
best_model_beta, best_beta, coherence_beta = tune_beta(
    corpus=corpus,
    dictionary=dictionary,
    texts=texts,
    num_topics=5,
    alpha=best_alpha,
    beta_list=beta_list
)
print(f"Best Beta: {best_beta}, Coherence: {coherence_beta}")

# Function to display complaints by topic
def display_complaints_for_best_topics(lda_model, corpus, narrative_column, num_samples=3):
    """
    Display sample complaints for the best LDA model (highest coherence score).
    """
    topic_probabilities = [lda_model.get_document_topics(doc, minimum_probability=0.0) for doc in corpus]
    dominant_topics = [max(prob, key=lambda x: x[1])[0] for prob in topic_probabilities]

    df_topics = pd.DataFrame({
        "Complaint": narrative_column,
        "Dominant_Topic": dominant_topics
    })

    for topic_num in range(lda_model.num_topics):
        print(f"\nTopic {topic_num}:\n{'-' * 40}")
        topic_complaints = df_topics[df_topics["Dominant_Topic"] == topic_num]["Complaint"]

        if not topic_complaints.empty:
            sample_complaints = topic_complaints.sample(min(num_samples, len(topic_complaints)))
            for complaint in sample_complaints:
                print(f"- {complaint}")
        else:
            print("No complaints found for this topic.")

for i, topic in best_model_beta.show_topics(formatted=True, num_topics=5, num_words=5):
  print(f"Topic {i}: {topic}")
