from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, roc_curve, auc, classification_report
from imblearn.over_sampling import SMOTE

X = vader['processed_text']
y = vader['vader_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_vec = tfidf_vectorizer.fit_transform(X_train)
X_test_vec = tfidf_vectorizer.transform(X_test)   

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)

def evaluate_train_3model(model):

    scoring_metrics3 = ['accuracy', 'precision', 'recall', 'f1']
    scores3 = cross_validate(model, X_train_smote, y_train_smote, cv=5, scoring=scoring_metrics3)

    mean3_accuracy = np.mean(scores3['test_accuracy']) * 100
    mean3_precision = np.mean(scores3['test_precision']) * 100
    mean3_recall = np.mean(scores3['test_recall']) * 100
    mean3_f1 = np.mean(scores3['test_f1']) * 100
    
    print(f"\nModel: {model.__class__.__name__}")
    print("Average accuracy: {:.4f}%".format(mean3_accuracy))
    print("Average precision: {:.4f}%".format(mean3_precision))
    print("Average recall: {:.4f}%".format(mean3_recall))
    print("Average F1-score: {:.4f}%".format(mean3_f1))
    
    return mean3_accuracy, mean3_precision, mean3_recall, mean3_f1

# Train Model
# Logistic Regression
lr = LogisticRegression(random_state=42)
evaluate_train_3model(lr)

# Random Forest
rf = RandomForestClassifier(random_state=42)
evaluate_train_3model(rf)

# SVM
svm = SVC(random_state=42)
evaluate_train_3model(svm)

# Hyperparameter Tuning Function
def find_best_parameters(model, param_grid):
  grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=5, verbose=1, n_jobs=-1, refit='F1')
  grid_search.fit(X_train_smote, y_train_smote)
  best_f1_score = grid_search.best_score_

  print("Best parameters:", grid_search.best_params_)
  print("Best cross-validation F1-score: {:.4f}%".format(grid_search.best_score_*100))
  return grid_search.best_estimator_, best_f1_score

# LR Tuning
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300, 400, 500]
}

lr_best_model, lr_best_f1 = find_best_parameters(lr, lr_param_grid)

# RF Tuning
rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
}

rf_best_model, rf_best_f1=find_best_parameters(rf, rf_param_grid)

# SVM Tuning
svm_param_grid = {
    'C': [0.1, 1.0, 2.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.1] 
}

svm_best_model, svm_best_f1 = find_best_parameters(svm, svm_param_grid)

# Testing Dataset Evaluation Function
def evaluate_test_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    print(f"\nModel: {model.__class__.__name__}")
    print('\nConfusion Matrix:')
    print(cm)
    print('\nTest Set Accuracy: {:.4f}%'.format(accuracy))
    print('Test Set Precision: {:.4f}%'.format(precision))
    print('Test Set Recall: {:.4f}%'.format(recall))
    print('Test Set F1-Score: {:.4f}%'.format(f1))

    return accuracy, precision, recall, f1

models = [lr_best_model,  rf_best_model, svm_best_model]
model_names = ['Logistic Regression', 'Random Forest', 'SVM']
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []

for model, name in zip(models, model_names):
    metrics = evaluate_test_model(model, X_test_vec, y_test)
    accuracy_values.append(metrics[0])
    precision_values.append(metrics[1])
    recall_values.append(metrics[2])
    f1_values.append(metrics[3])

# Plot ROC curves
for model, name in zip(models, model_names):
  y_pred_proba = model.predict_proba(X_test_vec)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_vec)
  fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
  roc_auc = auc(fpr, tpr)
  plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.show()
