# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier

# # Load data
# df = pd.read_csv("phi2_preds_dataset_large.csv")

# # Feature 1: Step count
# df["step_count"] = df["answer"].apply(lambda x: len(x.split("\n")))

# # Feature 2: Presence of division
# df["has_division"] = df["answer"].str.contains(r"÷|/|divided by", regex=True).astype(int)

# # Feature 3: Question-solution lexical overlap
# def jaccard_similarity(row):
#     question_words = set(row["question"].lower().split())
#     solution_words = set(row["answer"].lower().split())
#     intersection = question_words & solution_words
#     union = question_words | solution_words
#     return len(intersection) / len(union) if union else 0

# df["lexical_overlap"] = df.apply(jaccard_similarity, axis=1)

# # Train classifier
# X = df[["step_count", "has_division", "lexical_overlap"]]
# y = df["is_correct"]
# clf = RandomForestClassifier()

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, classification_report

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf.fit(X_train, y_train)

# # Predict & evaluate
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy import sparse

# Load and preprocess data
df = pd.read_csv("phi2_preds_dataset_large.csv")

# Feature engineering pipeline
# def create_features(df):
#     # Step-based features
#     df["step_count"] = df["answer"].apply(lambda x: len(x.split("\n")))
#     df["has_division"] = df["answer"].str.contains(r"÷|/|divided by", regex=True).astype(int)
    
#     # Text similarity
#     df["lexical_overlap"] = df.apply(
#         lambda row: len(set(row["question"].lower().split()) & set(row["answer"].lower().split())) 
#         / len(set(row["question"].lower().split()) | set(row["answer"].lower().split())), 
#         axis=1
#     )
    
#     # Math complexity features
#     df["operator_count"] = df["answer"].str.count(r'(\+|\-|\*|/|=|÷|\^)')
#     df["has_fraction"] = df["answer"].str.contains(r'\d+/\d+').astype(int)
#     df["equation_depth"] = df["answer"].apply(
#         lambda x: max([s.count('(') + s.count(')') for s in x.split('\n')])
#     )
    
#     # TF-IDF features
#     df["full_text"] = df["question"] + " " + df["answer"]
#     tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=5000, stop_words='english')
#     tfidf_features = tfidf.fit_transform(df["full_text"])
    
#     # Combine all features
#     numeric_features = df[["step_count", "has_division", "lexical_overlap",
#                           "operator_count", "has_fraction", "equation_depth"]]
#     return sparse.hstack([numeric_features, tfidf_features])

# # Create feature matrix and target
# X = create_features(df)
# y = df["is_correct"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Hyperparameter tuning
# rf_search = RandomizedSearchCV(
#     RandomForestClassifier(),
#     {
#         'n_estimators': [200, 400, 600],
#         'max_depth': [None, 10, 30, 50],
#         'min_samples_split': [2, 5, 10],
#         'max_features': ['sqrt', 'log2'],
#         'class_weight': ['balanced', None]
#     },
#     n_iter=50,
#     cv=5,
#     scoring='f1'
# )
# rf_search.fit(X_train, y_train)

# # Feature selection
# selector = SelectFromModel(rf_search.best_estimator_, max_features=100)
# X_train_sel = selector.fit_transform(X_train, y_train)
# X_test_sel = selector.transform(X_test)

# # Ensemble model
# stack = StackingClassifier(
#     estimators=[
#         ('rf', rf_search.best_estimator_),
#         ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
#     ],
#     final_estimator=LogisticRegression(),
#     stack_method='predict_proba'
# )
# stack.fit(X_train_sel, y_train)

# # Evaluation
# y_pred = stack.predict(X_test_sel)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print(f"\nFinal F1 Score: {f1_score(y_test, y_pred):.2f}")

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier

# # Load data
# df = pd.read_csv("phi2_preds_dataset_large.csv")

# # Feature 1: Step count
# df["step_count"] = df["answer"].apply(lambda x: len(x.split("\n")))

# # Feature 2: Presence of division
# df["has_division"] = df["answer"].str.contains(r"÷|/|divided by", regex=True).astype(int)

# # Feature 3: Question-solution lexical overlap
# def jaccard_similarity(row):
#     question_words = set(row["question"].lower().split())
#     solution_words = set(row["answer"].lower().split())
#     intersection = question_words & solution_words
#     union = question_words | solution_words
#     return len(intersection) / len(union) if union else 0

# df["lexical_overlap"] = df.apply(jaccard_similarity, axis=1)

# # Train classifier
# X = df[["step_count", "has_division", "lexical_overlap"]]
# y = df["is_correct"]
# clf = RandomForestClassifier()

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, classification_report

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf.fit(X_train, y_train)

# # Predict & evaluate
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy import sparse
import numpy as np

# Load and preprocess data
df = pd.read_csv("phi2_preds_dataset_large.csv")

# Feature engineering pipeline
# def create_features(df):
#     # Step-based features
#     df["step_count"] = df["answer"].apply(lambda x: len(x.split("\n")))
#     df["has_division"] = df["answer"].str.contains(r"÷|/|divided by", regex=True).astype(int)
    
#     # Text similarity
#     df["lexical_overlap"] = df.apply(
#         lambda row: len(set(row["question"].lower().split()) & set(row["answer"].lower().split())) 
#         / len(set(row["question"].lower().split()) | set(row["answer"].lower().split())), 
#         axis=1
#     )
    
#     # Math complexity features
#     df["operator_count"] = df["answer"].str.count(r'(\+|\-|\*|/|=|÷|\^)')
#     df["has_fraction"] = df["answer"].str.contains(r'\d+/\d+').astype(int)
#     df["equation_depth"] = df["answer"].apply(
#         lambda x: max([s.count('(') + s.count(')') for s in x.split('\n')])
#     )
    
#     # TF-IDF features
#     df["full_text"] = df["question"] + " " + df["answer"]
#     tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=5000, stop_words='english')
#     tfidf_features = tfidf.fit_transform(df["full_text"])
    
#     # Combine all features
#     numeric_features = df[["step_count", "has_division", "lexical_overlap",
#                           "operator_count", "has_fraction", "equation_depth"]]
#     return sparse.hstack([numeric_features, tfidf_features])

def create_features(df):
    # Step complexity features
    df["step_count"] = df["answer"].apply(lambda x: len(x.split("\n")))
    df["avg_step_length"] = df["answer"].apply(lambda x: sum(len(s) for s in x.split("\n"))/len(x.split("\n")))
    
    # Mathematical operation features
    operator_regex = r'(\+{1,}|\-{1,}|\*{1,}|/{1,}|÷|=|^)'
    df["operator_count"] = df["answer"].str.count(operator_regex)
    df["has_division"] = df["answer"].str.contains(r"÷|/|divided by", regex=True).astype(int)
    df["has_fraction"] = df["answer"].str.contains(r'\b\d+/\d+\b').astype(int)
    
    # Structural complexity
    df["equation_depth"] = df["answer"].apply(
        lambda x: max([s.count('(') + s.count(')') for s in x.split('\n')])
    )
    
    # Enhanced text features
    df["question_length"] = df["question"].apply(len)
    df["answer_length"] = df["answer"].apply(len)
    
    # TF-IDF with problem-specific preprocessing
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]+\b|\d+\.?\d*|[\+\-\*/\(\)]'
    )
    tfidf_features = tfidf.fit_transform(df["question"] + " " + df["answer"])
    
    # Combine features
    numeric_features = df[[
        "step_count", "avg_step_length", "operator_count",
        "has_division", "has_fraction", "equation_depth",
        "question_length", "answer_length"
    ]]
    return sparse.hstack([numeric_features, tfidf_features])

# Create feature matrix and target
X = create_features(df)
y = df["is_correct"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_dist = {
    'n_estimators': [400, 600, 800],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 0.33],
    'class_weight': [{False:1, True:2}, 'balanced']
}

# Hyperparameter tuning
rf_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=40,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
rf_search.fit(X_train, y_train)

# Feature selection
selector = SelectFromModel(
    rf_search.best_estimator_,
    threshold="1.25*median",  # More conservative feature selection
    max_features=150
)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# Final model with class weights
final_model = RandomForestClassifier(
    **rf_search.best_params_,
    n_jobs=-1
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_sel, y_train, 
    test_size=0.25, 
    stratify=y_train,
    random_state=42
)

# Train initial model
final_model.fit(X_train_sub, y_train_sub)


# final_model.fit(X_train_sel, y_train)

# # Threshold-optimized predictions
# y_probs = final_model.predict_proba(X_test_sel)[:, 1]
# optimal_threshold = 0.42  # Determined via validation set
# y_pred = y_probs > optimal_threshold

y_val_probs = final_model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.1, 0.9, 100)
f1_scores = [f1_score(y_val, y_val_probs >= t, pos_label=True) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Retrain on full training data
final_model.fit(X_train_sel, y_train)

# Final evaluation
y_test_probs = final_model.predict_proba(X_test_sel)[:, 1]
y_pred = y_test_probs >= optimal_threshold

# Evaluation
print("Optimized Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nBalanced F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
print(f"True Class F1: {f1_score(y_test, y_pred, pos_label=True):.2f}")

