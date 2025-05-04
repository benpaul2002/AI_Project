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
def create_features(df):
    # Step-based features
    df["step_count"] = df["answer"].apply(lambda x: len(x.split("\n")))
    df["has_division"] = df["answer"].str.contains(r"÷|/|divided by", regex=True).astype(int)
    
    # Text similarity
    df["lexical_overlap"] = df.apply(
        lambda row: len(set(row["question"].lower().split()) & set(row["answer"].lower().split())) 
        / len(set(row["question"].lower().split()) | set(row["answer"].lower().split())), 
        axis=1
    )
    
    # Math complexity features
    df["operator_count"] = df["answer"].str.count(r'(\+|\-|\*|/|=|÷|\^)')
    df["has_fraction"] = df["answer"].str.contains(r'\d+/\d+').astype(int)
    df["equation_depth"] = df["answer"].apply(
        lambda x: max([s.count('(') + s.count(')') for s in x.split('\n')])
    )
    
    # TF-IDF features
    df["full_text"] = df["question"] + " " + df["answer"]
    tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=5000, stop_words='english')
    tfidf_features = tfidf.fit_transform(df["full_text"])
    
    # Combine all features
    numeric_features = df[["step_count", "has_division", "lexical_overlap",
                          "operator_count", "has_fraction", "equation_depth"]]
    return sparse.hstack([numeric_features, tfidf_features])

# Create feature matrix and target
X = create_features(df)
y = df["is_correct"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning
rf_search = RandomizedSearchCV(
    RandomForestClassifier(),
    {
        'n_estimators': [200, 400, 600],
        'max_depth': [None, 10, 30, 50],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    },
    n_iter=50,
    cv=5,
    scoring='f1'
)
rf_search.fit(X_res, y_res)

# Feature selection
selector = SelectFromModel(rf_search.best_estimator_, max_features=100)
X_train_sel = selector.fit_transform(X_res, y_res)
X_test_sel = selector.transform(X_test)

# Ensemble model
stack = StackingClassifier(
    estimators=[
        ('rf', rf_search.best_estimator_),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ],
    final_estimator=LogisticRegression(),
    stack_method='predict_proba'
)
stack.fit(X_train_sel, y_res)

# Evaluation
y_pred = stack.predict(X_test_sel)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nFinal F1 Score: {f1_score(y_test, y_pred):.2f}")
