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

df = pd.read_csv("phi2_preds_dataset_larger.csv")

def create_features(df):
    df["step_count"] = df["answer"].apply(lambda x: len(x.split("\n")))
    df["has_division"] = df["answer"].str.contains(r"÷|/|divided by", regex=True).astype(int)
    
    df["lexical_overlap"] = df.apply(
        lambda row: len(set(row["question"].lower().split()) & set(row["answer"].lower().split())) 
        / len(set(row["question"].lower().split()) | set(row["answer"].lower().split())), 
        axis=1
    )
    
    df["operator_count"] = df["answer"].str.count(r'(\+|\-|\*|/|=|÷|\^)')
    df["has_fraction"] = df["answer"].str.contains(r'\d+/\d+').astype(int)
    df["equation_depth"] = df["answer"].apply(
        lambda x: max([s.count('(') + s.count(')') for s in x.split('\n')])
    )
    
    df["full_text"] = df["question"] + " " + df["answer"]
    tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=5000, stop_words='english')
    tfidf_features = tfidf.fit_transform(df["full_text"])
    
    numeric_features = df[["step_count", "has_division", "lexical_overlap",
                          "operator_count", "has_fraction", "equation_depth"]]
    return sparse.hstack([numeric_features, tfidf_features])

X = create_features(df)
y = df["is_correct"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

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
rf_search.fit(X_train, y_train)

# print best parameters
print("Best parameters found: ", rf_search.best_params_)

selector = SelectFromModel(rf_search.best_estimator_, max_features=100)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

stack = StackingClassifier(
    estimators=[
        ('rf', rf_search.best_estimator_),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ],
    final_estimator=LogisticRegression(),
    stack_method='predict_proba'
)
stack.fit(X_train_sel, y_train)

# Evaluation
y_pred = stack.predict(X_test_sel)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nFinal F1 Score: {f1_score(y_test, y_pred):.2f}")


from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
import re
import numpy as np
from tqdm import tqdm
import torch
import random
import pickle
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import time

def get_dataset():
    train_dataset = load_dataset("openai/gsm8k", "main", split='train')
    test_dataset = load_dataset("openai/gsm8k", "main", split='test')
    return train_dataset, test_dataset

temp_set = get_dataset()
gsm8k_dataset = {
    'train': temp_set[0],
    'test': temp_set[1]
}
num_problems = 500
start_idx = 2001
subset = gsm8k_dataset['train'].select(range(start_idx, start_idx+num_problems))
indices = list(range(start_idx, start_idx + num_problems))
subset_df = pd.DataFrame({
    'index': indices,
    'question': [item['question'] for item in subset],
    'answer': [item['answer'] for item in subset]
})

# Create features for subset (using same preprocessing pipeline)
X_subset = create_features(subset_df)

# Transform using feature selector from trained model
X_subset_sel = selector.transform(X_subset)

# Generate predictions
subset_preds = stack.predict(X_subset_sel)

# Add predictions to DataFrame and save
subset_df['label'] = subset_preds
subset_df.to_csv('gsm8k_difficulty_preds.csv', index=False)
