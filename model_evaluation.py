from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def get_baseline_preprocessing(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str]
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Базовый препроцессинг"""
    
    X = df[numerical_features + categorical_features]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
        
    numeric_pipeline = Pipeline([("scaler", StandardScaler(with_mean=False))])
    categorical_pipeline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder='drop',
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    return X_train_proc, X_test_proc, y_train, y_test


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """Обучает модель и возвращает метрики"""
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }


def get_baseline_models() -> Dict[str, Any]:
    """Возвращает словарь бейслайн моделей"""

    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'RandomForestClassifier' : RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, 
            class_weight='balanced', n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=20, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            random_state=42, verbose=-1
        )
    }


def run_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_name: str
) -> List[Dict[str, Any]]:
    """Запускает все модели и возвращает метрики"""

    print(f"\n{'='*60}")
    print(f"ЭКСПЕРИМЕНТ: {experiment_name}")
    print(f"{'='*60}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    models = get_baseline_models()
    results: List[Dict[str, Any]] = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
    
    return results
