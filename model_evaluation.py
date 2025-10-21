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

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency Encoder для категориальных признаков с высокой кардинальностью.
    Заменяет категории на их частоты, вычисленные на train данных.
    """
    def __init__(self):
        self.freq_maps_ = {}
    
    def fit(self, X, y=None):
        """Вычисляет частоты для каждого столбца на обучающих данных"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X_df.columns:
            # Вычисляем нормализованные частоты
            freq_map = X_df[col].value_counts(normalize=True).to_dict()
            self.freq_maps_[col] = freq_map
        
        return self
    
    def transform(self, X):
        """Применяет частотное кодирование"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        result = pd.DataFrame(index=X_df.index)
        
        for col in X_df.columns:
            # Применяем сохраненную карту частот
            freq_map = self.freq_maps_.get(col, {})
            result[f'{col}_freq'] = X_df[col].map(freq_map).fillna(0)
        
        return result.values


def get_baseline_preprocessing(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str]
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Базовый препроцессинг с time-based split и FrequencyEncoder"""
    
    df_sorted = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    split_index = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()
    
    categorical_high = ['merchant', 'job']
    categorical_low = ['category', 'gender']
    
    categorical_high_filtered = [col for col in categorical_high if col in categorical_features]
    categorical_low_filtered = [col for col in categorical_low if col in categorical_features]
    
    all_features = numerical_features + categorical_low_filtered + categorical_high_filtered
    
    X_train = train_df[all_features]
    y_train = train_df['is_fraud']
    X_test = test_df[all_features]
    y_test = test_df['is_fraud']
    
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False))
    ])
    
    categorical_low_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    
    categorical_high_pipeline = Pipeline([
        ("freq_encoder", FrequencyEncoder())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat_low", categorical_low_pipeline, categorical_low_filtered),
            ("cat_high", categorical_high_pipeline, categorical_high_filtered),
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
