"""
Unit tests for preprocessing module.

Run with: pytest tests/test_preprocessing.py
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import (
    create_preprocessor,
    split_data,
    apply_smote
)


class TestPreprocessing:
    """Test suite for preprocessing functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'num_feature_1': np.random.randn(n_samples),
            'num_feature_2': np.random.randn(n_samples) * 10,
            'cat_feature_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'cat_feature_2': np.random.choice(['X', 'Y'], n_samples)
        })
        
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
        
        return X, y
    
    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        num_features = ['num1', 'num2']
        cat_features = ['cat1', 'cat2']
        
        preprocessor = create_preprocessor(num_features, cat_features)
        
        assert preprocessor is not None
        assert len(preprocessor.transformers) == 2
    
    def test_split_data_shapes(self, sample_data):
        """Test that data splits have correct shapes."""
        X, y = sample_data
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.2, validation_size=0.15, random_state=42
        )
        
        # Check total samples
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)
        
        # Check proportions (approximately)
        assert len(X_test) / len(X) == pytest.approx(0.2, abs=0.02)
        assert len(X_val) / len(X) == pytest.approx(0.15, abs=0.02)
    
    def test_split_data_stratification(self, sample_data):
        """Test that stratification maintains class distribution."""
        X, y = sample_data
        
        original_dist = y.value_counts(normalize=True)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, stratify=True, random_state=42
        )
        
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        # Distributions should be similar (within 5%)
        for cls in [0, 1]:
            assert abs(original_dist[cls] - train_dist[cls]) < 0.05
            assert abs(original_dist[cls] - test_dist[cls]) < 0.05
    
    def test_smote_balances_classes(self, sample_data):
        """Test that SMOTE balances class distribution."""
        X, y = sample_data
        
        # Create imbalanced data
        X_imbalanced = X[y == 0].iloc[:700].append(X[y == 1].iloc[:100])
        y_imbalanced = np.array([0] * 700 + [1] * 100)
        
        X_resampled, y_resampled = apply_smote(
            X_imbalanced.values, y_imbalanced, random_state=42
        )
        
        # After SMOTE, classes should be balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1]  # Perfectly balanced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
