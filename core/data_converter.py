"""
Data Converter Utility

Xử lý chuyển đổi giữa NumPy arrays và Pandas DataFrames
để đảm bảo tính nhất quán trong framework

Nguyên tắc:
- Framework sử dụng NumPy/Adapter-first approach
- Một số thư viện (DoWhy, CausalML) yêu cầu Pandas
- DataConverter cung cấp seamless conversion
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union


class CausalInferenceError(Exception):
    """
    Exception raised when causal inference cannot be performed
    
    This is raised instead of falling back to correlation-based methods,
    which would violate the principle "Correlation is not Causation"
    """
    pass


class DataConverter:
    """
    Utility for converting between NumPy arrays and Pandas DataFrames
    
    Maintains feature names and ensures consistency across the framework
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Khởi tạo Data Converter
        
        Args:
            feature_names: List of feature names (optional)
        """
        self.feature_names = feature_names
    
    def numpy_to_dataframe(self, 
                          X: np.ndarray,
                          y: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None,
                          target_name: str = 'target') -> pd.DataFrame:
        """
        Convert NumPy arrays to Pandas DataFrame
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (optional)
            feature_names: List of feature names (optional)
            target_name: Name for target column
        
        Returns:
            Pandas DataFrame
        """
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Generate feature names if not provided
        if feature_names is None:
            if self.feature_names is not None and len(self.feature_names) == n_features:
                feature_names = self.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        
        # Add target if provided
        if y is not None:
            df[target_name] = y
        
        return df
    
    def dataframe_to_numpy(self,
                          df: pd.DataFrame,
                          target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert Pandas DataFrame to NumPy arrays
        
        Args:
            df: Pandas DataFrame
            target_col: Name of target column (optional)
        
        Returns:
            Tuple of (X, y) where y is None if target_col not specified
        """
        if target_col is not None and target_col in df.columns:
            y = df[target_col].values
            X = df.drop(columns=[target_col]).values
            
            # Store feature names
            self.feature_names = list(df.drop(columns=[target_col]).columns)
        else:
            X = df.values
            y = None
            
            # Store feature names
            self.feature_names = list(df.columns)
        
        return X, y
    
    def ensure_dataframe(self,
                        data: Union[np.ndarray, pd.DataFrame],
                        feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Ensure data is a DataFrame (convert if necessary)
        
        Args:
            data: Input data
            feature_names: Feature names (optional)
        
        Returns:
            Pandas DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return self.numpy_to_dataframe(data, feature_names=feature_names)
        else:
            raise TypeError(f"Expected np.ndarray or pd.DataFrame, got {type(data)}")
    
    def ensure_numpy(self,
                    data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Ensure data is a NumPy array (convert if necessary)
        
        Args:
            data: Input data
        
        Returns:
            NumPy array
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            X, _ = self.dataframe_to_numpy(data)
            return X
        else:
            raise TypeError(f"Expected np.ndarray or pd.DataFrame, got {type(data)}")
    
    def add_column_names(self,
                        X: np.ndarray,
                        y: Optional[np.ndarray] = None,
                        feature_names: Optional[List[str]] = None,
                        target_name: str = 'target',
                        sensitive_features: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        Add column names to NumPy arrays for causal inference
        
        Args:
            X: Feature array
            y: Target array (optional)
            feature_names: Feature names (optional)
            target_name: Target column name
            sensitive_features: Dict of sensitive feature names to arrays
        
        Returns:
            DataFrame with all columns named
        """
        df = self.numpy_to_dataframe(X, y, feature_names, target_name)
        
        # Add sensitive features
        if sensitive_features is not None:
            for name, values in sensitive_features.items():
                df[name] = values
        
        return df
    
    def extract_columns(self,
                       df: pd.DataFrame,
                       columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract specified columns and return remaining DataFrame
        
        Args:
            df: Input DataFrame
            columns: Columns to extract
        
        Returns:
            Tuple of (extracted_df, remaining_df)
        """
        extracted = df[columns]
        remaining = df.drop(columns=columns)
        
        return extracted, remaining
    
    def split_features_target(self,
                             df: pd.DataFrame,
                             target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target
        
        Args:
            df: Input DataFrame
            target_col: Target column name
        
        Returns:
            Tuple of (X_df, y_series)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y


class CausalDataValidator:
    """
    Validate data for causal inference
    
    Ensures data meets requirements for causal methods
    """
    
    @staticmethod
    def validate_treatment_outcome(data: pd.DataFrame,
                                   treatment: str,
                                   outcome: str) -> None:
        """
        Validate treatment and outcome columns exist
        
        Args:
            data: DataFrame
            treatment: Treatment column name
            outcome: Outcome column name
        
        Raises:
            CausalInferenceError: If validation fails
        """
        missing = []
        
        if treatment not in data.columns:
            missing.append(treatment)
        if outcome not in data.columns:
            missing.append(outcome)
        
        if missing:
            raise CausalInferenceError(
                f"Missing required columns for causal inference: {missing}\n"
                f"Available columns: {list(data.columns)}"
            )
    
    @staticmethod
    def validate_confounders(data: pd.DataFrame,
                            confounders: List[str]) -> None:
        """
        Validate confounder columns exist
        
        Args:
            data: DataFrame
            confounders: List of confounder column names
        
        Raises:
            CausalInferenceError: If validation fails
        """
        missing = [c for c in confounders if c not in data.columns]
        
        if missing:
            raise CausalInferenceError(
                f"Missing confounder columns: {missing}\n"
                f"Available columns: {list(data.columns)}"
            )
    
    @staticmethod
    def validate_binary_treatment(data: pd.DataFrame,
                                  treatment: str) -> None:
        """
        Validate treatment is binary
        
        Args:
            data: DataFrame
            treatment: Treatment column name
        
        Raises:
            CausalInferenceError: If treatment is not binary
        """
        unique_values = data[treatment].nunique()
        
        if unique_values != 2:
            raise CausalInferenceError(
                f"Treatment '{treatment}' must be binary (0/1), "
                f"but has {unique_values} unique values"
            )
    
    @staticmethod
    def check_sufficient_samples(data: pd.DataFrame,
                                min_samples: int = 100) -> None:
        """
        Check if there are sufficient samples for causal inference
        
        Args:
            data: DataFrame
            min_samples: Minimum required samples
        
        Raises:
            CausalInferenceError: If insufficient samples
        """
        n_samples = len(data)
        
        if n_samples < min_samples:
            raise CausalInferenceError(
                f"Insufficient samples for causal inference: {n_samples} < {min_samples}\n"
                f"Causal inference requires sufficient data to identify effects."
            )
    
    @staticmethod
    def validate_all(data: pd.DataFrame,
                    treatment: str,
                    outcome: str,
                    confounders: Optional[List[str]] = None,
                    require_binary_treatment: bool = True,
                    min_samples: int = 100) -> None:
        """
        Run all validations
        
        Args:
            data: DataFrame
            treatment: Treatment column
            outcome: Outcome column
            confounders: Confounder columns
            require_binary_treatment: Check if treatment is binary
            min_samples: Minimum sample size
        
        Raises:
            CausalInferenceError: If any validation fails
        """
        CausalDataValidator.validate_treatment_outcome(data, treatment, outcome)
        
        if confounders:
            CausalDataValidator.validate_confounders(data, confounders)
        
        if require_binary_treatment:
            CausalDataValidator.validate_binary_treatment(data, treatment)
        
        CausalDataValidator.check_sufficient_samples(data, min_samples)


def convert_for_causal_inference(X: np.ndarray,
                                 y: np.ndarray,
                                 treatment_col: str,
                                 outcome_col: str,
                                 feature_names: Optional[List[str]] = None,
                                 confounder_indices: Optional[List[int]] = None) -> pd.DataFrame:
    """
    High-level function to convert NumPy arrays for causal inference
    
    Args:
        X: Feature array
        y: Outcome array
        treatment_col: Name for treatment column
        outcome_col: Name for outcome column
        feature_names: Feature names
        confounder_indices: Indices of confounders in X
    
    Returns:
        DataFrame ready for causal inference
    
    Raises:
        CausalInferenceError: If data validation fails
    """
    converter = DataConverter(feature_names)
    
    # Separate treatment from confounders
    if confounder_indices is None:
        # Assume first column is treatment, rest are confounders
        treatment = X[:, 0]
        confounders = X[:, 1:] if X.shape[1] > 1 else None
        
        treatment_name = treatment_col
        confounder_names = [f'confounder_{i}' for i in range(X.shape[1] - 1)]
    else:
        # Use specified indices
        all_indices = set(range(X.shape[1]))
        confounder_indices_set = set(confounder_indices)
        treatment_indices = list(all_indices - confounder_indices_set)
        
        if len(treatment_indices) != 1:
            raise CausalInferenceError(
                f"Expected exactly 1 treatment column, got {len(treatment_indices)}"
            )
        
        treatment = X[:, treatment_indices[0]]
        confounders = X[:, confounder_indices]
        
        treatment_name = treatment_col
        confounder_names = [f'confounder_{i}' for i in confounder_indices]
    
    # Create DataFrame
    df = pd.DataFrame({treatment_name: treatment, outcome_col: y})
    
    # Add confounders
    if confounders is not None:
        for i, name in enumerate(confounder_names):
            df[name] = confounders[:, i]
    
    # Validate
    validator = CausalDataValidator()
    validator.validate_all(
        df,
        treatment=treatment_name,
        outcome=outcome_col,
        confounders=confounder_names if confounders is not None else None
    )
    
    return df

