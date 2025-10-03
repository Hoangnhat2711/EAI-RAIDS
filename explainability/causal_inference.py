"""
Causal Inference Integration

Tích hợp các thư viện chuyên biệt về Causal Inference:
- DoWhy (Microsoft Research)
- CausalML (Uber)
- EconML (Microsoft Research)

Move beyond correlation → causation

References:
- Sharma & Kiciman "DoWhy: An End-to-End Library for Causal Inference" (2020)
- Pearl "Causality: Models, Reasoning, and Inference" (2009)
- Imbens & Rubin "Causal Inference for Statistics, Social, and Biomedical Sciences" (2015)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import warnings


class DoWhyIntegration:
    """
    Integration với DoWhy library
    
    DoWhy provides:
    - Causal graph modeling
    - Multiple identification methods
    - Refutation tests
    """
    
    def __init__(self, treatment: str, outcome: str,
                 common_causes: Optional[List[str]] = None,
                 instruments: Optional[List[str]] = None):
        """
        Khởi tạo DoWhy Integration
        
        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name
            common_causes: List of confounders
            instruments: List of instrumental variables
        """
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes or []
        self.instruments = instruments or []
        
        self.model = None
        self.identified_estimand = None
        self.estimate = None
    
    def create_causal_model(self, data: pd.DataFrame,
                           graph: Optional[str] = None) -> Any:
        """
        Create causal model
        
        Args:
            data: Dataset
            graph: DOT format causal graph (optional)
        
        Returns:
            CausalModel instance
        """
        try:
            from dowhy import CausalModel
        except ImportError:
            raise ImportError(
                "DoWhy not installed. Install: pip install dowhy"
            )
        
        if graph is None:
            # Auto-generate graph
            graph = self._generate_default_graph()
        
        self.model = CausalModel(
            data=data,
            treatment=self.treatment,
            outcome=self.outcome,
            common_causes=self.common_causes,
            instruments=self.instruments,
            graph=graph
        )
        
        return self.model
    
    def _generate_default_graph(self) -> str:
        """Generate default causal graph in DOT format"""
        edges = []
        
        # Common causes → Treatment and Outcome
        for cause in self.common_causes:
            edges.append(f"{cause} -> {self.treatment}")
            edges.append(f"{cause} -> {self.outcome}")
        
        # Treatment → Outcome
        edges.append(f"{self.treatment} -> {self.outcome}")
        
        # Instruments → Treatment
        for inst in self.instruments:
            edges.append(f"{inst} -> {self.treatment}")
        
        graph = "digraph {" + "; ".join(edges) + ";}"
        return graph
    
    def identify_effect(self) -> Any:
        """
        Identify causal effect
        
        Returns:
            Identified estimand
        """
        if self.model is None:
            raise ValueError("Create causal model first")
        
        self.identified_estimand = self.model.identify_effect(
            proceed_when_unidentifiable=True
        )
        
        return self.identified_estimand
    
    def estimate_effect(self, method: str = 'backdoor.linear_regression',
                       **method_params) -> Dict[str, Any]:
        """
        Estimate causal effect
        
        Args:
            method: Estimation method
                - 'backdoor.linear_regression'
                - 'backdoor.propensity_score_matching'
                - 'iv.instrumental_variable'
                - 'frontdoor.two_stage_regression'
            **method_params: Method-specific parameters
        
        Returns:
            Causal estimate
        """
        if self.identified_estimand is None:
            raise ValueError("Identify effect first")
        
        self.estimate = self.model.estimate_effect(
            self.identified_estimand,
            method_name=method,
            **method_params
        )
        
        results = {
            'effect': self.estimate.value,
            'method': method,
            'estimand': str(self.identified_estimand),
            'params': method_params
        }
        
        return results
    
    def refute_estimate(self, refutation_type: str = 'random_common_cause') -> Dict:
        """
        Refute causal estimate (sensitivity analysis)
        
        Args:
            refutation_type:
                - 'random_common_cause': Add random confounder
                - 'placebo_treatment_refuter': Replace treatment with random
                - 'data_subset_refuter': Test on data subsets
                - 'add_unobserved_common_cause': Simulate unobserved confounder
        
        Returns:
            Refutation results
        """
        if self.estimate is None:
            raise ValueError("Estimate effect first")
        
        refutation = self.model.refute_estimate(
            self.identified_estimand,
            self.estimate,
            method_name=refutation_type
        )
        
        return {
            'refutation_type': refutation_type,
            'refutation_result': refutation.refutation_result,
            'new_effect': refutation.estimated_effect if hasattr(refutation, 'estimated_effect') else None
        }
    
    def complete_analysis(self, data: pd.DataFrame,
                         graph: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete causal analysis workflow
        
        Args:
            data: Dataset
            graph: Causal graph (optional)
        
        Returns:
            Complete results
        """
        # 1. Create model
        self.create_causal_model(data, graph)
        
        # 2. Identify
        self.identify_effect()
        
        # 3. Estimate
        estimate_results = self.estimate_effect()
        
        # 4. Refute
        refutations = []
        for refutation_type in ['random_common_cause', 'placebo_treatment_refuter']:
            try:
                refutation = self.refute_estimate(refutation_type)
                refutations.append(refutation)
            except Exception as e:
                warnings.warn(f"Refutation {refutation_type} failed: {e}")
        
        return {
            'causal_effect': estimate_results['effect'],
            'estimate': estimate_results,
            'refutations': refutations,
            'robust': all(r.get('refutation_result', False) for r in refutations)
        }


class CausalMLIntegration:
    """
    Integration với CausalML library (Uber)
    
    Focus on:
    - Uplift modeling
    - Heterogeneous treatment effects
    - Meta-learners (S-Learner, T-Learner, X-Learner)
    """
    
    def __init__(self, treatment_col: str, outcome_col: str):
        """
        Khởi tạo CausalML Integration
        
        Args:
            treatment_col: Treatment column name
            outcome_col: Outcome column name
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        self.model = None
    
    def estimate_cate(self, X: np.ndarray, treatment: np.ndarray,
                     y: np.ndarray, method: str = 'T-Learner',
                     base_estimator: Optional[Any] = None) -> np.ndarray:
        """
        Estimate Conditional Average Treatment Effect (CATE)
        
        CATE(x) = E[Y(1) - Y(0) | X = x]
        
        Args:
            X: Features
            treatment: Treatment assignment (0 or 1)
            y: Outcomes
            method: 'S-Learner', 'T-Learner', 'X-Learner'
            base_estimator: Base ML model
        
        Returns:
            CATE estimates for each instance
        """
        try:
            from causalml.inference.meta import BaseSLearner, BaseTLearner, BaseXLearner
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError(
                "CausalML not installed. Install: pip install causalml"
            )
        
        if base_estimator is None:
            base_estimator = RandomForestRegressor(n_estimators=100)
        
        # Select meta-learner
        if method == 'S-Learner':
            self.model = BaseSLearner(learner=base_estimator)
        elif method == 'T-Learner':
            self.model = BaseTLearner(learner=base_estimator)
        elif method == 'X-Learner':
            self.model = BaseXLearner(learner=base_estimator)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit
        self.model.fit(X=X, treatment=treatment, y=y)
        
        # Predict CATE
        cate = self.model.predict(X=X)
        
        return cate
    
    def estimate_ate(self, X: np.ndarray, treatment: np.ndarray,
                    y: np.ndarray, **kwargs) -> float:
        """
        Estimate Average Treatment Effect (ATE)
        
        ATE = E[Y(1) - Y(0)]
        
        Args:
            X: Features
            treatment: Treatment
            y: Outcomes
        
        Returns:
            ATE estimate
        """
        cate = self.estimate_cate(X, treatment, y, **kwargs)
        ate = np.mean(cate)
        
        return ate


class CausalFeatureSelector:
    """
    Feature selection based on causal relationships
    
    Select features that have causal relationship with outcome,
    not just correlation
    """
    
    def __init__(self, outcome: str, method: str = 'pc_algorithm'):
        """
        Khởi tạo Causal Feature Selector
        
        Args:
            outcome: Outcome variable name
            method: 'pc_algorithm', 'ges', 'lingam'
        """
        self.outcome = outcome
        self.method = method
        
        self.causal_graph = None
        self.selected_features = []
    
    def fit(self, data: pd.DataFrame) -> List[str]:
        """
        Discover causal graph and select features
        
        Args:
            data: Dataset
        
        Returns:
            List of causally relevant features
        """
        if self.method == 'pc_algorithm':
            self._fit_pc_algorithm(data)
        elif self.method == 'lingam':
            self._fit_lingam(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.selected_features
    
    def _fit_pc_algorithm(self, data: pd.DataFrame):
        """
        PC Algorithm for causal discovery
        
        (Simplified implementation - use causal-learn library for production)
        """
        try:
            from causallearn.search.ConstraintBased.PC import pc
        except ImportError:
            warnings.warn("causal-learn not installed, using correlation-based fallback")
            self._fallback_correlation(data)
            return
        
        # Run PC algorithm
        data_array = data.values
        cg = pc(data_array)
        
        # Extract features that have edge to outcome
        outcome_idx = list(data.columns).index(self.outcome)
        
        graph_matrix = cg.G.graph
        self.selected_features = []
        
        for i in range(len(data.columns)):
            if i != outcome_idx:
                # Check if there's an edge from feature i to outcome
                if graph_matrix[i, outcome_idx] != 0 or graph_matrix[outcome_idx, i] != 0:
                    self.selected_features.append(data.columns[i])
    
    def _fit_lingam(self, data: pd.DataFrame):
        """
        LiNGAM (Linear Non-Gaussian Acyclic Model) for causal discovery
        """
        try:
            from lingam import DirectLiNGAM
        except ImportError:
            warnings.warn("lingam not installed, using correlation-based fallback")
            self._fallback_correlation(data)
            return
        
        model = DirectLiNGAM()
        model.fit(data.values)
        
        # Get causal order
        causal_order = model.causal_order_
        outcome_idx = list(data.columns).index(self.outcome)
        
        # Features that come before outcome in causal order
        outcome_position = np.where(causal_order == outcome_idx)[0][0]
        
        self.selected_features = [
            data.columns[causal_order[i]]
            for i in range(outcome_position)
        ]
    
    def _fallback_correlation(self, data: pd.DataFrame):
        """Fallback: Use correlation (not causal!)"""
        warnings.warn("Using correlation-based selection (NOT causal)")
        
        corr = data.corr()[self.outcome].abs()
        corr = corr.drop(self.outcome)
        
        # Select top correlated features
        threshold = 0.3
        self.selected_features = list(corr[corr > threshold].index)


def estimate_causal_effect(data: pd.DataFrame,
                           treatment: str,
                           outcome: str,
                           confounders: Optional[List[str]] = None,
                           method: str = 'dowhy') -> Dict[str, Any]:
    """
    High-level function to estimate causal effect
    
    Args:
        data: Dataset
        treatment: Treatment variable
        outcome: Outcome variable
        confounders: List of confounders
        method: 'dowhy' or 'causalml'
    
    Returns:
        Causal effect estimate
    """
    if method == 'dowhy':
        dowhy = DoWhyIntegration(
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders
        )
        
        results = dowhy.complete_analysis(data)
        return results
    
    elif method == 'causalml':
        causalml = CausalMLIntegration(
            treatment_col=treatment,
            outcome_col=outcome
        )
        
        # Prepare data
        X = data.drop([treatment, outcome], axis=1).values
        treatment_vals = data[treatment].values
        y = data[outcome].values
        
        ate = causalml.estimate_ate(X, treatment_vals, y)
        
        return {
            'causal_effect': ate,
            'method': 'causalml',
            'ate': ate
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def discover_causal_graph(data: pd.DataFrame,
                         method: str = 'pc_algorithm') -> Dict[str, Any]:
    """
    Discover causal graph from data
    
    Args:
        data: Dataset
        method: Discovery algorithm
    
    Returns:
        Causal graph information
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.GraphUtils import GraphUtils
    except ImportError:
        raise ImportError("causal-learn required: pip install causal-learn")
    
    # Run PC algorithm
    cg = pc(data.values)
    
    # Visualize
    pdy = GraphUtils.to_pydot(cg.G)
    
    return {
        'graph': cg.G,
        'edges': cg.G.get_graph_edges(),
        'visualization': pdy
    }

