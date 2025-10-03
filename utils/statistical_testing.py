"""
Statistical Testing Utilities

Công cụ thử nghiệm thống kê cho các tuyên bố nghiên cứu:
- Significance testing (t-test, Mann-Whitney U)
- Multiple comparisons correction (Bonferroni, Holm)
- Effect size calculation (Cohen's d, Cliff's delta)
- Bootstrap confidence intervals

Essential for publication: Không chỉ so sánh numbers, mà phải chứng minh
statistical significance và effect size
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from scipy import stats
import warnings


class SignificanceTest:
    """
    Statistical significance testing
    
    Compare model performances với statistical rigor
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Khởi tạo Significance Test
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
    
    def paired_t_test(self, scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, Any]:
        """
        Paired t-test
        
        Use when: Same test instances for both models (paired samples)
        
        H0: mean(scores_a) = mean(scores_b)
        H1: mean(scores_a) ≠ mean(scores_b)
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
        
        Returns:
            Test results
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Scores must have same length for paired test")
        
        # Compute differences
        differences = scores_a - scores_b
        
        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Effect size (Cohen's d for paired samples)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Confidence interval
        ci = stats.t.interval(
            1 - self.alpha,
            df=len(differences) - 1,
            loc=np.mean(differences),
            scale=stats.sem(differences)
        )
        
        return {
            'test': 'paired_t_test',
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_diff': np.mean(differences),
            'effect_size_cohens_d': effect_size,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'alpha': self.alpha
        }
    
    def independent_t_test(self, scores_a: np.ndarray, scores_b: np.ndarray,
                          equal_var: bool = False) -> Dict[str, Any]:
        """
        Independent t-test (Welch's t-test if equal_var=False)
        
        Use when: Different test instances for each model
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            equal_var: Assume equal variances (default: False for Welch's test)
        
        Returns:
            Test results
        """
        # Independent t-test
        t_statistic, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=equal_var)
        
        # Effect size (Cohen's d for independent samples)
        pooled_std = np.sqrt(
            ((len(scores_a) - 1) * np.var(scores_a, ddof=1) +
             (len(scores_b) - 1) * np.var(scores_b, ddof=1)) /
            (len(scores_a) + len(scores_b) - 2)
        )
        effect_size = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
        
        return {
            'test': 'independent_t_test' if equal_var else 'welch_t_test',
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'mean_diff': np.mean(scores_a) - np.mean(scores_b),
            'effect_size_cohens_d': effect_size,
            'alpha': self.alpha
        }
    
    def mann_whitney_u_test(self, scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, Any]:
        """
        Mann-Whitney U test (non-parametric alternative to t-test)
        
        Use when: Data not normally distributed
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
        
        Returns:
            Test results
        """
        # Mann-Whitney U test
        u_statistic, p_value = stats.mannwhitneyu(
            scores_a, scores_b, alternative='two-sided'
        )
        
        # Effect size (Cliff's delta)
        cliff_delta = self._compute_cliff_delta(scores_a, scores_b)
        
        return {
            'test': 'mann_whitney_u',
            'u_statistic': u_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'median_a': np.median(scores_a),
            'median_b': np.median(scores_b),
            'effect_size_cliff_delta': cliff_delta,
            'alpha': self.alpha
        }
    
    def _compute_cliff_delta(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Cliff's Delta effect size
        
        Cliff's Delta ∈ [-1, 1]:
        - δ = 1: All values in A > all values in B
        - δ = 0: Distributions overlap completely
        - δ = -1: All values in A < all values in B
        """
        n_a, n_b = len(a), len(b)
        
        # Count pairs where a[i] > b[j]
        greater = sum((a[i] > b[j]) for i in range(n_a) for j in range(n_b))
        
        # Count pairs where a[i] < b[j]
        less = sum((a[i] < b[j]) for i in range(n_a) for j in range(n_b))
        
        # Cliff's delta
        delta = (greater - less) / (n_a * n_b)
        
        return delta
    
    def wilcoxon_signed_rank_test(self, scores_a: np.ndarray,
                                  scores_b: np.ndarray) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test (non-parametric paired test)
        
        Use when: Paired samples, data not normally distributed
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
        
        Returns:
            Test results
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Scores must have same length for paired test")
        
        # Wilcoxon signed-rank test
        w_statistic, p_value = stats.wilcoxon(scores_a, scores_b)
        
        return {
            'test': 'wilcoxon_signed_rank',
            'w_statistic': w_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'median_diff': np.median(scores_a - scores_b),
            'alpha': self.alpha
        }


class MultipleComparisonCorrection:
    """
    Multiple comparison correction
    
    When comparing multiple models/methods, must correct for
    multiple testing to avoid false discoveries
    """
    
    def __init__(self, method: str = 'bonferroni'):
        """
        Khởi tạo Multiple Comparison Correction
        
        Args:
            method: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)
        """
        self.method = method
    
    def correct(self, p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Apply multiple comparison correction
        
        Args:
            p_values: List of p-values
            alpha: Family-wise error rate
        
        Returns:
            Corrected results
        """
        p_values = np.array(p_values)
        
        if self.method == 'bonferroni':
            return self._bonferroni(p_values, alpha)
        elif self.method == 'holm':
            return self._holm(p_values, alpha)
        elif self.method == 'fdr_bh':
            return self._fdr_bh(p_values, alpha)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _bonferroni(self, p_values: np.ndarray, alpha: float) -> Dict[str, Any]:
        """
        Bonferroni correction (most conservative)
        
        Adjusted alpha = α / m where m = number of tests
        """
        m = len(p_values)
        adjusted_alpha = alpha / m
        
        rejected = p_values < adjusted_alpha
        
        return {
            'method': 'bonferroni',
            'original_alpha': alpha,
            'adjusted_alpha': adjusted_alpha,
            'num_tests': m,
            'rejected': rejected,
            'num_rejected': rejected.sum(),
            'corrected_p_values': p_values * m  # Adjusted p-values
        }
    
    def _holm(self, p_values: np.ndarray, alpha: float) -> Dict[str, Any]:
        """
        Holm-Bonferroni correction (less conservative than Bonferroni)
        
        Step-down procedure
        """
        m = len(p_values)
        
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Holm procedure
        rejected = np.zeros(m, dtype=bool)
        
        for i in range(m):
            adjusted_alpha = alpha / (m - i)
            if sorted_p[i] < adjusted_alpha:
                rejected[sorted_indices[i]] = True
            else:
                break  # Stop after first non-rejection
        
        return {
            'method': 'holm',
            'original_alpha': alpha,
            'num_tests': m,
            'rejected': rejected,
            'num_rejected': rejected.sum()
        }
    
    def _fdr_bh(self, p_values: np.ndarray, alpha: float) -> Dict[str, Any]:
        """
        Benjamini-Hochberg FDR control
        
        Controls False Discovery Rate instead of Family-Wise Error Rate
        (less conservative, more power)
        """
        m = len(p_values)
        
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # BH procedure
        rejected = np.zeros(m, dtype=bool)
        
        for i in range(m - 1, -1, -1):
            threshold = (i + 1) / m * alpha
            if sorted_p[i] <= threshold:
                # Reject this and all previous
                for j in range(i + 1):
                    rejected[sorted_indices[j]] = True
                break
        
        return {
            'method': 'benjamini_hochberg_fdr',
            'original_alpha': alpha,
            'num_tests': m,
            'rejected': rejected,
            'num_rejected': rejected.sum()
        }


class BootstrapCI:
    """
    Bootstrap confidence intervals
    
    Non-parametric confidence intervals via resampling
    """
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95):
        """
        Khởi tạo Bootstrap CI
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 0.95)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
    
    def compute_ci(self, data: np.ndarray, statistic: str = 'mean') -> Dict[str, Any]:
        """
        Compute bootstrap confidence interval
        
        Args:
            data: Data array
            statistic: 'mean', 'median', 'std'
        
        Returns:
            CI results
        """
        # Bootstrap samples
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=len(data), replace=True)
            
            # Compute statistic
            if statistic == 'mean':
                stat_value = np.mean(sample)
            elif statistic == 'median':
                stat_value = np.median(sample)
            elif statistic == 'std':
                stat_value = np.std(sample, ddof=1)
            else:
                raise ValueError(f"Unknown statistic: {statistic}")
            
            bootstrap_stats.append(stat_value)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Confidence interval (percentile method)
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        # Point estimate
        if statistic == 'mean':
            point_estimate = np.mean(data)
        elif statistic == 'median':
            point_estimate = np.median(data)
        elif statistic == 'std':
            point_estimate = np.std(data, ddof=1)
        
        return {
            'statistic': statistic,
            'point_estimate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': self.confidence_level,
            'n_bootstrap': self.n_bootstrap
        }


class NormalityTest:
    """
    Test for normality assumptions
    
    Critical for choosing appropriate statistical tests
    """
    
    @staticmethod
    def shapiro_wilk(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Shapiro-Wilk test for normality
        
        H0: Data is normally distributed
        H1: Data is not normally distributed
        
        Args:
            data: Data array
            alpha: Significance level
        
        Returns:
            Test results
        """
        if len(data) < 3:
            return {
                'test': 'shapiro_wilk',
                'statistic': None,
                'p_value': None,
                'normal': None,
                'warning': 'Insufficient samples for Shapiro-Wilk test (n < 3)'
            }
        
        statistic, p_value = stats.shapiro(data)
        
        return {
            'test': 'shapiro_wilk',
            'statistic': statistic,
            'p_value': p_value,
            'normal': p_value >= alpha,
            'alpha': alpha
        }
    
    @staticmethod
    def anderson_darling(data: np.ndarray) -> Dict[str, Any]:
        """
        Anderson-Darling test for normality
        
        More powerful than Shapiro-Wilk for larger samples
        
        Args:
            data: Data array
        
        Returns:
            Test results
        """
        result = stats.anderson(data, dist='norm')
        
        # Check at 5% significance level (index 2)
        critical_value = result.critical_values[2]
        is_normal = result.statistic < critical_value
        
        return {
            'test': 'anderson_darling',
            'statistic': result.statistic,
            'critical_value_5pct': critical_value,
            'normal': is_normal,
            'all_critical_values': result.critical_values,
            'significance_levels': result.significance_level
        }
    
    @staticmethod
    def check_assumptions(scores_a: np.ndarray, scores_b: np.ndarray,
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Check assumptions for parametric tests
        
        Checks:
        1. Normality (Shapiro-Wilk)
        2. Equal variances (Levene's test)
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            alpha: Significance level
        
        Returns:
            Assumption check results with recommendations
        """
        # Normality tests
        normal_a = NormalityTest.shapiro_wilk(scores_a, alpha)
        normal_b = NormalityTest.shapiro_wilk(scores_b, alpha)
        
        both_normal = (normal_a.get('normal', False) and 
                      normal_b.get('normal', False))
        
        # Equal variance test (Levene's test)
        levene_stat, levene_p = stats.levene(scores_a, scores_b)
        equal_variances = levene_p >= alpha
        
        # Recommendations
        if both_normal and equal_variances:
            recommendation = "✓ Use parametric test (t-test with equal_var=True)"
        elif both_normal and not equal_variances:
            recommendation = "✓ Use parametric test (Welch's t-test with equal_var=False)"
        else:
            recommendation = "⚠ Use non-parametric test (Mann-Whitney U or Wilcoxon)"
        
        return {
            'normality_a': normal_a,
            'normality_b': normal_b,
            'both_normal': both_normal,
            'equal_variances': {
                'test': 'levene',
                'statistic': levene_stat,
                'p_value': levene_p,
                'equal': equal_variances,
                'alpha': alpha
            },
            'recommendation': recommendation,
            'use_parametric': both_normal,
            'use_equal_var': equal_variances if both_normal else None
        }


class ModelComparison:
    """
    Comprehensive model comparison with statistical testing
    """
    
    def __init__(self, alpha: float = 0.05, check_assumptions: bool = True):
        """
        Khởi tạo Model Comparison
        
        Args:
            alpha: Significance level
            check_assumptions: Automatically check test assumptions
        """
        self.alpha = alpha
        self.sig_test = SignificanceTest(alpha)
        self.check_assumptions = check_assumptions
    
    def compare_two_models(self, scores_a: np.ndarray, scores_b: np.ndarray,
                          model_a_name: str = "Model A",
                          model_b_name: str = "Model B",
                          paired: bool = True,
                          parametric: Optional[bool] = None) -> Dict[str, Any]:
        """
        Compare two models comprehensively
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            paired: Whether data is paired
            parametric: Use parametric test (t-test) or non-parametric (Mann-Whitney/Wilcoxon)
                       If None, automatically decide based on normality tests
        
        Returns:
            Comprehensive comparison
        """
        # Check assumptions if requested
        assumption_results = None
        if self.check_assumptions and parametric is None:
            assumption_results = NormalityTest.check_assumptions(scores_a, scores_b, self.alpha)
            parametric = assumption_results['use_parametric']
            
            print(f"📊 Assumption Check:")
            print(f"  Normality A: {'✓' if assumption_results['normality_a'].get('normal') else '✗'} "
                  f"(p={assumption_results['normality_a'].get('p_value', 'N/A')})")
            print(f"  Normality B: {'✓' if assumption_results['normality_b'].get('normal') else '✗'} "
                  f"(p={assumption_results['normality_b'].get('p_value', 'N/A')})")
            print(f"  {assumption_results['recommendation']}")
        
        # Descriptive statistics
        desc_a = {
            'mean': np.mean(scores_a),
            'median': np.median(scores_a),
            'std': np.std(scores_a, ddof=1),
            'min': np.min(scores_a),
            'max': np.max(scores_a)
        }
        
        desc_b = {
            'mean': np.mean(scores_b),
            'median': np.median(scores_b),
            'std': np.std(scores_b, ddof=1),
            'min': np.min(scores_b),
            'max': np.max(scores_b)
        }
        
        # Statistical test
        if paired:
            if parametric:
                test_results = self.sig_test.paired_t_test(scores_a, scores_b)
            else:
                test_results = self.sig_test.wilcoxon_signed_rank_test(scores_a, scores_b)
        else:
            if parametric:
                test_results = self.sig_test.independent_t_test(scores_a, scores_b)
            else:
                test_results = self.sig_test.mann_whitney_u_test(scores_a, scores_b)
        
        # Bootstrap CI for difference
        if paired:
            differences = scores_a - scores_b
            bootstrap = BootstrapCI()
            diff_ci = bootstrap.compute_ci(differences, statistic='mean')
        else:
            diff_ci = None
        
        # Winner
        if test_results['significant']:
            if paired:
                winner = model_a_name if test_results['mean_diff'] > 0 else model_b_name
            else:
                winner = model_a_name if test_results['mean_a'] > test_results['mean_b'] else model_b_name
        else:
            winner = "No significant difference"
        
        return {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'descriptives_a': desc_a,
            'descriptives_b': desc_b,
            'assumption_check': assumption_results,
            'statistical_test': test_results,
            'difference_ci': diff_ci,
            'winner': winner,
            'conclusion': self._generate_conclusion(test_results, model_a_name, model_b_name)
        }
    
    def _generate_conclusion(self, test_results: Dict, model_a: str, model_b: str) -> str:
        """Generate human-readable conclusion"""
        if test_results['significant']:
            if 'mean_diff' in test_results:
                if test_results['mean_diff'] > 0:
                    return f"{model_a} significantly outperforms {model_b} (p={test_results['p_value']:.4f})"
                else:
                    return f"{model_b} significantly outperforms {model_a} (p={test_results['p_value']:.4f})"
            else:
                return f"Significant difference detected (p={test_results['p_value']:.4f})"
        else:
            return f"No significant difference (p={test_results['p_value']:.4f})"
    
    def compare_multiple_models(self, scores_dict: Dict[str, np.ndarray],
                               correction_method: str = 'holm') -> Dict[str, Any]:
        """
        Compare multiple models with multiple comparison correction
        
        Args:
            scores_dict: Dict mapping model names to score arrays
            correction_method: Multiple comparison correction method
        
        Returns:
            Comparison results
        """
        model_names = list(scores_dict.keys())
        n_models = len(model_names)
        
        # Pairwise comparisons
        pairwise_results = []
        p_values = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_a = model_names[i]
                model_b = model_names[j]
                
                # Paired t-test
                test = self.sig_test.paired_t_test(
                    scores_dict[model_a],
                    scores_dict[model_b]
                )
                
                pairwise_results.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'test': test
                })
                
                p_values.append(test['p_value'])
        
        # Multiple comparison correction
        corrector = MultipleComparisonCorrection(method=correction_method)
        correction_results = corrector.correct(p_values, self.alpha)
        
        # Update significance based on correction
        for i, result in enumerate(pairwise_results):
            result['test']['significant_corrected'] = correction_results['rejected'][i]
        
        return {
            'models': model_names,
            'pairwise_comparisons': pairwise_results,
            'multiple_comparison_correction': correction_results
        }

