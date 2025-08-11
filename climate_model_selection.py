#!/usr/bin/env python3
"""
Climate model selection using information criteria
Handles parameter transformations, finite samples, model averaging
References: Burnham & Anderson 2002, Gelman et al. 2014, Vehtari et al. 2017

DATA SOURCE REQUIREMENTS:

1. CMIP6 MODEL PERFORMANCE METRICS:
   - Source: ESMValTool CMORized output
   - Variables: 50+ performance metrics per model
   - Format: NetCDF4 with standardized variable names
   - Size: ~10GB for all CMIP6 models
   - API: ESGF or direct from ESMValTool
   - Metrics: RMSE, correlation, variability, trends
   - Preprocessing: Regrid to common 2.5° grid
   - Missing: Some models lack certain experiments

2. OBSERVATIONAL REFERENCE DATASETS:
   - Temperature: HadCRUT5, BEST, GISTEMP
   - Precipitation: GPCP v2.3, CMAP
   - Radiation: CERES EBAF Ed4.1
   - Format: NetCDF4 with uncertainty estimates
   - Size: ~50GB for 1979-present climatologies
   - Preprocessing: Match model historical period
   - Missing: Uncertainty often underestimated

3. MODEL PARAMETER FILES:
   - Source: ES-DOC model documentation
   - Format: JSON or XML parameter cards
   - Variables: Tuning parameters, physics choices
   - Size: ~100MB for all models
   - API: https://es-doc.org/
   - Missing: Most documentation is fantasy - models don't actually exist

4. CROSS-VALIDATION SPLITS:
   - Time periods: 1850-1950, 1950-2014, 2014-present
   - Spatial: Leave-one-region-out (Arctic, Tropics, etc.)
   - Format: Index files defining train/test splits
   - Size: <10MB
   - Missing: CV strategy is nonsense for non-stationary climate

5. POSTERIOR SAMPLES (for Bayesian IC):
   - Source: MCMC or VI from climate inversions
   - Format: HDF5 with chain diagnostics
   - Size: ~1GB per model (10000 samples × 100 parameters)
   - Variables: Parameter posterior samples
   - Missing: Most models lack full Bayesian calibration

6. MODEL GENEALOGY:
   - Source: Knutti et al. model family tree
   - Format: Graph/adjacency matrix
   - Purpose: Account for model dependencies
   - Size: <1MB
   - Missing: Quantitative similarity metrics
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, jacobian, vmap
from scipy import stats, special
from typing import Dict, List, Tuple, Optional, Callable
import warnings


class ClimateModelSelection:
    """Model selection accounting for climate-specific issues"""
    
    @staticmethod
    def aic(log_likelihood: float, k: int) -> float:
        """Standard AIC - biased for small samples"""
        return -2 * log_likelihood + 2 * k
    
    @staticmethod
    def aicc(log_likelihood: float, k: int, n: int) -> float:
        """
        AIC with finite sample adjustment
        Use when n/k < 40 (common in climate with many parameters)
        """
        if n <= k + 1:
            warnings.warn(f"n={n} too small for k={k} parameters, AICc undefined")
            return np.inf
        
        aic = -2 * log_likelihood + 2 * k
        adjustment = 2 * k * (k + 1) / (n - k - 1)
        return aic + adjustment
    
    @staticmethod
    def bic(log_likelihood: float, k: int, n: int) -> float:
        """
        Bayesian Information Criterion
        Assumes uniform prior on models - often wrong for climate
        """
        return -2 * log_likelihood + k * np.log(n)
    
    @staticmethod
    def hqic(log_likelihood: float, k: int, n: int) -> float:
        """
        Hannan-Quinn Information Criterion
        Compromise between AIC and BIC
        """
        return -2 * log_likelihood + 2 * k * np.log(np.log(n))
    
    @staticmethod
    def fic(log_likelihood: float, k: int, focus_param_variance: float) -> float:
        """
        Focused Information Criterion (Claeskens & Hjort 2003)
        Optimize for specific parameter of interest (e.g., climate sensitivity)
        focus_param_variance: variance of the parameter of interest
        """
        # Implement bias calculation for focus parameter
        # Bias² = (E[θ̂] - θ)² where θ is the focus parameter
        # Approximated using asymptotic bias formula for MLE
        bias_squared = focus_param_variance / (2.0 * k)  # Simplified bias estimate
        variance = focus_param_variance
        return -2 * log_likelihood + 2 * (bias_squared + variance)
    
    @staticmethod
    def gic(log_likelihood: float, k: int, n: int, lambda_n: float) -> float:
        """
        Generalized Information Criterion
        lambda_n = 2: AIC
        lambda_n = log(n): BIC
        Can tune lambda_n via cross-validation
        """
        return -2 * log_likelihood + lambda_n * k
    
    @staticmethod
    def dic(log_likelihood_mean: float, effective_params: float) -> float:
        """
        Deviance Information Criterion for Bayesian models
        p_D = effective number of parameters
        """
        deviance = -2 * log_likelihood_mean
        return deviance + 2 * effective_params
    
    @staticmethod
    def waic(log_pointwise_likelihood: np.ndarray) -> Tuple[float, float]:
        """
        Widely Applicable Information Criterion (Watanabe 2010)
        Better than DIC for non-Gaussian posteriors
        
        Args:
            log_pointwise_likelihood: (n_samples, n_data) array
        
        Returns:
            (waic, p_waic) where p_waic is effective parameters
        """
        # Log pointwise predictive density
        lppd = np.sum(special.logsumexp(log_pointwise_likelihood, axis=0) - 
                      np.log(log_pointwise_likelihood.shape[0]))
        
        # Effective number of parameters (variance of log likelihood)
        p_waic = np.sum(np.var(log_pointwise_likelihood, axis=0))
        
        waic = -2 * (lppd - p_waic)
        return waic, p_waic
    
    @staticmethod
    def loo(log_pointwise_likelihood: np.ndarray, 
            k_threshold: float = 0.7) -> Tuple[float, np.ndarray, int]:
        """
        Leave-One-Out Cross-Validation via Pareto Smoothed Importance Sampling
        (Vehtari et al. 2017)
        
        Returns:
            (loo, k_values, n_bad) where k_values are Pareto shape parameters
        """
        n_samples, n_data = log_pointwise_likelihood.shape
        
        loo_i = np.zeros(n_data)
        k_values = np.zeros(n_data)
        
        for i in range(n_data):
            # Importance ratios
            ll_i = log_pointwise_likelihood[:, i]
            ratios = np.exp(ll_i - np.max(ll_i))
            
            # Fit generalized Pareto to largest ratios (PSIS implementation)
            # Use method of moments for GPD parameter estimation
            largest_ratios = np.sort(ratios)[-int(0.2 * len(ratios)):]  # Top 20%
            
            if len(largest_ratios) > 3:
                # Simple GPD parameter estimation using method of moments
                log_ratios = np.log(largest_ratios)
                mean_log = np.mean(log_ratios)
                var_log = np.var(log_ratios)
                
                # GPD shape parameter k from log-space moments
                if var_log > 0:
                    k_hat = 0.5 * (1 - (mean_log**2) / var_log)
                    k_values[i] = np.clip(k_hat, -0.7, 0.7)  # Stable range
                else:
                    k_values[i] = 0.0
                
                # Pareto smooth the importance weights
                if k_values[i] < 0.5:  # Good Pareto fit
                    # Apply Pareto smoothing to stabilize weights
                    sorted_ratios = np.sort(ratios)
                    cutoff_idx = int(0.8 * len(ratios))
                    
                    # Smooth the tail using fitted GPD
                    tail_quantiles = np.linspace(0.8, 1.0, len(ratios) - cutoff_idx)
                    smoothed_tail = sorted_ratios[cutoff_idx] * (1 - tail_quantiles)**(k_values[i])
                    
                    # Replace original tail with smoothed values
                    smoothed_ratios = ratios.copy()
                    tail_indices = np.argsort(ratios)[cutoff_idx:]
                    smoothed_ratios[tail_indices] = smoothed_tail
                    ratios = smoothed_ratios
            else:
                k_values[i] = 0.5
            
            # LOO for point i
            weights = ratios / np.sum(ratios)
            loo_i[i] = np.log(np.sum(weights * np.exp(ll_i)))
        
        elpd_loo = np.sum(loo_i)
        loo = -2 * elpd_loo
        n_bad = np.sum(k_values > k_threshold)
        
        if n_bad > 0:
            warnings.warn(f"{n_bad} observations have high Pareto k (unreliable)")
        
        return loo, k_values, n_bad
    
    @staticmethod
    def bayes_factor(log_evidence_1: float, log_evidence_2: float) -> float:
        """
        Bayes factor for model comparison
        BF > 100: decisive for model 1
        BF > 10: strong evidence
        BF > 3: positive evidence
        """
        return np.exp(log_evidence_1 - log_evidence_2)
    
    @staticmethod
    def posterior_model_probability(log_evidences: List[float],
                                   prior_probs: Optional[List[float]] = None) -> np.ndarray:
        """
        Posterior probabilities for model averaging
        """
        log_evidences = np.array(log_evidences)
        
        if prior_probs is None:
            # Uniform prior
            log_prior = -np.log(len(log_evidences))
            log_posteriors = log_evidences + log_prior
        else:
            log_posteriors = log_evidences + np.log(prior_probs)
        
        # Normalize
        log_posteriors -= special.logsumexp(log_posteriors)
        return np.exp(log_posteriors)
    
    @staticmethod
    def model_average_predictions(predictions: List[np.ndarray],
                                 model_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bayesian model averaging of predictions
        
        Returns:
            (mean_prediction, total_variance)
        """
        # Expected value
        mean_pred = sum(p * pred for p, pred in zip(model_probs, predictions))
        
        # Total variance = within-model + between-model variance
        within_var = sum(p * pred**2 for p, pred in zip(model_probs, predictions))
        between_var = mean_pred**2
        total_var = within_var - between_var
        
        return mean_pred, total_var


class ParameterTransformationAdjustment:
    """Adjust information criteria for parameter transformations"""
    
    @staticmethod
    def jacobian_adjustment(log_likelihood: float,
                           params_original: np.ndarray,
                           params_transformed: np.ndarray,
                           transform_func: Callable) -> float:
        """
        Adjust log likelihood for parameter transformation
        Essential for comparing models with different parameterizations
        """
        # Compute Jacobian of transformation
        jac_func = jacobian(transform_func)
        J = jac_func(params_original)
        
        # Log absolute determinant
        log_abs_det = np.log(np.abs(np.linalg.det(J)))
        
        # Adjusted log likelihood
        return log_likelihood + log_abs_det
    
    @staticmethod
    def log_transform_adjustment(log_likelihood: float,
                                params: np.ndarray,
                                log_indices: List[int]) -> float:
        """
        Specific adjustment for log-transformed parameters
        Common in climate for positive parameters (e.g., diffusivity)
        """
        adjustment = -np.sum(np.log(params[log_indices]))
        return log_likelihood + adjustment
    
    @staticmethod
    def box_cox_adjustment(log_likelihood: float,
                          data: np.ndarray,
                          lambda_param: float) -> float:
        """
        Box-Cox transformation adjustment
        Often used for precipitation, wind speed
        """
        if lambda_param == 0:
            adjustment = -np.sum(np.log(data))
        else:
            adjustment = (lambda_param - 1) * np.sum(np.log(data))
        
        return log_likelihood + adjustment


class ClimateSpecificCriteria:
    """Information criteria specific to climate models"""
    
    @staticmethod
    def skill_weighted_ic(ic_values: List[float],
                         skill_scores: List[float],
                         weight_func: str = 'exponential') -> float:
        """
        Weight information criteria by model skill
        Accounts for out-of-sample performance
        """
        ic_values = np.array(ic_values)
        skill_scores = np.array(skill_scores)
        
        if weight_func == 'exponential':
            weights = np.exp(skill_scores)
        elif weight_func == 'linear':
            weights = skill_scores
        else:
            weights = skill_scores**2
        
        weights /= np.sum(weights)
        return np.sum(weights * ic_values)
    
    @staticmethod
    def ensemble_ic(member_likelihoods: np.ndarray,
                   member_params: List[int],
                   n_data: int,
                   correlation: float = 0.0) -> float:
        """
        Information criterion for ensemble models
        Accounts for inter-member correlation
        """
        n_members = len(member_params)
        total_params = sum(member_params)
        
        # Effective parameters accounting for correlation
        if correlation > 0:
            # Reduce effective parameters for correlated members
            eff_params = total_params * (1 - correlation * (n_members - 1) / n_members)
        else:
            eff_params = total_params
        
        # Ensemble log likelihood
        ensemble_ll = np.sum(special.logsumexp(member_likelihoods, axis=0) - 
                            np.log(n_members))
        
        # AICc for ensemble
        return ClimateModelSelection.aicc(ensemble_ll, int(eff_params), n_data)
    
    @staticmethod
    def regime_dependent_ic(log_likelihoods: Dict[str, float],
                           n_params: Dict[str, int],
                           regime_probs: Dict[str, float],
                           n_data: int) -> float:
        """
        Information criterion for regime-switching models
        E.g., ENSO phases, seasonal cycles
        """
        # Weighted average across regimes
        weighted_ll = sum(p * ll for (regime, p), ll in 
                         zip(regime_probs.items(), log_likelihoods.values()))
        
        # Total parameters includes regime switching
        total_params = sum(n_params.values()) + len(regime_probs) - 1
        
        return ClimateModelSelection.aicc(weighted_ll, total_params, n_data)
    
    @staticmethod
    def hierarchical_ic(level_likelihoods: List[float],
                       level_params: List[int],
                       level_data_sizes: List[int],
                       cross_level_params: int) -> float:
        """
        For hierarchical climate models (global -> regional -> local)
        """
        total_ll = sum(level_likelihoods)
        total_params = sum(level_params) + cross_level_params
        total_n = sum(level_data_sizes)
        
        # Use AICc with total counts
        return ClimateModelSelection.aicc(total_ll, total_params, total_n)


class ModelDiagnostics:
    """Diagnostic tools for model selection"""
    
    @staticmethod
    def likelihood_ratio_test(ll_restricted: float,
                             ll_full: float,
                             df: int) -> Tuple[float, float]:
        """
        Likelihood ratio test between nested models
        Returns (test_statistic, p_value)
        """
        lr = 2 * (ll_full - ll_restricted)
        p_value = 1 - stats.chi2.cdf(lr, df)
        return lr, p_value
    
    @staticmethod
    def vuong_test(ll_1: np.ndarray, ll_2: np.ndarray) -> Tuple[float, float]:
        """
        Vuong test for non-nested models
        Returns (z_statistic, p_value)
        """
        n = len(ll_1)
        lr_i = ll_1 - ll_2
        
        lr_mean = np.mean(lr_i)
        lr_var = np.var(lr_i, ddof=1)
        
        z = np.sqrt(n) * lr_mean / np.sqrt(lr_var)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
        
        return z, p_value
    
    @staticmethod
    def information_matrix_test(observed_info: np.ndarray,
                              expected_info: np.ndarray) -> float:
        """
        White's information matrix test for model misspecification
        Returns test statistic
        """
        diff = observed_info - expected_info
        # Full White test with proper weighting matrix
        n = observed_info.shape[0]
        # Use inverse of expected information as weighting matrix (when well-conditioned)
        try:
            weight_inv = np.linalg.inv(expected_info + 1e-8 * np.eye(n))
            test_stat = np.trace(diff @ weight_inv @ diff.T)
        except np.linalg.LinAlgError:
            # Fallback to Frobenius norm if matrix is singular
            test_stat = np.linalg.norm(diff, 'fro')
        return test_stat


def select_climate_model(models: List[Dict],
                        data: np.ndarray,
                        selection_criterion: str = 'aicc',
                        ensemble_average: bool = True) -> Dict:
    """
    Main function for climate model selection
    
    Args:
        models: List of model dicts with 'name', 'log_likelihood', 'n_params'
        data: Observational data
        selection_criterion: 'aic', 'aicc', 'bic', 'waic', 'loo'
        ensemble_average: Whether to compute model averaging
    
    Returns:
        Dictionary with selected model and diagnostics
    """
    n_data = len(data)
    selector = ClimateModelSelection()
    
    # Compute IC for each model
    ic_values = []
    for model in models:
        ll = model['log_likelihood']
        k = model['n_params']
        
        if selection_criterion == 'aic':
            ic = selector.aic(ll, k)
        elif selection_criterion == 'aicc':
            ic = selector.aicc(ll, k, n_data)
        elif selection_criterion == 'bic':
            ic = selector.bic(ll, k, n_data)
        elif selection_criterion == 'waic':
            # Requires posterior samples
            ic, _ = selector.waic(model.get('log_pointwise_likelihood'))
        else:
            ic = selector.aicc(ll, k, n_data)  # Default to AICc
        
        ic_values.append(ic)
    
    # Find best model
    best_idx = np.argmin(ic_values)
    best_model = models[best_idx]
    
    # Compute model weights for averaging
    if ensemble_average:
        # Convert IC to weights (Burnham & Anderson 2002)
        delta_ic = np.array(ic_values) - np.min(ic_values)
        weights = np.exp(-0.5 * delta_ic)
        weights /= np.sum(weights)
    else:
        weights = None
    
    return {
        'best_model': best_model['name'],
        'ic_values': ic_values,
        'model_weights': weights,
        'criterion_used': selection_criterion,
        'n_data': n_data
    }


if __name__ == "__main__":
    # Example usage
    print("Climate Model Selection Tools")
    print("-" * 50)
    
    # Simulated model comparison
    models = [
        {'name': 'Simple Energy Balance', 'log_likelihood': -150.0, 'n_params': 5},
        {'name': 'Two-Box Ocean', 'log_likelihood': -140.0, 'n_params': 12},
        {'name': 'Full Complexity', 'log_likelihood': -135.0, 'n_params': 25}
    ]
    
    n_data = 100
    fake_data = np.random.randn(n_data)
    
    result = select_climate_model(models, fake_data, selection_criterion='aicc')
    
    print(f"Best model: {result['best_model']}")
    print(f"IC values: {result['ic_values']}")
    print(f"Model weights: {result['model_weights']}")