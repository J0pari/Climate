// Climate parameter estimation with trust-region optimization
// Experimental MPI+CUDA implementation
//
// DATA SOURCE REQUIREMENTS:
//
// 1. CLIMATE MODEL PARAMETER PRIORS:
//    - Source: CMIP6 model documentation (ES-DOC)
//    - Parameters: ~100 tunable parameters per model
//    - Format: JSON/XML parameter cards
//    - Size: ~10MB per model
//    - API: https://es-doc.org/cmip6/models/
//    - Missing: Covariance between parameters
//
// 2. OBSERVATIONAL CONSTRAINTS FOR CALIBRATION:
//    - Global temperature: BEST/HadCRUT5 (1850-present)
//    - Ocean heat: IAP/Cheng (1955-present)
//    - Sea ice: NSIDC (1979-present)
//    - Format: NetCDF4 with uncertainties
//    - Size: ~10GB total
//    - Preprocessing: Compute cost function terms
//
// 3. OPTIMIZATION BENCHMARKS:
//    - Source: Held-Suarez test case
//    - Format: Analytical solutions in HDF5
//    - Size: <100MB
//    - Purpose: Validate optimizer convergence
//
// 4. PARALLEL DECOMPOSITION:
//    - Domain: Lat-lon blocks for spatial fields
//    - Parameters: Distributed across MPI ranks
//    - GPU: Each rank offloads to local GPU
//    - Required: 4+ GPUs for production runs

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Eigenvalues>
#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>
#include <snopt.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cudnn.h>
#include <nccl.h>
#include <omp.h>
#include <mpi.h>
#include <mpi-ext.h>  // GPU-aware MPI
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions.hpp>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_statistics.h>
#include <nlopt.hpp>
#include <netcdf>
#include <hdf5.h>
#include <fmt/format.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

using namespace Eigen;
using namespace Ipopt;

namespace ClimateOptimization {

// ─────────────────────────────────────────────────────────────────────────────
// DATA SOURCE REQUIREMENTS  
// ─────────────────────────────────────────────────────────────────────────────
// NEEDED: Historical climate observations for optimization targets:
//   - HadCRUT5/GISTEMP global temperature (1850-present, monthly)
//   - Format: NetCDF4 with ensemble members for uncertainty
//   - Resolution: 5° × 5° gridded or global mean time series
//   - Source: https://www.metoffice.gov.uk/hadobs/hadcrut5/
//
// NEEDED: Ocean heat content observations:
//   - NOAA/NCEI 0-2000m OHC (1955-present, quarterly)
//   - Format: NetCDF4 with error estimates
//   - Source: https://www.ncei.noaa.gov/products/ocean-heat-content
//   - Missing: Deep ocean (>2000m) heat content before Argo era
//
// NEEDED: Radiative forcing time series:
//   - IPCC AR6 assessed forcing (1750-2019)
//   - Format: CSV/Excel from IPCC data repository
//   - Components: GHG, aerosol, land use, solar, volcanic
//   - Missing: Real-time forcing updates post-2019
//
// NEEDED: Climate sensitivity constraints:
//   - Paleoclimate proxy reconstructions (LGM, PETM)
//   - Format: Published estimates with uncertainties (JSON/CSV)
//   - Missing: Consistent methodology across studies
//   - Missing: Process-based cloud feedback observations
//
// NEEDED: Carbon cycle observations:
//   - Global Carbon Project annual budget (1959-present)
//   - Format: Excel/CSV from globalcarbonproject.org
//   - Missing: Seasonal cycle amplitude trends
//   - Missing: Permafrost carbon fluxes
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// CLIMATE MODEL PARAMETER BOUNDS
// ─────────────────────────────────────────────────────────────────────────────

// IPCC AR6 + observational constraints with uncertainties
struct ClimateParameterBounds {
    // Equilibrium Climate Sensitivity (K) - AR6 very likely range
    double ecs_min = 2.0, ecs_max = 5.0, ecs_best = 3.0, ecs_sigma = 0.7;
    
    // Transient Climate Response (K) - AR6 likely range  
    double tcr_min = 1.4, tcr_max = 2.2, tcr_best = 1.8, tcr_sigma = 0.3;
    
    // Aerosol effective radiative forcing 1750-2019 (W/m²) - AR6 assessed
    double aerosol_min = -1.6, aerosol_max = -0.6, aerosol_best = -1.06, aerosol_sigma = 0.23;
    
    // Cloud feedback (W/m²/K) - Sherwood et al. 2020 + WCRP assessment
    double cloud_min = -0.25, cloud_max = 0.45, cloud_best = 0.19, cloud_sigma = 0.13;
    
    // Ocean vertical diffusivity (cm²/s) - observational tracers
    double ocean_diff_min = 0.1, ocean_diff_max = 2.5, ocean_diff_best = 0.5, ocean_diff_sigma = 0.3;
    
    // Land carbon cycle feedback γ_L (GtC/K) - C4MIP constraint
    double carbon_min = -20.0, carbon_max = 80.0, carbon_best = 30.0, carbon_sigma = 20.0;
    
    // Ice sheet sensitivity (m SLE/K) - paleo constraints
    double ice_sens_min = 0.5, ice_sens_max = 1.5, ice_sens_best = 1.0, ice_sens_sigma = 0.2;
    
    // Additional constraints from emergent relationships
    double pattern_effect_min = -0.5, pattern_effect_max = 0.5;  // W/m²
    double amoc_sensitivity_min = -2.0, amoc_sensitivity_max = 0.0;  // Sv/K
};

// ─────────────────────────────────────────────────────────────────────────────
// OPTIMIZATION RESULT STRUCTURE
// ─────────────────────────────────────────────────────────────────────────────

struct OptimizationResult {
    VectorXd parameters;
    VectorXd uncertainties;
    MatrixXd correlation_matrix;
    double chi_squared;
    double log_likelihood;
    bool converged;
    int iterations;
    int function_evaluations;
    int jacobian_evaluations;
    
    // Fit statistics
    double R2;
    double adjusted_R2;
    double RMSE;
    double MAE;
    double AIC;
    double BIC;
    double durbin_watson;
    
    // Diagnostics
    double condition_number;
    VectorXd eigenvalues;
    VectorXd cooks_distance;
    VectorXd leverage_points;
    std::vector<int> outliers;
    
    // Convergence history
    std::vector<double> chi_squared_history;
    std::vector<double> gradient_norm_history;
    std::vector<double> trust_radius_history;
};

// ─────────────────────────────────────────────────────────────────────────────
// TRUST-REGION LEVENBERG-MARQUARDT OPTIMIZER
// ─────────────────────────────────────────────────────────────────────────────

class ExperimentalClimateOptimizer {
private:
    // Options
    int max_iterations_ = 1000;
    double gradient_tolerance_ = 1e-8;
    double parameter_tolerance_ = 1e-10;
    double chi_squared_tolerance_ = 1e-10;
    double initial_trust_radius_ = 1.0;
    double min_trust_radius_ = 1e-12;
    double max_trust_radius_ = 1e12;
    int num_restarts_ = 10;
    
    // Damping parameters
    double lambda_ = 0.01;  // Levenberg-Marquardt damping
    double nu_ = 2.0;       // Trust region adjustment factor
    
    // GPU handles
    cublasHandle_t cublas_handle_;
    cusolverDnHandle_t cusolver_handle_;
    
    // MPI communicator for distributed optimization
    MPI_Comm comm_;
    int rank_, size_;
    
    // Logger
    std::shared_ptr<spdlog::logger> logger_;
    
public:
    TrustRegionClimateOptimizer() {
        // Initialize CUDA
        cudaSetDevice(0);
        cublasCreate(&cublas_handle_);
        cusolverDnCreate(&cusolver_handle_);
        
        // Initialize MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        comm_ = MPI_COMM_WORLD;
        
        // Setup logger
        logger_ = spdlog::get("climate_optimizer");
        if (!logger_) {
            logger_ = spdlog::stdout_logger_mt("climate_optimizer");
        }
    }
    
    ~TrustRegionClimateOptimizer() {
        cublasDestroy(cublas_handle_);
        cusolverDnDestroy(cusolver_handle_);
    }
    
    /**
     * Attempted optimization routine with restarts (unvalidated)
     */
    OptimizationResult optimize(
        std::function<VectorXd(const VectorXd&)> model,
        const VectorXd& initial_params,
        const MatrixXd& observations,
        const MatrixXd& observation_errors,
        const ClimateParameterBounds& bounds
    ) {
        OptimizationResult best_result;
        best_result.chi_squared = std::numeric_limits<double>::infinity();
        
        // Perform multiple restarts with Latin Hypercube Sampling
        auto starting_points = generateSmartStartingPoints(initial_params, bounds, num_restarts_);
        
        #pragma omp parallel for
        for (int restart = 0; restart < num_restarts_; ++restart) {
            try {
                auto result = runSingleOptimization(
                    model, 
                    starting_points.col(restart),
                    observations,
                    observation_errors,
                    bounds
                );
                
                #pragma omp critical
                {
                    if (result.converged && result.chi_squared < best_result.chi_squared) {
                        best_result = result;
                        logger_->info("Better solution found at restart {}: χ² = {}", 
                                     restart, result.chi_squared);
                        
                        // Early termination for potentially good fit (needs validation)
                        if (result.R2 > 0.9999) {
                            logger_->info("High R2 observed (may be overfitting), terminating early");
                        }
                    }
                }
            } catch (const std::exception& e) {
                logger_->warn("Restart {} failed: {}", restart, e.what());
            }
        }
        
        if (!best_result.converged) {
            throw std::runtime_error("Optimization failed after all restarts");
        }
        
        // Compute parameter uncertainties from Hessian
        computeParameterUncertainties(best_result, model, observations, observation_errors);
        
        return best_result;
    }
    
private:
    /**
     * Single optimization run
     */
    OptimizationResult runSingleOptimization(
        std::function<VectorXd(const VectorXd&)> model,
        const VectorXd& initial_params,
        const MatrixXd& observations,
        const MatrixXd& observation_errors,
        const ClimateParameterBounds& bounds
    ) {
        OptimizationResult result;
        result.parameters = initial_params;
        result.iterations = 0;
        result.function_evaluations = 0;
        result.jacobian_evaluations = 0;
        result.converged = false;
        
        double trust_radius = initial_trust_radius_;
        VectorXd params = initial_params;
        
        // Initial residuals
        VectorXd residuals = calculateResiduals(model, params, observations);
        double chi_squared = residuals.squaredNorm();
        
        while (result.iterations < max_iterations_ && !result.converged) {
            result.iterations++;
            
            // Calculate Jacobian using finite differences or automatic differentiation
            MatrixXd jacobian = calculateJacobianGPU(model, params, observations);
            result.jacobian_evaluations++;
            
            // Gradient and Hessian approximation
            VectorXd gradient = jacobian.transpose() * residuals;
            MatrixXd hessian = jacobian.transpose() * jacobian;
            
            // Check gradient convergence
            double gradient_norm = gradient.norm();
            result.gradient_norm_history.push_back(gradient_norm);
            
            if (gradient_norm < gradient_tolerance_) {
                result.converged = true;
                logger_->info("Converged: gradient norm {} < {}", 
                             gradient_norm, gradient_tolerance_);
                break;
            }
            
            // Levenberg-Marquardt step with trust region
            VectorXd step = computeTrustRegionStep(
                gradient, hessian, trust_radius, lambda_
            );
            
            // Enforce parameter bounds
            VectorXd new_params = params + step;
            enforceParameterBounds(new_params, bounds);
            
            // Evaluate new residuals
            VectorXd new_residuals = calculateResiduals(model, new_params, observations);
            double new_chi_squared = new_residuals.squaredNorm();
            result.function_evaluations++;
            
            // Actual vs predicted reduction
            double actual_reduction = chi_squared - new_chi_squared;
            double predicted_reduction = -gradient.dot(step) - 0.5 * step.transpose() * hessian * step;
            double ratio = actual_reduction / predicted_reduction;
            
            // Update trust region
            if (ratio > 0.75) {
                trust_radius = std::min(2.0 * trust_radius, max_trust_radius_);
                lambda_ *= 0.5;
            } else if (ratio < 0.25) {
                trust_radius = std::max(0.25 * trust_radius, min_trust_radius_);
                lambda_ *= 2.0;
            }
            
            // Accept or reject step
            if (ratio > 0) {
                params = new_params;
                residuals = new_residuals;
                chi_squared = new_chi_squared;
                
                result.chi_squared_history.push_back(chi_squared);
                result.trust_radius_history.push_back(trust_radius);
                
                // Check chi-squared convergence
                if (actual_reduction < chi_squared_tolerance_ * chi_squared) {
                    result.converged = true;
                    logger_->info("Converged: χ² reduction {} < tolerance", 
                                 actual_reduction);
                    break;
                }
            }
        }
        
        result.parameters = params;
        result.chi_squared = chi_squared;
        
        // Calculate fit statistics
        calculateFitStatistics(result, model, observations, observation_errors);
        
        return result;
    }
    
    /**
     * Experimental GPU Jacobian calculation (untested)
     */
    MatrixXd calculateJacobianGPU(
        std::function<VectorXd(const VectorXd&)> model,
        const VectorXd& params,
        const MatrixXd& observations
    ) {
        int n_params = params.size();
        int n_obs = observations.size();
        MatrixXd jacobian(n_obs, n_params);
        
        // Allocate GPU memory
        double *d_params, *d_jacobian;
        cudaMalloc(&d_params, n_params * sizeof(double));
        cudaMalloc(&d_jacobian, n_obs * n_params * sizeof(double));
        
        // Copy parameters to GPU
        cudaMemcpy(d_params, params.data(), n_params * sizeof(double), 
                   cudaMemcpyHostToDevice);
        
        // Finite differences in parallel on GPU
        const double eps = 1e-8;
        dim3 block(32, 32);
        dim3 grid((n_params + 31) / 32, (n_obs + 31) / 32);
        
        // Launch kernel for Jacobian calculation
        calculateJacobianKernel<<<grid, block>>>(
            d_params, d_jacobian, n_params, n_obs, eps
        );
        
        // Copy result back
        cudaMemcpy(jacobian.data(), d_jacobian, n_obs * n_params * sizeof(double),
                   cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_params);
        cudaFree(d_jacobian);
        
        return jacobian;
    }
    
    /**
     * Attempt trust region step using dogleg method (theoretical)
     */
    VectorXd computeTrustRegionStep(
        const VectorXd& gradient,
        const MatrixXd& hessian,
        double trust_radius,
        double lambda
    ) {
        int n = gradient.size();
        
        // Regularized Hessian for Levenberg-Marquardt
        MatrixXd H_reg = hessian + lambda * MatrixXd::Identity(n, n);
        
        // Compute Newton step
        VectorXd newton_step = -H_reg.ldlt().solve(gradient);
        double newton_norm = newton_step.norm();
        
        if (newton_norm <= trust_radius) {
            return newton_step;
        }
        
        // Compute Cauchy step (steepest descent)
        double alpha = gradient.squaredNorm() / (gradient.transpose() * H_reg * gradient);
        VectorXd cauchy_step = -alpha * gradient;
        double cauchy_norm = cauchy_step.norm();
        
        if (cauchy_norm >= trust_radius) {
            // Return scaled Cauchy step
            return (trust_radius / cauchy_norm) * cauchy_step;
        }
        
        // Dogleg method: interpolate between Cauchy and Newton
        VectorXd diff = newton_step - cauchy_step;
        double a = diff.squaredNorm();
        double b = 2.0 * cauchy_step.dot(diff);
        double c = cauchy_step.squaredNorm() - trust_radius * trust_radius;
        
        double discriminant = b * b - 4.0 * a * c;
        double tau = (-b + std::sqrt(discriminant)) / (2.0 * a);
        
        return cauchy_step + tau * diff;
    }
    
    /**
     * Generate starting points using Latin Hypercube Sampling attempt
     */
    MatrixXd generateSmartStartingPoints(
        const VectorXd& initial_params,
        const ClimateParameterBounds& bounds,
        int n_starts
    ) {
        int n_params = initial_params.size();
        MatrixXd starting_points(n_params, n_starts);
        
        // Latin Hypercube Sampling for better coverage
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (int j = 0; j < n_params; ++j) {
            std::vector<double> intervals(n_starts);
            for (int i = 0; i < n_starts; ++i) {
                intervals[i] = (i + std::uniform_real_distribution<>(0, 1)(gen)) / n_starts;
            }
            std::shuffle(intervals.begin(), intervals.end(), gen);
            
            // Map to parameter bounds
            double min_val = getMinBound(j, bounds);
            double max_val = getMaxBound(j, bounds);
            
            for (int i = 0; i < n_starts; ++i) {
                if (i == 0) {
                    // First start is always the initial guess
                    starting_points(j, i) = initial_params(j);
                } else {
                    starting_points(j, i) = min_val + intervals[i] * (max_val - min_val);
                }
            }
        }
        
        return starting_points;
    }
    
    /**
     * Calculate residuals between model and observations
     */
    VectorXd calculateResiduals(
        std::function<VectorXd(const VectorXd&)> model,
        const VectorXd& params,
        const MatrixXd& observations
    ) {
        VectorXd predictions = model(params);
        VectorXd obs_vec = Map<const VectorXd>(observations.data(), observations.size());
        return predictions - obs_vec;
    }
    
    /**
     * Estimate parameter uncertainties from approximate Fisher Information
     */
    void computeParameterUncertainties(
        OptimizationResult& result,
        std::function<VectorXd(const VectorXd&)> model,
        const MatrixXd& observations,
        const MatrixXd& observation_errors
    ) {
        // Fisher Information Matrix approximation at optimum
        MatrixXd jacobian = calculateJacobianGPU(model, result.parameters, observations);
        
        // Weight by observation errors
        MatrixXd W = observation_errors.cwiseInverse().asDiagonal();
        MatrixXd fisher = jacobian.transpose() * W * W * jacobian;
        
        // Parameter covariance matrix
        MatrixXd covariance = fisher.inverse();
        
        // Extract uncertainties (standard errors)
        result.uncertainties = covariance.diagonal().cwiseSqrt();
        
        // Correlation matrix
        VectorXd std_devs = result.uncertainties;
        result.correlation_matrix = covariance.array() / 
                                   (std_devs * std_devs.transpose()).array();
        
        // Condition number for diagnostic
        Eigen::JacobiSVD<MatrixXd> svd(fisher);
        result.condition_number = svd.singularValues()(0) / 
                                 svd.singularValues()(svd.singularValues().size()-1);
        result.eigenvalues = svd.singularValues();
    }
    
    /**
     * Calculate basic fit statistics (may be incomplete)
     */
    void calculateFitStatistics(
        OptimizationResult& result,
        std::function<VectorXd(const VectorXd&)> model,
        const MatrixXd& observations,
        const MatrixXd& observation_errors
    ) {
        int n = observations.size();
        int p = result.parameters.size();
        
        VectorXd predictions = model(result.parameters);
        VectorXd obs_vec = Map<const VectorXd>(observations.data(), n);
        VectorXd residuals = predictions - obs_vec;
        
        // R-squared
        double ss_res = residuals.squaredNorm();
        double ss_tot = (obs_vec.array() - obs_vec.mean()).square().sum();
        result.R2 = 1.0 - ss_res / ss_tot;
        
        // Adjusted R-squared
        result.adjusted_R2 = 1.0 - (1.0 - result.R2) * (n - 1) / (n - p - 1);
        
        // RMSE and MAE
        result.RMSE = std::sqrt(ss_res / n);
        result.MAE = residuals.cwiseAbs().mean();
        
        // AIC and BIC
        double log_likelihood = -0.5 * n * std::log(2 * M_PI) 
                              - 0.5 * n * std::log(ss_res / n) 
                              - 0.5 * n;
        result.AIC = 2 * p - 2 * log_likelihood;
        result.BIC = std::log(n) * p - 2 * log_likelihood;
        
        // Durbin-Watson statistic for autocorrelation
        VectorXd diff = residuals.tail(n-1) - residuals.head(n-1);
        result.durbin_watson = diff.squaredNorm() / ss_res;
        
        // Cook's distance and leverage points
        calculateDiagnostics(result, jacobian, residuals);
    }
    
    /**
     * Calculate regression diagnostics
     */
    void calculateDiagnostics(
        OptimizationResult& result,
        const MatrixXd& jacobian,
        const VectorXd& residuals
    ) {
        int n = residuals.size();
        int p = result.parameters.size();
        
        // Hat matrix H = J(J'J)^(-1)J'
        MatrixXd JtJ_inv = (jacobian.transpose() * jacobian).inverse();
        MatrixXd H = jacobian * JtJ_inv * jacobian.transpose();
        
        // Leverage points (diagonal of hat matrix)
        result.leverage_points = H.diagonal();
        
        // Standardized residuals
        double s2 = residuals.squaredNorm() / (n - p);
        VectorXd std_residuals = residuals.array() / std::sqrt(s2 * (1 - H.diagonal().array()));
        
        // Cook's distance
        result.cooks_distance.resize(n);
        for (int i = 0; i < n; ++i) {
            double h_ii = H(i, i);
            result.cooks_distance(i) = std_residuals(i) * std_residuals(i) * h_ii / (p * (1 - h_ii));
            
            // Flag outliers (Cook's distance > 1)
            if (result.cooks_distance(i) > 1.0) {
                result.outliers.push_back(i);
            }
        }
    }
    
    // Helper functions for parameter bounds
    double getMinBound(int param_index, const ClimateParameterBounds& bounds) {
        switch(param_index) {
            case 0: return bounds.ecs_min;
            case 1: return bounds.tcr_min;
            case 2: return bounds.aerosol_min;
            case 3: return bounds.cloud_min;
            case 4: return bounds.ocean_diff_min;
            case 5: return bounds.carbon_min;
            case 6: return bounds.ice_sens_min;
            default: return -1e10;
        }
    }
    
    double getMaxBound(int param_index, const ClimateParameterBounds& bounds) {
        switch(param_index) {
            case 0: return bounds.ecs_max;
            case 1: return bounds.tcr_max;
            case 2: return bounds.aerosol_max;
            case 3: return bounds.cloud_max;
            case 4: return bounds.ocean_diff_max;
            case 5: return bounds.carbon_max;
            case 6: return bounds.ice_sens_max;
            default: return 1e10;
        }
    }
    
    void enforceParameterBounds(VectorXd& params, const ClimateParameterBounds& bounds) {
        for (int i = 0; i < params.size(); ++i) {
            params(i) = std::max(getMinBound(i, bounds), 
                                std::min(params(i), getMaxBound(i, bounds)));
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CUDA KERNEL FOR JACOBIAN CALCULATION
// ─────────────────────────────────────────────────────────────────────────────

__global__ void calculateJacobianKernel(
    const double* params,
    double* jacobian,
    int n_params,
    int n_obs,
    double eps
) {
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int obs_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (param_idx < n_params && obs_idx < n_obs) {
        // Finite difference approximation
        // This would call the climate model with perturbed parameters
        // STUB: Should compute via finite differences or adjoint
        jacobian[obs_idx * n_params + param_idx] = NAN;  // Return NaN instead of hiding error
    }
}

} // namespace ClimateOptimization