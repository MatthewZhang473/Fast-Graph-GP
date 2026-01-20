def set_gp_defaults(linear_operator_settings, gpytorch_settings):
    """Set default settings for GPyTorch linear operators used in GRF-GP models."""
    linear_operator_settings.verbose_linalg._default = False
    linear_operator_settings._fast_covar_root_decomposition._default = False
    linear_operator_settings.fast_computations.log_prob._state = True
    gpytorch_settings.max_cholesky_size._global_value = 0
    gpytorch_settings.cg_tolerance._global_value = 1e-2
    gpytorch_settings.max_lanczos_quadrature_iterations._global_value = 1
    gpytorch_settings.num_trace_samples._global_value = 64
    gpytorch_settings.min_preconditioning_size._global_value = 1e10  # no preconditioner
