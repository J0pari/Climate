program test_dry_thermodynamics
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_dry_thermodynamics, only: dp, dry_thermo_parameters, THERMO_OK, &
        THERMO_ERR_NONFINITE, THERMO_ERR_PARAMETERS, THERMO_ERR_PRESSURE, &
        THERMO_ERR_TEMPERATURE, THERMO_ERR_PRESSURE_ORDER, compute_exner, &
        compute_potential_temperature, compute_temperature_from_potential, &
        compute_dry_air_density, compute_isothermal_hydrostatic_thickness
    implicit none

    call test_reference_state()
    call test_potential_temperature_round_trip()
    call test_pressure_monotonicity()
    call test_ideal_gas_scaling()
    call test_isothermal_scale_height()
    call test_parameter_sensitivity()
    call test_invalid_inputs_fail_closed()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_reference_state()
        type(dry_thermo_parameters) :: parameters
        real(dp) :: exner, theta
        integer :: ierr

        call compute_exner(parameters%reference_pressure_pa, parameters, exner, ierr)
        call require(ierr == THERMO_OK, 'reference-pressure Exner evaluation must succeed')
        call require(abs(exner - 1.0_dp) < 1.0e-15_dp, 'Exner function must equal one at reference pressure')

        call compute_potential_temperature(300.0_dp, parameters%reference_pressure_pa, &
                                           parameters, theta, ierr)
        call require(ierr == THERMO_OK, 'reference-pressure potential temperature must succeed')
        call require(abs(theta - 300.0_dp) < 1.0e-12_dp, &
                     'potential temperature must equal temperature at reference pressure')
    end subroutine test_reference_state


    subroutine test_potential_temperature_round_trip()
        type(dry_thermo_parameters) :: parameters
        real(dp) :: theta, recovered_temperature
        integer :: ierr

        call compute_potential_temperature(285.0_dp, 85000.0_dp, parameters, theta, ierr)
        call require(ierr == THERMO_OK, 'potential-temperature transform must succeed')
        call compute_temperature_from_potential(theta, 85000.0_dp, parameters, recovered_temperature, ierr)
        call require(ierr == THERMO_OK, 'inverse potential-temperature transform must succeed')
        call require(abs(recovered_temperature - 285.0_dp) < 2.0e-13_dp * 285.0_dp, &
                     'potential-temperature transform and inverse must round trip')
    end subroutine test_potential_temperature_round_trip


    subroutine test_pressure_monotonicity()
        type(dry_thermo_parameters) :: parameters
        real(dp) :: theta_low_pressure, theta_high_pressure
        integer :: ierr

        call compute_potential_temperature(280.0_dp, 70000.0_dp, parameters, theta_low_pressure, ierr)
        call require(ierr == THERMO_OK, 'low-pressure potential temperature must succeed')
        call compute_potential_temperature(280.0_dp, 90000.0_dp, parameters, theta_high_pressure, ierr)
        call require(ierr == THERMO_OK, 'high-pressure potential temperature must succeed')
        call require(theta_low_pressure > theta_high_pressure, &
                     'at fixed temperature, dry potential temperature must increase as pressure decreases')
    end subroutine test_pressure_monotonicity


    subroutine test_ideal_gas_scaling()
        type(dry_thermo_parameters) :: parameters
        real(dp) :: rho, rho_double_pressure, rho_double_temperature
        integer :: ierr

        call compute_dry_air_density(80000.0_dp, 260.0_dp, parameters, rho, ierr)
        call require(ierr == THERMO_OK, 'baseline dry-air density must succeed')
        call compute_dry_air_density(160000.0_dp, 260.0_dp, parameters, rho_double_pressure, ierr)
        call require(ierr == THERMO_OK, 'double-pressure density must succeed')
        call compute_dry_air_density(80000.0_dp, 520.0_dp, parameters, rho_double_temperature, ierr)
        call require(ierr == THERMO_OK, 'double-temperature density must succeed')

        call require(abs(rho_double_pressure - 2.0_dp*rho) < 2.0e-14_dp * rho_double_pressure, &
                     'ideal-gas density must scale linearly with pressure')
        call require(abs(rho_double_temperature - 0.5_dp*rho) < 2.0e-14_dp * rho, &
                     'ideal-gas density must scale inversely with absolute temperature')
    end subroutine test_ideal_gas_scaling


    subroutine test_isothermal_scale_height()
        type(dry_thermo_parameters) :: parameters
        real(dp), parameter :: temperature_k = 250.0_dp
        real(dp) :: thickness, expected_scale_height
        integer :: ierr

        call compute_isothermal_hydrostatic_thickness(100000.0_dp, 100000.0_dp/exp(1.0_dp), &
                                                      temperature_k, parameters, thickness, ierr)
        call require(ierr == THERMO_OK, 'isothermal hydrostatic thickness must succeed')
        expected_scale_height = parameters%gas_constant_j_kg_k * temperature_k / parameters%gravity_m_s2
        call require(abs(thickness - expected_scale_height) < 2.0e-12_dp * expected_scale_height, &
                     'one e-fold pressure decrease must span exactly one isothermal scale height')
    end subroutine test_isothermal_scale_height


    subroutine test_parameter_sensitivity()
        type(dry_thermo_parameters) :: baseline, modified
        real(dp) :: theta_baseline, theta_modified, rho_baseline, rho_modified
        integer :: ierr

        modified = baseline
        modified%heat_capacity_cp_j_kg_k = 1100.0_dp
        call compute_potential_temperature(280.0_dp, 80000.0_dp, baseline, theta_baseline, ierr)
        call require(ierr == THERMO_OK, 'baseline parameter set must succeed')
        call compute_potential_temperature(280.0_dp, 80000.0_dp, modified, theta_modified, ierr)
        call require(ierr == THERMO_OK, 'modified heat capacity must succeed')
        call require(abs(theta_modified - theta_baseline) > 1.0e-6_dp, &
                     'heat-capacity parameter must demonstrably affect the Poisson transform')

        modified = baseline
        modified%gas_constant_j_kg_k = 300.0_dp
        call compute_dry_air_density(90000.0_dp, 290.0_dp, baseline, rho_baseline, ierr)
        call require(ierr == THERMO_OK, 'baseline density parameter set must succeed')
        call compute_dry_air_density(90000.0_dp, 290.0_dp, modified, rho_modified, ierr)
        call require(ierr == THERMO_OK, 'modified gas constant must succeed')
        call require(rho_modified < rho_baseline, &
                     'larger gas constant must reduce dry-air density at fixed pressure and temperature')
    end subroutine test_parameter_sensitivity


    subroutine test_invalid_inputs_fail_closed()
        type(dry_thermo_parameters) :: parameters, invalid_parameters
        real(dp) :: result, nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
        call compute_exner(nan_value, parameters, result, ierr)
        call require(ierr == THERMO_ERR_NONFINITE .and. abs(result) < tiny(1.0_dp), &
                     'non-finite pressure must fail closed')

        call compute_potential_temperature(280.0_dp, 0.0_dp, parameters, result, ierr)
        call require(ierr == THERMO_ERR_PRESSURE .and. abs(result) < tiny(1.0_dp), &
                     'non-positive pressure must fail closed')

        call compute_dry_air_density(100000.0_dp, -1.0_dp, parameters, result, ierr)
        call require(ierr == THERMO_ERR_TEMPERATURE .and. abs(result) < tiny(1.0_dp), &
                     'non-positive absolute temperature must fail closed')

        call compute_isothermal_hydrostatic_thickness(80000.0_dp, 90000.0_dp, 280.0_dp, &
                                                      parameters, result, ierr)
        call require(ierr == THERMO_ERR_PRESSURE_ORDER .and. abs(result) < tiny(1.0_dp), &
                     'geometric layer API must reject reversed lower/upper pressure order')

        invalid_parameters = parameters
        invalid_parameters%heat_capacity_cp_j_kg_k = invalid_parameters%gas_constant_j_kg_k
        call compute_exner(90000.0_dp, invalid_parameters, result, ierr)
        call require(ierr == THERMO_ERR_PARAMETERS .and. abs(result) < tiny(1.0_dp), &
                     'invalid dry-air parameter set must fail closed')
    end subroutine test_invalid_inputs_fail_closed

end program test_dry_thermodynamics
