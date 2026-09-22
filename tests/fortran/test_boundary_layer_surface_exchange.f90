program test_boundary_layer_surface_exchange
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_thermodynamic_reference_values, only: &
        DRY_AIR_HEAT_CAPACITY_CP_J_KG_K
    use climate_component_exchange, only: component_exchange_budget
    use climate_boundary_layer_surface_exchange, only: &
        boundary_layer_exchange_parameters, boundary_layer_surface_fluxes, &
        boundary_layer_tendencies, reference_boundary_layer_exchange_parameters, &
        evaluate_boundary_layer_surface_exchange, &
        BOUNDARY_LAYER_EXCHANGE_OK, BOUNDARY_LAYER_EXCHANGE_ERR_MASS, &
        BOUNDARY_LAYER_EXCHANGE_ERR_HEAT_CAPACITY, &
        BOUNDARY_LAYER_EXCHANGE_ERR_NONFINITE, BOUNDARY_LAYER_EXCHANGE_ERR_COMPONENT
    implicit none

    integer, parameter :: dp = real64

    call test_fluxes_map_to_explicit_tendencies_and_budget()
    call test_negative_flux_reverses_tendencies()
    call test_tendencies_are_independent_of_accounting_timestep()
    call test_heat_capacity_is_explicit_and_overridable()
    call test_invalid_layer_mass_and_heat_capacity_fail_closed()
    call test_invalid_component_domain_fails_closed()
    call test_nonfinite_flux_fails_closed()

contains

    subroutine test_fluxes_map_to_explicit_tendencies_and_budget()
        type(boundary_layer_surface_fluxes) :: fluxes
        type(boundary_layer_exchange_parameters) :: parameters
        type(boundary_layer_tendencies) :: tendencies
        type(component_exchange_budget) :: budget
        real(dp), parameter :: area = 100.0_dp
        real(dp), parameter :: mass = 2000.0_dp
        real(dp), parameter :: dt = 20.0_dp
        integer :: ierr

        parameters = reference_boundary_layer_exchange_parameters()
        fluxes = boundary_layer_surface_fluxes( &
            sensible_heat_w_m2=50.0_dp, water_mass_kg_m2_s=2.0e-4_dp, &
            stress_x_n_m2=0.4_dp, stress_y_n_m2=-0.1_dp)
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, mass, area, dt, parameters, tendencies, budget, ierr)

        call require(ierr == BOUNDARY_LAYER_EXCHANGE_OK, 'surface exchange succeeds')
        call assert_close(tendencies%dry_air_temperature_k_s, &
            50.0_dp * area / (mass * DRY_AIR_HEAT_CAPACITY_CP_J_KG_K), &
            1.0e-14_dp, 'temperature tendency units')
        call assert_close(tendencies%water_mass_fraction_s_inv, &
            2.0e-4_dp * area / mass, 1.0e-14_dp, 'water tendency units')
        call assert_close(tendencies%wind_x_m_s2, &
            0.4_dp * area / mass, 1.0e-14_dp, 'x momentum tendency units')
        call assert_close(tendencies%wind_y_m_s2, &
            -0.1_dp * area / mass, 1.0e-14_dp, 'y momentum tendency units')

        call assert_close(budget%source_energy_j + budget%target_energy_j, &
            0.0_dp, 0.0_dp, 'energy exchange closes')
        call assert_close(budget%source_water_mass_kg + budget%target_water_mass_kg, &
            0.0_dp, 0.0_dp, 'water exchange closes')
        call assert_close(budget%source_momentum_x_n_s + budget%target_momentum_x_n_s, &
            0.0_dp, 0.0_dp, 'x momentum exchange closes')
        call assert_close(budget%source_momentum_y_n_s + budget%target_momentum_y_n_s, &
            0.0_dp, 0.0_dp, 'y momentum exchange closes')
    end subroutine test_fluxes_map_to_explicit_tendencies_and_budget


    subroutine test_negative_flux_reverses_tendencies()
        type(boundary_layer_surface_fluxes) :: fluxes
        type(boundary_layer_exchange_parameters) :: parameters
        type(boundary_layer_tendencies) :: tendencies
        type(component_exchange_budget) :: budget
        integer :: ierr

        parameters = reference_boundary_layer_exchange_parameters()
        fluxes = boundary_layer_surface_fluxes( &
            sensible_heat_w_m2=-10.0_dp, water_mass_kg_m2_s=-1.0e-4_dp, &
            stress_x_n_m2=-0.2_dp, stress_y_n_m2=0.0_dp)
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 1000.0_dp, 10.0_dp, 5.0_dp, parameters, tendencies, budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_OK, 'negative exchange succeeds')
        call require(tendencies%dry_air_temperature_k_s < 0.0_dp, &
            'negative sensible heat cools declared atmospheric target')
        call require(tendencies%water_mass_fraction_s_inv < 0.0_dp, &
            'negative water flux removes target water')
        call require(tendencies%wind_x_m_s2 < 0.0_dp, &
            'negative stress reverses acceleration')
    end subroutine test_negative_flux_reverses_tendencies


    subroutine test_tendencies_are_independent_of_accounting_timestep()
        type(boundary_layer_surface_fluxes) :: fluxes
        type(boundary_layer_exchange_parameters) :: parameters
        type(boundary_layer_tendencies) :: short_step, long_step
        type(component_exchange_budget) :: short_budget, long_budget
        integer :: ierr

        parameters = reference_boundary_layer_exchange_parameters()
        fluxes = boundary_layer_surface_fluxes( &
            sensible_heat_w_m2=12.0_dp, water_mass_kg_m2_s=3.0e-5_dp, &
            stress_x_n_m2=0.05_dp, stress_y_n_m2=0.0_dp)
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 500.0_dp, 20.0_dp, 5.0_dp, parameters, &
            short_step, short_budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_OK, 'short step succeeds')
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 500.0_dp, 20.0_dp, 10.0_dp, parameters, &
            long_step, long_budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_OK, 'long step succeeds')

        call assert_close(short_step%dry_air_temperature_k_s, &
            long_step%dry_air_temperature_k_s, 0.0_dp, 'temperature tendency rate')
        call assert_close(short_step%water_mass_fraction_s_inv, &
            long_step%water_mass_fraction_s_inv, 0.0_dp, 'water tendency rate')
        call assert_close(short_step%wind_x_m_s2, &
            long_step%wind_x_m_s2, 0.0_dp, 'momentum tendency rate')
        call assert_close(long_budget%target_energy_j, &
            2.0_dp * short_budget%target_energy_j, 0.0_dp, 'budget scales with time')
    end subroutine test_tendencies_are_independent_of_accounting_timestep


    subroutine test_heat_capacity_is_explicit_and_overridable()
        type(boundary_layer_surface_fluxes) :: fluxes
        type(boundary_layer_exchange_parameters) :: reference_parameters
        type(boundary_layer_exchange_parameters) :: doubled_parameters
        type(boundary_layer_tendencies) :: reference_tendency, doubled_tendency
        type(component_exchange_budget) :: budget
        integer :: ierr

        reference_parameters = reference_boundary_layer_exchange_parameters()
        doubled_parameters = reference_boundary_layer_exchange_parameters()
        fluxes = boundary_layer_surface_fluxes( &
            sensible_heat_w_m2=20.0_dp, water_mass_kg_m2_s=0.0_dp, &
            stress_x_n_m2=0.0_dp, stress_y_n_m2=0.0_dp)
        doubled_parameters%dry_air_heat_capacity_j_kg_k = &
            2.0_dp * reference_parameters%dry_air_heat_capacity_j_kg_k
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 1000.0_dp, 10.0_dp, 1.0_dp, reference_parameters, &
            reference_tendency, budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_OK, 'reference cp succeeds')
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 1000.0_dp, 10.0_dp, 1.0_dp, doubled_parameters, &
            doubled_tendency, budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_OK, 'override cp succeeds')
        call assert_close(doubled_tendency%dry_air_temperature_k_s, &
            0.5_dp * reference_tendency%dry_air_temperature_k_s, 1.0e-14_dp, &
            'temperature tendency responds to explicit cp')
    end subroutine test_heat_capacity_is_explicit_and_overridable


    subroutine test_invalid_layer_mass_and_heat_capacity_fail_closed()
        type(boundary_layer_surface_fluxes) :: fluxes
        type(boundary_layer_exchange_parameters) :: parameters
        type(boundary_layer_tendencies) :: tendencies
        type(component_exchange_budget) :: budget
        integer :: ierr

        parameters = reference_boundary_layer_exchange_parameters()
        fluxes = boundary_layer_surface_fluxes( &
            sensible_heat_w_m2=1.0_dp, water_mass_kg_m2_s=0.0_dp, &
            stress_x_n_m2=0.0_dp, stress_y_n_m2=0.0_dp)
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 0.0_dp, 1.0_dp, 1.0_dp, parameters, tendencies, budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_ERR_MASS, &
            'layer air mass must be positive')
        call require(tendencies%dry_air_temperature_k_s == 0.0_dp, &
            'invalid mass clears tendency')

        parameters%dry_air_heat_capacity_j_kg_k = 0.0_dp
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 1.0_dp, 1.0_dp, 1.0_dp, parameters, tendencies, budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_ERR_HEAT_CAPACITY, &
            'heat capacity must be positive')
    end subroutine test_invalid_layer_mass_and_heat_capacity_fail_closed


    subroutine test_invalid_component_domain_fails_closed()
        type(boundary_layer_surface_fluxes) :: fluxes
        type(boundary_layer_exchange_parameters) :: parameters
        type(boundary_layer_tendencies) :: tendencies
        type(component_exchange_budget) :: budget
        integer :: ierr

        parameters = reference_boundary_layer_exchange_parameters()
        fluxes = boundary_layer_surface_fluxes( &
            sensible_heat_w_m2=1.0_dp, water_mass_kg_m2_s=0.0_dp, &
            stress_x_n_m2=0.0_dp, stress_y_n_m2=0.0_dp)
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 1.0_dp, 0.0_dp, 1.0_dp, parameters, tendencies, budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_ERR_COMPONENT, &
            'invalid exchange area is delegated and fails closed')
        call require(budget%target_energy_j == 0.0_dp, &
            'failed component exchange clears budget')
    end subroutine test_invalid_component_domain_fails_closed


    subroutine test_nonfinite_flux_fails_closed()
        type(boundary_layer_surface_fluxes) :: fluxes
        type(boundary_layer_exchange_parameters) :: parameters
        type(boundary_layer_tendencies) :: tendencies
        type(component_exchange_budget) :: budget
        real(dp) :: nan_value
        integer :: ierr

        parameters = reference_boundary_layer_exchange_parameters()
        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        fluxes = boundary_layer_surface_fluxes( &
            sensible_heat_w_m2=0.0_dp, water_mass_kg_m2_s=nan_value, &
            stress_x_n_m2=0.0_dp, stress_y_n_m2=0.0_dp)
        call evaluate_boundary_layer_surface_exchange( &
            fluxes, 1.0_dp, 1.0_dp, 1.0_dp, parameters, tendencies, budget, ierr)
        call require(ierr == BOUNDARY_LAYER_EXCHANGE_ERR_NONFINITE, &
            'nonfinite boundary flux fails closed')
        call require(tendencies%water_mass_fraction_s_inv == 0.0_dp, &
            'nonfinite failure clears tendencies')
    end subroutine test_nonfinite_flux_fails_closed


    subroutine assert_close(actual, expected, relative_tolerance, label)
        real(dp), intent(in) :: actual, expected, relative_tolerance
        character(len=*), intent(in) :: label
        real(dp) :: scale

        scale = max(abs(expected), 1.0_dp)
        if (abs(actual - expected) > relative_tolerance * scale) then
            write(error_unit, '(A,2(1X,ES24.16))') trim(label), actual, expected
            error stop 1
        end if
    end subroutine assert_close


    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(error_unit, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require

end program test_boundary_layer_surface_exchange
