program test_component_exchange
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_component_exchange, only: &
        component_exchange_fluxes, component_exchange_budget, &
        integrate_component_exchange, COMPONENT_EXCHANGE_OK, &
        COMPONENT_EXCHANGE_ERR_AREA, COMPONENT_EXCHANGE_ERR_TIMESTEP, &
        COMPONENT_EXCHANGE_ERR_NONFINITE
    implicit none

    integer, parameter :: dp = real64

    call test_equal_and_opposite_extensive_transfers()
    call test_negative_flux_reverses_direction()
    call test_area_and_time_scaling_are_explicit()
    call test_invalid_domain_fails_closed()
    call test_nonfinite_flux_fails_closed()

contains

    subroutine test_equal_and_opposite_extensive_transfers()
        type(component_exchange_fluxes) :: fluxes
        type(component_exchange_budget) :: budget
        integer :: ierr

        fluxes%energy_w_m2 = 20.0_dp
        fluxes%water_mass_kg_m2_s = 1.0e-4_dp
        fluxes%momentum_x_n_m2 = 0.25_dp
        fluxes%momentum_y_n_m2 = -0.10_dp
        call integrate_component_exchange(fluxes, 50.0_dp, 10.0_dp, budget, ierr)

        call require(ierr == COMPONENT_EXCHANGE_OK, 'exchange must succeed')
        call require(budget%source_energy_j == -10000.0_dp, 'source energy loss')
        call require(budget%target_energy_j == 10000.0_dp, 'target energy gain')
        call require(budget%source_water_mass_kg == -0.05_dp, 'source water loss')
        call require(budget%target_water_mass_kg == 0.05_dp, 'target water gain')
        call require(budget%source_momentum_x_n_s == -125.0_dp, 'source x momentum')
        call require(budget%target_momentum_x_n_s == 125.0_dp, 'target x momentum')
        call require(budget%source_momentum_y_n_s == 50.0_dp, 'source y momentum')
        call require(budget%target_momentum_y_n_s == -50.0_dp, 'target y momentum')
        call require(budget%source_energy_j + budget%target_energy_j == 0.0_dp, &
            'energy exchange must close exactly')
        call require(budget%source_water_mass_kg + budget%target_water_mass_kg == 0.0_dp, &
            'water exchange must close exactly')
        call require(budget%source_momentum_x_n_s + budget%target_momentum_x_n_s == 0.0_dp, &
            'x momentum exchange must close exactly')
        call require(budget%source_momentum_y_n_s + budget%target_momentum_y_n_s == 0.0_dp, &
            'y momentum exchange must close exactly')
    end subroutine test_equal_and_opposite_extensive_transfers


    subroutine test_negative_flux_reverses_direction()
        type(component_exchange_fluxes) :: fluxes
        type(component_exchange_budget) :: budget
        integer :: ierr

        fluxes%energy_w_m2 = -4.0_dp
        fluxes%water_mass_kg_m2_s = -2.0e-3_dp
        call integrate_component_exchange(fluxes, 2.0_dp, 5.0_dp, budget, ierr)

        call require(ierr == COMPONENT_EXCHANGE_OK, 'negative exchange must succeed')
        call require(budget%source_energy_j == 40.0_dp, &
            'negative energy flux transfers target to source')
        call require(budget%target_energy_j == -40.0_dp, &
            'negative energy flux debits target')
        call require(budget%source_water_mass_kg == 0.02_dp, &
            'negative water flux transfers target to source')
        call require(budget%target_water_mass_kg == -0.02_dp, &
            'negative water flux debits target')
    end subroutine test_negative_flux_reverses_direction


    subroutine test_area_and_time_scaling_are_explicit()
        type(component_exchange_fluxes) :: fluxes
        type(component_exchange_budget) :: first, second
        integer :: ierr

        fluxes%energy_w_m2 = 3.0_dp
        call integrate_component_exchange(fluxes, 4.0_dp, 5.0_dp, first, ierr)
        call require(ierr == COMPONENT_EXCHANGE_OK, 'first scaling case')
        call integrate_component_exchange(fluxes, 8.0_dp, 10.0_dp, second, ierr)
        call require(ierr == COMPONENT_EXCHANGE_OK, 'second scaling case')
        call require(second%target_energy_j == 4.0_dp * first%target_energy_j, &
            'extensive transfer scales with area times time')
    end subroutine test_area_and_time_scaling_are_explicit


    subroutine test_invalid_domain_fails_closed()
        type(component_exchange_fluxes) :: fluxes
        type(component_exchange_budget) :: budget
        integer :: ierr

        fluxes%energy_w_m2 = 1.0_dp
        call integrate_component_exchange(fluxes, 0.0_dp, 1.0_dp, budget, ierr)
        call require(ierr == COMPONENT_EXCHANGE_ERR_AREA, 'area must be positive')
        call require(budget%target_energy_j == 0.0_dp, 'invalid area clears output')

        call integrate_component_exchange(fluxes, 1.0_dp, 0.0_dp, budget, ierr)
        call require(ierr == COMPONENT_EXCHANGE_ERR_TIMESTEP, 'timestep must be positive')
        call require(budget%target_energy_j == 0.0_dp, 'invalid timestep clears output')
    end subroutine test_invalid_domain_fails_closed


    subroutine test_nonfinite_flux_fails_closed()
        type(component_exchange_fluxes) :: fluxes
        type(component_exchange_budget) :: budget
        real(dp) :: nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        fluxes%energy_w_m2 = nan_value
        call integrate_component_exchange(fluxes, 1.0_dp, 1.0_dp, budget, ierr)
        call require(ierr == COMPONENT_EXCHANGE_ERR_NONFINITE, &
            'nonfinite component flux must fail closed')
        call require(budget%source_energy_j == 0.0_dp .and. &
            budget%target_energy_j == 0.0_dp, 'nonfinite failure clears budget')
    end subroutine test_nonfinite_flux_fails_closed


    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(error_unit, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require

end program test_component_exchange
