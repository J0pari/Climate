program test_microphysics_phase_change_budget
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_microphysics_phase_change_budget, only: &
        microphysics_column_budget_rate, evaluate_microphysics_phase_change_budget, &
        MICROPHYSICS_BUDGET_OK, MICROPHYSICS_BUDGET_ERR_EMPTY, &
        MICROPHYSICS_BUDGET_ERR_DIMENSION, MICROPHYSICS_BUDGET_ERR_LATENT_HEAT, &
        MICROPHYSICS_BUDGET_ERR_EXPORT, MICROPHYSICS_BUDGET_ERR_NONFINITE
    implicit none

    integer, parameter :: dp = real64

    call test_phase_change_water_and_energy_budget()
    call test_evaporation_reverses_mass_and_energy_signs()
    call test_precipitation_export_closes_water_and_energy_budgets()
    call test_latent_heat_is_caller_selected()
    call test_invalid_domains_fail_closed()
    call test_nonfinite_inputs_fail_closed()

contains

    subroutine test_phase_change_water_and_energy_budget()
        real(dp), allocatable :: vapor(:), condensate(:), energy(:)
        type(microphysics_column_budget_rate) :: budget
        real(dp), parameter :: latent_heat = 2.5e6_dp
        integer :: ierr

        call evaluate_microphysics_phase_change_budget( &
            [1.0e-4_dp, 2.0e-4_dp], [0.0_dp, 0.0_dp], [0.0_dp, 0.0_dp], &
            latent_heat, vapor, condensate, energy, budget, ierr)

        call require(ierr == MICROPHYSICS_BUDGET_OK, 'condensation budget succeeds')
        call assert_array_close(vapor, [-1.0e-4_dp, -2.0e-4_dp], 0.0_dp, &
            'condensation removes vapor')
        call assert_array_close(condensate, [1.0e-4_dp, 2.0e-4_dp], 0.0_dp, &
            'condensation adds condensate')
        call assert_array_close(energy, latent_heat * [1.0e-4_dp, 2.0e-4_dp], &
            1.0e-12_dp, 'condensation latent heating')
        call assert_close(budget%total_water_tendency_kg_m2_s, 0.0_dp, 1.0e-16_dp, &
            'phase change conserves represented water')
        call assert_close(budget%water_closure_residual_kg_m2_s, 0.0_dp, 1.0e-16_dp, &
            'phase-change water budget closes')
        call assert_close(budget%energy_closure_residual_w_m2, 0.0_dp, 1.0e-12_dp, &
            'phase-change energy budget closes')
    end subroutine test_phase_change_water_and_energy_budget


    subroutine test_evaporation_reverses_mass_and_energy_signs()
        real(dp), allocatable :: vapor(:), condensate(:), energy(:)
        type(microphysics_column_budget_rate) :: budget
        integer :: ierr

        call evaluate_microphysics_phase_change_budget( &
            [-2.0e-4_dp], [0.0_dp], [0.0_dp], 2.5e6_dp, &
            vapor, condensate, energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_OK, 'evaporation budget succeeds')
        call require(vapor(1) > 0.0_dp, 'evaporation adds vapor')
        call require(condensate(1) < 0.0_dp, 'evaporation removes condensate')
        call require(energy(1) < 0.0_dp, 'evaporation cools represented energy budget')
    end subroutine test_evaporation_reverses_mass_and_energy_signs


    subroutine test_precipitation_export_closes_water_and_energy_budgets()
        real(dp), allocatable :: vapor(:), condensate(:), energy(:)
        type(microphysics_column_budget_rate) :: budget
        integer :: ierr

        call evaluate_microphysics_phase_change_budget( &
            [3.0e-4_dp, 1.0e-4_dp], &
            [1.0e-4_dp, 2.0e-4_dp], &
            [40.0_dp, 10.0_dp], &
            2.5e6_dp, vapor, condensate, energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_OK, 'precipitation budget succeeds')
        call assert_close(budget%total_water_tendency_kg_m2_s, -3.0e-4_dp, 1.0e-16_dp, &
            'precipitation is the represented column water sink')
        call assert_close(budget%precipitation_export_kg_m2_s, 3.0e-4_dp, 1.0e-16_dp, &
            'precipitation export is explicit')
        call assert_close(budget%water_closure_residual_kg_m2_s, 0.0_dp, 1.0e-16_dp, &
            'precipitation water budget closes')
        call assert_close(budget%precipitation_energy_export_w_m2, 50.0_dp, 0.0_dp, &
            'precipitation energy export is caller supplied')
        call assert_close(budget%energy_closure_residual_w_m2, 0.0_dp, 1.0e-12_dp, &
            'precipitation energy budget closes')
    end subroutine test_precipitation_export_closes_water_and_energy_budgets


    subroutine test_latent_heat_is_caller_selected()
        real(dp), allocatable :: vapor(:), condensate(:), first_energy(:), second_energy(:)
        type(microphysics_column_budget_rate) :: budget
        integer :: ierr

        call evaluate_microphysics_phase_change_budget( &
            [1.0e-4_dp], [0.0_dp], [0.0_dp], 1.0e6_dp, &
            vapor, condensate, first_energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_OK, 'first latent heat succeeds')
        call evaluate_microphysics_phase_change_budget( &
            [1.0e-4_dp], [0.0_dp], [0.0_dp], 2.0e6_dp, &
            vapor, condensate, second_energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_OK, 'second latent heat succeeds')
        call assert_close(second_energy(1), 2.0_dp * first_energy(1), 0.0_dp, &
            'phase choice latent heat is explicit')
    end subroutine test_latent_heat_is_caller_selected


    subroutine test_invalid_domains_fail_closed()
        real(dp), allocatable :: vapor(:), condensate(:), energy(:)
        type(microphysics_column_budget_rate) :: budget
        integer :: ierr

        call evaluate_microphysics_phase_change_budget( &
            [real(dp) ::], [real(dp) ::], [real(dp) ::], 2.5e6_dp, &
            vapor, condensate, energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_ERR_EMPTY, 'empty column rejected')
        call require(size(vapor) == 0, 'empty failure emits no tendency')

        call evaluate_microphysics_phase_change_budget( &
            [1.0e-4_dp], [0.0_dp, 0.0_dp], [0.0_dp], 2.5e6_dp, &
            vapor, condensate, energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_ERR_DIMENSION, &
            'microphysics vectors must share shape')

        call evaluate_microphysics_phase_change_budget( &
            [1.0e-4_dp], [0.0_dp], [0.0_dp], 0.0_dp, &
            vapor, condensate, energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_ERR_LATENT_HEAT, &
            'latent heat must be explicit and positive')

        call evaluate_microphysics_phase_change_budget( &
            [1.0e-4_dp], [-1.0e-5_dp], [0.0_dp], 2.5e6_dp, &
            vapor, condensate, energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_ERR_EXPORT, &
            'precipitation mass export cannot be negative')
    end subroutine test_invalid_domains_fail_closed


    subroutine test_nonfinite_inputs_fail_closed()
        real(dp), allocatable :: vapor(:), condensate(:), energy(:)
        type(microphysics_column_budget_rate) :: budget
        real(dp) :: nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        call evaluate_microphysics_phase_change_budget( &
            [nan_value], [0.0_dp], [0.0_dp], 2.5e6_dp, &
            vapor, condensate, energy, budget, ierr)
        call require(ierr == MICROPHYSICS_BUDGET_ERR_NONFINITE, &
            'nonfinite phase rate rejected')
        call require(size(vapor) == 0, 'nonfinite failure emits no tendency')
    end subroutine test_nonfinite_inputs_fail_closed


    subroutine assert_array_close(actual, expected, tolerance, label)
        real(dp), intent(in) :: actual(:), expected(:), tolerance
        character(len=*), intent(in) :: label
        if (size(actual) /= size(expected) .or. any(abs(actual - expected) > tolerance)) then
            write(error_unit, '(A)') trim(label)
            error stop 1
        end if
    end subroutine assert_array_close


    subroutine assert_close(actual, expected, tolerance, label)
        real(dp), intent(in) :: actual, expected, tolerance
        character(len=*), intent(in) :: label
        if (abs(actual - expected) > tolerance) then
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

end program test_microphysics_phase_change_budget
