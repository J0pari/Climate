program test_convective_column_redistribution
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_convective_column_redistribution, only: &
        convective_column_budget_rate, evaluate_convective_redistribution, &
        CONVECTIVE_REDISTRIBUTION_OK, CONVECTIVE_REDISTRIBUTION_ERR_EMPTY, &
        CONVECTIVE_REDISTRIBUTION_ERR_DIMENSION, &
        CONVECTIVE_REDISTRIBUTION_ERR_MASS, CONVECTIVE_REDISTRIBUTION_ERR_NONFINITE
    implicit none

    integer, parameter :: dp = real64

    call test_closed_column_redistributes_without_net_budget_change()
    call test_open_boundary_fluxes_match_column_tendencies()
    call test_upward_flux_sign_is_explicit()
    call test_invalid_shapes_and_mass_fail_closed()
    call test_nonfinite_flux_fails_closed()

contains

    subroutine test_closed_column_redistributes_without_net_budget_change()
        real(dp), allocatable :: energy_tendency(:), water_tendency(:)
        real(dp), allocatable :: momentum_x_tendency(:), momentum_y_tendency(:)
        type(convective_column_budget_rate) :: budget
        integer :: ierr

        call evaluate_convective_redistribution( &
            [100.0_dp, 200.0_dp, 300.0_dp], &
            [0.0_dp, 6.0_dp, -2.0_dp, 0.0_dp], &
            [0.0_dp, 3.0e-4_dp, 1.0e-4_dp, 0.0_dp], &
            [0.0_dp, 0.4_dp, -0.1_dp, 0.0_dp], &
            [0.0_dp, -0.2_dp, 0.3_dp, 0.0_dp], &
            energy_tendency, water_tendency, momentum_x_tendency, &
            momentum_y_tendency, budget, ierr)

        call require(ierr == CONVECTIVE_REDISTRIBUTION_OK, &
            'closed convective redistribution succeeds')
        call assert_close(budget%column_energy_tendency_w_m2, 0.0_dp, 1.0e-14_dp, &
            'closed energy column conserves')
        call assert_close(budget%column_water_tendency_kg_m2_s, 0.0_dp, 1.0e-14_dp, &
            'closed water column conserves')
        call assert_close(budget%column_momentum_x_tendency_n_m2, 0.0_dp, 1.0e-14_dp, &
            'closed x momentum column conserves')
        call assert_close(budget%column_momentum_y_tendency_n_m2, 0.0_dp, 1.0e-14_dp, &
            'closed y momentum column conserves')
        call require(any(abs(energy_tendency) > 0.0_dp), &
            'closed column can redistribute energy internally')
        call require(any(abs(water_tendency) > 0.0_dp), &
            'closed column can redistribute water internally')
    end subroutine test_closed_column_redistributes_without_net_budget_change


    subroutine test_open_boundary_fluxes_match_column_tendencies()
        real(dp), allocatable :: energy_tendency(:), water_tendency(:)
        real(dp), allocatable :: momentum_x_tendency(:), momentum_y_tendency(:)
        type(convective_column_budget_rate) :: budget
        integer :: ierr

        call evaluate_convective_redistribution( &
            [100.0_dp, 200.0_dp], &
            [8.0_dp, 3.0_dp, 1.5_dp], &
            [5.0e-4_dp, 2.0e-4_dp, 1.0e-4_dp], &
            [0.5_dp, 0.1_dp, -0.2_dp], &
            [-0.3_dp, 0.2_dp, 0.1_dp], &
            energy_tendency, water_tendency, momentum_x_tendency, &
            momentum_y_tendency, budget, ierr)

        call require(ierr == CONVECTIVE_REDISTRIBUTION_OK, &
            'open convective redistribution succeeds')
        call assert_close(budget%column_energy_tendency_w_m2, 6.5_dp, 1.0e-14_dp, &
            'energy boundary budget')
        call assert_close(budget%column_water_tendency_kg_m2_s, 4.0e-4_dp, 1.0e-14_dp, &
            'water boundary budget')
        call assert_close(budget%column_momentum_x_tendency_n_m2, 0.7_dp, 1.0e-14_dp, &
            'x momentum boundary budget')
        call assert_close(budget%column_momentum_y_tendency_n_m2, -0.4_dp, 1.0e-14_dp, &
            'y momentum boundary budget')
        call assert_close(budget%energy_closure_residual_w_m2, 0.0_dp, 1.0e-14_dp, &
            'energy budget closes')
        call assert_close(budget%water_closure_residual_kg_m2_s, 0.0_dp, 1.0e-14_dp, &
            'water budget closes')
        call assert_close(budget%momentum_x_closure_residual_n_m2, 0.0_dp, 1.0e-14_dp, &
            'x momentum budget closes')
        call assert_close(budget%momentum_y_closure_residual_n_m2, 0.0_dp, 1.0e-14_dp, &
            'y momentum budget closes')
    end subroutine test_open_boundary_fluxes_match_column_tendencies


    subroutine test_upward_flux_sign_is_explicit()
        real(dp), allocatable :: energy_tendency(:), water_tendency(:)
        real(dp), allocatable :: momentum_x_tendency(:), momentum_y_tendency(:)
        type(convective_column_budget_rate) :: budget
        integer :: ierr

        call evaluate_convective_redistribution( &
            [100.0_dp, 100.0_dp], &
            [0.0_dp, 10.0_dp, 0.0_dp], &
            [0.0_dp, 0.0_dp, 0.0_dp], &
            [0.0_dp, 0.0_dp, 0.0_dp], &
            [0.0_dp, 0.0_dp, 0.0_dp], &
            energy_tendency, water_tendency, momentum_x_tendency, &
            momentum_y_tendency, budget, ierr)
        call require(ierr == CONVECTIVE_REDISTRIBUTION_OK, 'sign witness succeeds')
        call assert_close(energy_tendency(1), -0.1_dp, 1.0e-14_dp, &
            'upward interface flux cools lower layer')
        call assert_close(energy_tendency(2), 0.1_dp, 1.0e-14_dp, &
            'same upward flux warms upper layer')
    end subroutine test_upward_flux_sign_is_explicit


    subroutine test_invalid_shapes_and_mass_fail_closed()
        real(dp), allocatable :: energy_tendency(:), water_tendency(:)
        real(dp), allocatable :: momentum_x_tendency(:), momentum_y_tendency(:)
        type(convective_column_budget_rate) :: budget
        integer :: ierr

        call evaluate_convective_redistribution( &
            [real(dp) ::], [0.0_dp], [0.0_dp], [0.0_dp], [0.0_dp], &
            energy_tendency, water_tendency, momentum_x_tendency, &
            momentum_y_tendency, budget, ierr)
        call require(ierr == CONVECTIVE_REDISTRIBUTION_ERR_EMPTY, &
            'empty convective column rejected')
        call require(size(energy_tendency) == 0, 'empty failure emits no tendency')

        call evaluate_convective_redistribution( &
            [100.0_dp], [0.0_dp], [0.0_dp, 0.0_dp], [0.0_dp, 0.0_dp], &
            [0.0_dp, 0.0_dp], energy_tendency, water_tendency, &
            momentum_x_tendency, momentum_y_tendency, budget, ierr)
        call require(ierr == CONVECTIVE_REDISTRIBUTION_ERR_DIMENSION, &
            'interface dimensions must be n+1')

        call evaluate_convective_redistribution( &
            [0.0_dp], [0.0_dp, 0.0_dp], [0.0_dp, 0.0_dp], &
            [0.0_dp, 0.0_dp], [0.0_dp, 0.0_dp], &
            energy_tendency, water_tendency, momentum_x_tendency, &
            momentum_y_tendency, budget, ierr)
        call require(ierr == CONVECTIVE_REDISTRIBUTION_ERR_MASS, &
            'layer mass must be finite and positive')
    end subroutine test_invalid_shapes_and_mass_fail_closed


    subroutine test_nonfinite_flux_fails_closed()
        real(dp), allocatable :: energy_tendency(:), water_tendency(:)
        real(dp), allocatable :: momentum_x_tendency(:), momentum_y_tendency(:)
        type(convective_column_budget_rate) :: budget
        real(dp) :: nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        call evaluate_convective_redistribution( &
            [100.0_dp], [0.0_dp, nan_value], [0.0_dp, 0.0_dp], &
            [0.0_dp, 0.0_dp], [0.0_dp, 0.0_dp], &
            energy_tendency, water_tendency, momentum_x_tendency, &
            momentum_y_tendency, budget, ierr)
        call require(ierr == CONVECTIVE_REDISTRIBUTION_ERR_NONFINITE, &
            'nonfinite convective flux rejected')
        call require(size(energy_tendency) == 0, 'nonfinite failure emits no tendency')
    end subroutine test_nonfinite_flux_fails_closed


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

end program test_convective_column_redistribution
