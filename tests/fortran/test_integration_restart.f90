module integration_restart_test_rhs
    use climate_time_integration, only: dp, TIME_OK
    implicit none
contains

    subroutine time_dependent_rhs(time, state, tendency, ierr)
        real(dp), intent(in) :: time
        real(dp), intent(in) :: state(:)
        real(dp), intent(out) :: tendency(:)
        integer, intent(out) :: ierr

        if (size(state) /= 1 .or. size(tendency) /= 1) then
            ierr = 91
            return
        end if
        tendency(1) = time + 0.5_dp * state(1)
        ierr = TIME_OK
    end subroutine time_dependent_rhs

end module integration_restart_test_rhs


program test_integration_restart
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_time_integration, only: integrate_rk4, TIME_OK
    use climate_integration_restart, only: &
        restart_record, capture_restart, restore_restart, validate_restart, &
        RESTART_OK, RESTART_ERR_NONFINITE, RESTART_ERR_IDENTITY, &
        RESTART_ERR_UNINITIALIZED
    use integration_restart_test_rhs, only: time_dependent_rhs
    implicit none

    integer, parameter :: dp = real64

    call test_split_run_matches_uninterrupted_time_dependent_integration()
    call test_restart_copies_state_and_binds_identities()
    call test_missing_identity_fails_closed()
    call test_nonfinite_state_fails_closed()
    call test_uninitialized_restore_fails_closed()

contains

    subroutine test_split_run_matches_uninterrupted_time_dependent_integration()
        integer, parameter :: first_steps = 4, remaining_steps = 6
        real(dp), parameter :: dt = 0.1_dp
        real(dp) :: initial(1), restart_time, expected_restart_time
        real(dp), allocatable :: uninterrupted(:), first_segment(:)
        real(dp), allocatable :: restored_state(:), restarted(:)
        type(restart_record) :: checkpoint
        integer :: ierr, step

        initial = [1.0_dp]
        call integrate_rk4( &
            time_dependent_rhs, 0.0_dp, initial, dt, first_steps + remaining_steps, &
            uninterrupted, ierr)
        call require(ierr == TIME_OK, 'uninterrupted integration must succeed')

        call integrate_rk4( &
            time_dependent_rhs, 0.0_dp, initial, dt, first_steps, &
            first_segment, ierr)
        call require(ierr == TIME_OK, 'first split segment must succeed')

        expected_restart_time = 0.0_dp
        do step = 1, first_steps
            expected_restart_time = expected_restart_time + dt
        end do
        call capture_restart( &
            expected_restart_time, first_segment, &
            'climate.rk4.fixed-step/v1', 'fixture:restart-config/v1', &
            'none:deterministic', 'fixture:manufactured-time-dependent-ode/v1', &
            checkpoint, ierr)
        call require(ierr == RESTART_OK, 'restart capture must succeed')
        call restore_restart(checkpoint, restart_time, restored_state, ierr)
        call require(ierr == RESTART_OK, 'restart restore must succeed')
        call require(restart_time == expected_restart_time, &
            'restart time must be restored exactly')

        call integrate_rk4( &
            time_dependent_rhs, restart_time, restored_state, dt, remaining_steps, &
            restarted, ierr)
        call require(ierr == TIME_OK, 'restarted integration must succeed')
        call require(size(restarted) == size(uninterrupted), &
            'restarted state shape must match uninterrupted state')
        call require(all(restarted == uninterrupted), &
            'split restart run must be bitwise equal to uninterrupted RK4 run')
    end subroutine test_split_run_matches_uninterrupted_time_dependent_integration


    subroutine test_restart_copies_state_and_binds_identities()
        real(dp) :: state(2)
        type(restart_record) :: checkpoint
        integer :: ierr

        state = [2.0_dp, -1.0_dp]
        call capture_restart( &
            12.5_dp, state, 'solver:v1', 'config:v3', 'rng:seed-42', &
            'provenance:fixture/v1', checkpoint, ierr)
        call require(ierr == RESTART_OK, 'identity-bearing restart must succeed')
        state = 0.0_dp
        call validate_restart(checkpoint, ierr)
        call require(ierr == RESTART_OK, 'captured restart must remain valid')
        call require(all(checkpoint%physical_state == [2.0_dp, -1.0_dp]), &
            'restart state must be an owned copy')
        call require(checkpoint%solver_id == 'solver:v1', 'solver identity must bind')
        call require(checkpoint%configuration_id == 'config:v3', &
            'configuration identity must bind')
        call require(checkpoint%stochastic_state_id == 'rng:seed-42', &
            'stochastic state identity must bind')
        call require(checkpoint%provenance_id == 'provenance:fixture/v1', &
            'provenance identity must bind')
    end subroutine test_restart_copies_state_and_binds_identities


    subroutine test_missing_identity_fails_closed()
        type(restart_record) :: checkpoint
        integer :: ierr

        call capture_restart( &
            0.0_dp, [1.0_dp], '', 'config:v1', 'none:deterministic', &
            'provenance:v1', checkpoint, ierr)
        call require(ierr == RESTART_ERR_IDENTITY, &
            'restart without solver identity must be rejected')
        call require(.not. allocated(checkpoint%physical_state), &
            'failed identity capture must not retain state')
    end subroutine test_missing_identity_fails_closed


    subroutine test_nonfinite_state_fails_closed()
        type(restart_record) :: checkpoint
        real(dp) :: nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        call capture_restart( &
            0.0_dp, [nan_value], 'solver:v1', 'config:v1', 'none:deterministic', &
            'provenance:v1', checkpoint, ierr)
        call require(ierr == RESTART_ERR_NONFINITE, &
            'nonfinite restart state must be rejected')
    end subroutine test_nonfinite_state_fails_closed


    subroutine test_uninitialized_restore_fails_closed()
        type(restart_record) :: checkpoint
        real(dp) :: time
        real(dp), allocatable :: state(:)
        integer :: ierr

        call restore_restart(checkpoint, time, state, ierr)
        call require(ierr == RESTART_ERR_UNINITIALIZED, &
            'uninitialized restart record must be rejected')
        call require(size(state) == 0, &
            'uninitialized restart must not emit physical state')
    end subroutine test_uninitialized_restore_fails_closed


    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message

        if (.not. condition) then
            write(error_unit, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require

end program test_integration_restart
