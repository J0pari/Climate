module time_integration_test_rhs
    use climate_time_integration, only: dp, TIME_OK
    implicit none
contains

    subroutine constant_rhs(time, state, tendency, ierr)
        real(dp), intent(in) :: time
        real(dp), intent(in) :: state(:)
        real(dp), intent(out) :: tendency(:)
        integer, intent(out) :: ierr

        if (size(state) /= 2 .or. size(tendency) /= 2) then
            ierr = 91
            return
        end if
        tendency = [2.0_dp, -3.0_dp]
        ierr = TIME_OK
    end subroutine constant_rhs


    subroutine exponential_rhs(time, state, tendency, ierr)
        real(dp), intent(in) :: time
        real(dp), intent(in) :: state(:)
        real(dp), intent(out) :: tendency(:)
        integer, intent(out) :: ierr

        if (size(state) /= 1 .or. size(tendency) /= 1) then
            ierr = 92
            return
        end if
        tendency(1) = state(1)
        ierr = TIME_OK
    end subroutine exponential_rhs


    subroutine oscillator_rhs(time, state, tendency, ierr)
        real(dp), intent(in) :: time
        real(dp), intent(in) :: state(:)
        real(dp), intent(out) :: tendency(:)
        integer, intent(out) :: ierr

        if (size(state) /= 2 .or. size(tendency) /= 2) then
            ierr = 93
            return
        end if
        tendency(1) = state(2)
        tendency(2) = -state(1)
        ierr = TIME_OK
    end subroutine oscillator_rhs


    subroutine failing_rhs(time, state, tendency, ierr)
        real(dp), intent(in) :: time
        real(dp), intent(in) :: state(:)
        real(dp), intent(out) :: tendency(:)
        integer, intent(out) :: ierr

        tendency = 0.0_dp
        ierr = 99
    end subroutine failing_rhs

end module time_integration_test_rhs


program test_time_integration
    use climate_time_integration, only: dp, TIME_OK, TIME_ERR_INVALID_STEP, &
        TIME_ERR_RHS, rk4_step, integrate_rk4
    use time_integration_test_rhs, only: constant_rhs, exponential_rhs, oscillator_rhs, failing_rhs
    implicit none

    call test_constant_tendency_step()
    call test_fourth_order_convergence()
    call test_coupled_oscillator()
    call test_invalid_step_fails_closed()
    call test_rhs_failure_fails_closed()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_constant_tendency_step()
        real(dp) :: state(2)
        real(dp), allocatable :: result(:)
        integer :: ierr

        state = [1.0_dp, -2.0_dp]
        call rk4_step(constant_rhs, 0.0_dp, state, 0.25_dp, result, ierr)

        call require(ierr == TIME_OK, 'constant-tendency RK4 step must succeed')
        call require(size(result) == 2, 'successful step must preserve state dimension')
        call require(maxval(abs(result - [1.5_dp, -2.75_dp])) < 1.0e-14_dp, &
                     'RK4 must integrate a constant tendency exactly')
    end subroutine test_constant_tendency_step


    subroutine test_fourth_order_convergence()
        real(dp) :: initial(1)
        real(dp), allocatable :: coarse(:), fine(:)
        real(dp) :: exact, coarse_error, fine_error, ratio
        integer :: ierr

        initial = [1.0_dp]
        exact = exp(1.0_dp)

        call integrate_rk4(exponential_rhs, 0.0_dp, initial, 1.0_dp / 8.0_dp, 8, coarse, ierr)
        call require(ierr == TIME_OK, 'coarse exponential integration must succeed')
        call integrate_rk4(exponential_rhs, 0.0_dp, initial, 1.0_dp / 16.0_dp, 16, fine, ierr)
        call require(ierr == TIME_OK, 'fine exponential integration must succeed')

        coarse_error = abs(coarse(1) - exact)
        fine_error = abs(fine(1) - exact)
        ratio = coarse_error / fine_error

        call require(fine_error < coarse_error, 'step halving must reduce RK4 error')
        call require(ratio > 13.0_dp .and. ratio < 19.0_dp, &
                     'step-halving error ratio must witness fourth-order convergence')
    end subroutine test_fourth_order_convergence


    subroutine test_coupled_oscillator()
        integer, parameter :: steps = 400
        real(dp), parameter :: pi = acos(-1.0_dp)
        real(dp) :: initial(2), dt
        real(dp), allocatable :: result(:)
        integer :: ierr

        initial = [1.0_dp, 0.0_dp]
        dt = 2.0_dp * pi / real(steps, dp)
        call integrate_rk4(oscillator_rhs, 0.0_dp, initial, dt, steps, result, ierr)

        call require(ierr == TIME_OK, 'coupled oscillator integration must succeed')
        call require(maxval(abs(result - initial)) < 5.0e-9_dp, &
                     'one oscillator period must return close to the initial state')
    end subroutine test_coupled_oscillator


    subroutine test_invalid_step_fails_closed()
        real(dp) :: state(1)
        real(dp), allocatable :: result(:)
        integer :: ierr

        state = [1.0_dp]
        call rk4_step(exponential_rhs, 0.0_dp, state, 0.0_dp, result, ierr)

        call require(ierr == TIME_ERR_INVALID_STEP, 'zero time step must be rejected')
        call require(size(result) == 0, 'invalid step must not emit a state')
    end subroutine test_invalid_step_fails_closed


    subroutine test_rhs_failure_fails_closed()
        real(dp) :: state(1)
        real(dp), allocatable :: result(:)
        integer :: ierr

        state = [1.0_dp]
        call rk4_step(failing_rhs, 0.0_dp, state, 0.1_dp, result, ierr)

        call require(ierr == TIME_ERR_RHS, 'RHS failure must propagate as an integration failure')
        call require(size(result) == 0, 'RHS failure must not emit a state')
    end subroutine test_rhs_failure_fails_closed

end program test_time_integration
