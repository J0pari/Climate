module climate_time_integration
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: TIME_OK = 0
    integer, parameter, public :: TIME_ERR_EMPTY_STATE = 1
    integer, parameter, public :: TIME_ERR_INVALID_STEP = 2
    integer, parameter, public :: TIME_ERR_INVALID_STEPS = 3
    integer, parameter, public :: TIME_ERR_RHS = 4
    integer, parameter, public :: TIME_ERR_NONFINITE = 5

    abstract interface
        subroutine tendency_function(time, state, tendency, ierr)
            import :: dp
            real(dp), intent(in) :: time
            real(dp), intent(in) :: state(:)
            real(dp), intent(out) :: tendency(:)
            integer, intent(out) :: ierr
        end subroutine tendency_function
    end interface

    public :: rk4_step
    public :: integrate_rk4

contains

    subroutine rk4_step(rhs, time, state, dt, next_state, ierr)
        procedure(tendency_function) :: rhs
        real(dp), intent(in) :: time
        real(dp), intent(in) :: state(:)
        real(dp), intent(in) :: dt
        real(dp), allocatable, intent(out) :: next_state(:)
        integer, intent(out) :: ierr

        real(dp) :: k1(size(state)), k2(size(state))
        real(dp) :: k3(size(state)), k4(size(state))
        real(dp) :: stage(size(state))
        integer :: rhs_ierr

        ierr = TIME_OK

        if (size(state) == 0) then
            ierr = TIME_ERR_EMPTY_STATE
            allocate(next_state(0))
            return
        end if

        if (.not. ieee_is_finite(time) .or. .not. ieee_is_finite(dt) .or. dt == 0.0_dp) then
            ierr = TIME_ERR_INVALID_STEP
            allocate(next_state(0))
            return
        end if

        if (.not. all(ieee_is_finite(state))) then
            ierr = TIME_ERR_NONFINITE
            allocate(next_state(0))
            return
        end if

        call rhs(time, state, k1, rhs_ierr)
        if (rhs_ierr /= 0) then
            ierr = TIME_ERR_RHS
            allocate(next_state(0))
            return
        end if
        if (.not. all(ieee_is_finite(k1))) then
            ierr = TIME_ERR_NONFINITE
            allocate(next_state(0))
            return
        end if

        stage = state + 0.5_dp * dt * k1
        if (.not. all(ieee_is_finite(stage))) then
            ierr = TIME_ERR_NONFINITE
            allocate(next_state(0))
            return
        end if
        call rhs(time + 0.5_dp * dt, stage, k2, rhs_ierr)
        if (rhs_ierr /= 0) then
            ierr = TIME_ERR_RHS
            allocate(next_state(0))
            return
        end if
        if (.not. all(ieee_is_finite(k2))) then
            ierr = TIME_ERR_NONFINITE
            allocate(next_state(0))
            return
        end if

        stage = state + 0.5_dp * dt * k2
        if (.not. all(ieee_is_finite(stage))) then
            ierr = TIME_ERR_NONFINITE
            allocate(next_state(0))
            return
        end if
        call rhs(time + 0.5_dp * dt, stage, k3, rhs_ierr)
        if (rhs_ierr /= 0) then
            ierr = TIME_ERR_RHS
            allocate(next_state(0))
            return
        end if
        if (.not. all(ieee_is_finite(k3))) then
            ierr = TIME_ERR_NONFINITE
            allocate(next_state(0))
            return
        end if

        stage = state + dt * k3
        if (.not. all(ieee_is_finite(stage))) then
            ierr = TIME_ERR_NONFINITE
            allocate(next_state(0))
            return
        end if
        call rhs(time + dt, stage, k4, rhs_ierr)
        if (rhs_ierr /= 0) then
            ierr = TIME_ERR_RHS
            allocate(next_state(0))
            return
        end if
        if (.not. all(ieee_is_finite(k4))) then
            ierr = TIME_ERR_NONFINITE
            allocate(next_state(0))
            return
        end if

        allocate(next_state(size(state)))
        next_state = state + (dt / 6.0_dp) * (k1 + 2.0_dp * k2 + 2.0_dp * k3 + k4)
        if (.not. all(ieee_is_finite(next_state))) then
            deallocate(next_state)
            allocate(next_state(0))
            ierr = TIME_ERR_NONFINITE
        end if
    end subroutine rk4_step


    subroutine integrate_rk4(rhs, initial_time, initial_state, dt, steps, final_state, ierr)
        procedure(tendency_function) :: rhs
        real(dp), intent(in) :: initial_time
        real(dp), intent(in) :: initial_state(:)
        real(dp), intent(in) :: dt
        integer, intent(in) :: steps
        real(dp), allocatable, intent(out) :: final_state(:)
        integer, intent(out) :: ierr

        real(dp), allocatable :: current(:), next_state(:)
        real(dp) :: time
        integer :: step

        ierr = TIME_OK

        if (steps <= 0) then
            ierr = TIME_ERR_INVALID_STEPS
            allocate(final_state(0))
            return
        end if

        allocate(current(size(initial_state)))
        current = initial_state
        time = initial_time

        do step = 1, steps
            call rk4_step(rhs, time, current, dt, next_state, ierr)
            if (ierr /= TIME_OK) then
                allocate(final_state(0))
                return
            end if
            call move_alloc(next_state, current)
            time = time + dt
        end do

        call move_alloc(current, final_state)
    end subroutine integrate_rk4

end module climate_time_integration
