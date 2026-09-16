module climate_linear_implicit
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: IMPLICIT_OK = 0
    integer, parameter, public :: IMPLICIT_ERR_EMPTY = 1
    integer, parameter, public :: IMPLICIT_ERR_DIMENSION = 2
    integer, parameter, public :: IMPLICIT_ERR_TOLERANCE = 3
    integer, parameter, public :: IMPLICIT_ERR_PIVOT = 4
    integer, parameter, public :: IMPLICIT_ERR_NONFINITE = 5
    integer, parameter, public :: IMPLICIT_ERR_PARAMETER = 6
    integer, parameter, public :: IMPLICIT_ERR_BAND_STRUCTURE = 7
    integer, parameter, public :: IMPLICIT_ERR_UNFACTORED = 8

    type, public :: linear_solve_diagnostics
        real(dp) :: matrix_inf_norm = 0.0_dp
        real(dp) :: min_abs_pivot = 0.0_dp
        real(dp) :: min_scaled_pivot = 0.0_dp
        real(dp) :: residual_inf_norm = 0.0_dp
    end type linear_solve_diagnostics

    type, public :: tridiagonal_factorization
        private
        integer :: n = 0
        real(dp) :: matrix_inf_norm = 0.0_dp
        real(dp) :: min_abs_pivot = 0.0_dp
        real(dp) :: min_scaled_pivot = 0.0_dp
        real(dp), allocatable :: lower(:)
        real(dp), allocatable :: diagonal(:)
        real(dp), allocatable :: upper(:)
        real(dp), allocatable :: multipliers(:)
        real(dp), allocatable :: pivots(:)
    end type tridiagonal_factorization

    public :: factor_tridiagonal
    public :: solve_factored_tridiagonal
    public :: solve_tridiagonal
    public :: theta_tridiagonal_step

contains

    subroutine reset_factorization(factor)
        type(tridiagonal_factorization), intent(inout) :: factor

        factor%n = 0
        factor%matrix_inf_norm = 0.0_dp
        factor%min_abs_pivot = 0.0_dp
        factor%min_scaled_pivot = 0.0_dp
        if (allocated(factor%lower)) deallocate(factor%lower)
        if (allocated(factor%diagonal)) deallocate(factor%diagonal)
        if (allocated(factor%upper)) deallocate(factor%upper)
        if (allocated(factor%multipliers)) deallocate(factor%multipliers)
        if (allocated(factor%pivots)) deallocate(factor%pivots)
    end subroutine reset_factorization


    subroutine empty_solution(solution)
        real(dp), allocatable, intent(out) :: solution(:)
        allocate(solution(0))
    end subroutine empty_solution


    subroutine zero_diagnostics(diagnostics)
        type(linear_solve_diagnostics), intent(out) :: diagnostics
        diagnostics%matrix_inf_norm = 0.0_dp
        diagnostics%min_abs_pivot = 0.0_dp
        diagnostics%min_scaled_pivot = 0.0_dp
        diagnostics%residual_inf_norm = 0.0_dp
    end subroutine zero_diagnostics


    subroutine factor_tridiagonal(lower, diagonal, upper, abs_pivot_tolerance, rel_pivot_tolerance, &
                                  factor, ierr, diagnostics)
        real(dp), intent(in) :: lower(:), diagonal(:), upper(:)
        real(dp), intent(in) :: abs_pivot_tolerance, rel_pivot_tolerance
        type(tridiagonal_factorization), intent(inout) :: factor
        integer, intent(out) :: ierr
        type(linear_solve_diagnostics), intent(out) :: diagnostics

        real(dp) :: pivot_threshold, row_norm
        integer :: i, n

        call reset_factorization(factor)
        call zero_diagnostics(diagnostics)
        ierr = IMPLICIT_OK
        n = size(diagonal)

        if (n == 0) then
            ierr = IMPLICIT_ERR_EMPTY
            return
        end if
        if (size(lower) /= n .or. size(upper) /= n) then
            ierr = IMPLICIT_ERR_DIMENSION
            return
        end if
        if (.not. ieee_is_finite(abs_pivot_tolerance) .or. abs_pivot_tolerance < 0.0_dp .or. &
            .not. ieee_is_finite(rel_pivot_tolerance) .or. rel_pivot_tolerance < 0.0_dp) then
            ierr = IMPLICIT_ERR_TOLERANCE
            return
        end if
        if (.not. all(ieee_is_finite(lower)) .or. .not. all(ieee_is_finite(diagonal)) .or. &
            .not. all(ieee_is_finite(upper))) then
            ierr = IMPLICIT_ERR_NONFINITE
            return
        end if
        if (lower(1) /= 0.0_dp .or. upper(n) /= 0.0_dp) then
            ierr = IMPLICIT_ERR_BAND_STRUCTURE
            return
        end if

        do i = 1, n
            row_norm = abs(diagonal(i))
            if (i > 1) row_norm = row_norm + abs(lower(i))
            if (i < n) row_norm = row_norm + abs(upper(i))
            diagnostics%matrix_inf_norm = max(diagnostics%matrix_inf_norm, row_norm)
        end do
        pivot_threshold = abs_pivot_tolerance + rel_pivot_tolerance * diagnostics%matrix_inf_norm

        allocate(factor%lower(n), factor%diagonal(n), factor%upper(n), &
                 factor%multipliers(n), factor%pivots(n))
        factor%lower = lower
        factor%diagonal = diagonal
        factor%upper = upper
        factor%multipliers = 0.0_dp
        factor%pivots = 0.0_dp
        factor%n = n
        factor%matrix_inf_norm = diagnostics%matrix_inf_norm

        factor%pivots(1) = diagonal(1)
        diagnostics%min_abs_pivot = abs(factor%pivots(1))
        if (abs(factor%pivots(1)) <= pivot_threshold) then
            ierr = IMPLICIT_ERR_PIVOT
            call reset_factorization(factor)
            return
        end if

        do i = 2, n
            factor%multipliers(i) = lower(i) / factor%pivots(i - 1)
            factor%pivots(i) = diagonal(i) - factor%multipliers(i) * upper(i - 1)
            diagnostics%min_abs_pivot = min(diagnostics%min_abs_pivot, abs(factor%pivots(i)))
            if (abs(factor%pivots(i)) <= pivot_threshold) then
                ierr = IMPLICIT_ERR_PIVOT
                call reset_factorization(factor)
                return
            end if
        end do

        factor%min_abs_pivot = diagnostics%min_abs_pivot
        if (diagnostics%matrix_inf_norm > 0.0_dp) then
            diagnostics%min_scaled_pivot = diagnostics%min_abs_pivot / diagnostics%matrix_inf_norm
        else
            diagnostics%min_scaled_pivot = 0.0_dp
        end if
        factor%min_scaled_pivot = diagnostics%min_scaled_pivot
    end subroutine factor_tridiagonal


    subroutine solve_factored_tridiagonal(factor, rhs, solution, ierr, diagnostics)
        type(tridiagonal_factorization), intent(in) :: factor
        real(dp), intent(in) :: rhs(:)
        real(dp), allocatable, intent(out) :: solution(:)
        integer, intent(out) :: ierr
        type(linear_solve_diagnostics), intent(out) :: diagnostics

        real(dp), allocatable :: transformed_rhs(:), residual(:)
        integer :: i, n

        call zero_diagnostics(diagnostics)
        ierr = IMPLICIT_OK
        n = factor%n

        if (n <= 0 .or. .not. allocated(factor%pivots)) then
            ierr = IMPLICIT_ERR_UNFACTORED
            call empty_solution(solution)
            return
        end if
        if (size(rhs) /= n) then
            ierr = IMPLICIT_ERR_DIMENSION
            call empty_solution(solution)
            return
        end if
        if (.not. all(ieee_is_finite(rhs))) then
            ierr = IMPLICIT_ERR_NONFINITE
            call empty_solution(solution)
            return
        end if

        diagnostics%matrix_inf_norm = factor%matrix_inf_norm
        diagnostics%min_abs_pivot = factor%min_abs_pivot
        diagnostics%min_scaled_pivot = factor%min_scaled_pivot

        allocate(transformed_rhs(n))
        transformed_rhs = rhs
        do i = 2, n
            transformed_rhs(i) = transformed_rhs(i) - factor%multipliers(i) * transformed_rhs(i - 1)
        end do

        allocate(solution(n))
        solution(n) = transformed_rhs(n) / factor%pivots(n)
        do i = n - 1, 1, -1
            solution(i) = (transformed_rhs(i) - factor%upper(i) * solution(i + 1)) / factor%pivots(i)
        end do

        if (.not. all(ieee_is_finite(solution))) then
            deallocate(solution)
            call empty_solution(solution)
            ierr = IMPLICIT_ERR_NONFINITE
            return
        end if

        allocate(residual(n))
        residual = factor%diagonal * solution - rhs
        do i = 2, n
            residual(i) = residual(i) + factor%lower(i) * solution(i - 1)
        end do
        do i = 1, n - 1
            residual(i) = residual(i) + factor%upper(i) * solution(i + 1)
        end do
        diagnostics%residual_inf_norm = maxval(abs(residual))
    end subroutine solve_factored_tridiagonal


    subroutine solve_tridiagonal(lower, diagonal, upper, rhs, abs_pivot_tolerance, rel_pivot_tolerance, &
                                 solution, ierr, diagnostics)
        real(dp), intent(in) :: lower(:), diagonal(:), upper(:), rhs(:)
        real(dp), intent(in) :: abs_pivot_tolerance, rel_pivot_tolerance
        real(dp), allocatable, intent(out) :: solution(:)
        integer, intent(out) :: ierr
        type(linear_solve_diagnostics), intent(out) :: diagnostics

        type(tridiagonal_factorization) :: factor
        type(linear_solve_diagnostics) :: factor_diagnostics

        call factor_tridiagonal(lower, diagonal, upper, abs_pivot_tolerance, rel_pivot_tolerance, &
                                factor, ierr, factor_diagnostics)
        if (ierr /= IMPLICIT_OK) then
            diagnostics = factor_diagnostics
            call empty_solution(solution)
            return
        end if

        call solve_factored_tridiagonal(factor, rhs, solution, ierr, diagnostics)
    end subroutine solve_tridiagonal


    subroutine theta_tridiagonal_step(lower, diagonal, upper, state, explicit_source, dt, theta, &
                                      abs_pivot_tolerance, rel_pivot_tolerance, next_state, ierr, diagnostics)
        real(dp), intent(in) :: lower(:), diagonal(:), upper(:)
        real(dp), intent(in) :: state(:), explicit_source(:)
        real(dp), intent(in) :: dt, theta, abs_pivot_tolerance, rel_pivot_tolerance
        real(dp), allocatable, intent(out) :: next_state(:)
        integer, intent(out) :: ierr
        type(linear_solve_diagnostics), intent(out) :: diagnostics

        real(dp), allocatable :: lhs_lower(:), lhs_diagonal(:), lhs_upper(:), step_rhs(:)
        real(dp), allocatable :: operator_state(:)
        integer :: i, n

        call zero_diagnostics(diagnostics)
        ierr = IMPLICIT_OK
        n = size(state)

        if (n == 0) then
            ierr = IMPLICIT_ERR_EMPTY
            call empty_solution(next_state)
            return
        end if
        if (size(explicit_source) /= n .or. size(lower) /= n .or. size(diagonal) /= n .or. size(upper) /= n) then
            ierr = IMPLICIT_ERR_DIMENSION
            call empty_solution(next_state)
            return
        end if
        if (.not. ieee_is_finite(dt) .or. dt <= 0.0_dp .or. .not. ieee_is_finite(theta) .or. &
            theta < 0.0_dp .or. theta > 1.0_dp) then
            ierr = IMPLICIT_ERR_PARAMETER
            call empty_solution(next_state)
            return
        end if
        if (.not. ieee_is_finite(abs_pivot_tolerance) .or. abs_pivot_tolerance < 0.0_dp .or. &
            .not. ieee_is_finite(rel_pivot_tolerance) .or. rel_pivot_tolerance < 0.0_dp) then
            ierr = IMPLICIT_ERR_TOLERANCE
            call empty_solution(next_state)
            return
        end if
        if (lower(1) /= 0.0_dp .or. upper(n) /= 0.0_dp) then
            ierr = IMPLICIT_ERR_BAND_STRUCTURE
            call empty_solution(next_state)
            return
        end if
        if (.not. all(ieee_is_finite(lower)) .or. .not. all(ieee_is_finite(diagonal)) .or. &
            .not. all(ieee_is_finite(upper)) .or. .not. all(ieee_is_finite(state)) .or. &
            .not. all(ieee_is_finite(explicit_source))) then
            ierr = IMPLICIT_ERR_NONFINITE
            call empty_solution(next_state)
            return
        end if

        allocate(operator_state(n), lhs_lower(n), lhs_diagonal(n), lhs_upper(n), step_rhs(n))
        operator_state = diagonal * state
        do i = 2, n
            operator_state(i) = operator_state(i) + lower(i) * state(i - 1)
        end do
        do i = 1, n - 1
            operator_state(i) = operator_state(i) + upper(i) * state(i + 1)
        end do

        lhs_lower = -theta * dt * lower
        lhs_diagonal = 1.0_dp - theta * dt * diagonal
        lhs_upper = -theta * dt * upper

        ! explicit_source is an additive tendency evaluated by the caller at the
        ! beginning of the step. It is intentionally not folded into the linear
        ! operator or silently re-evaluated inside this numerical kernel.
        step_rhs = state + dt * ((1.0_dp - theta) * operator_state + explicit_source)

        if (.not. all(ieee_is_finite(lhs_lower)) .or. .not. all(ieee_is_finite(lhs_diagonal)) .or. &
            .not. all(ieee_is_finite(lhs_upper)) .or. .not. all(ieee_is_finite(step_rhs))) then
            ierr = IMPLICIT_ERR_NONFINITE
            call empty_solution(next_state)
            return
        end if

        call solve_tridiagonal(lhs_lower, lhs_diagonal, lhs_upper, step_rhs, &
                               abs_pivot_tolerance, rel_pivot_tolerance, next_state, ierr, diagnostics)
    end subroutine theta_tridiagonal_step

end module climate_linear_implicit
