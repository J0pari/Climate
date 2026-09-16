module climate_lapack_tridiagonal
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: LAPACK_TRI_OK = 0
    integer, parameter, public :: LAPACK_TRI_ERR_EMPTY = 1
    integer, parameter, public :: LAPACK_TRI_ERR_DIMENSION = 2
    integer, parameter, public :: LAPACK_TRI_ERR_NONFINITE = 3
    integer, parameter, public :: LAPACK_TRI_ERR_BAND_STRUCTURE = 4
    integer, parameter, public :: LAPACK_TRI_ERR_SINGULAR = 5
    integer, parameter, public :: LAPACK_TRI_ERR_UNFACTORED = 6
    integer, parameter, public :: LAPACK_TRI_ERR_LAPACK = 7

    type, public :: lapack_tridiagonal_diagnostics
        real(dp) :: matrix_inf_norm = 0.0_dp
        real(dp) :: min_abs_u_diagonal = 0.0_dp
        real(dp) :: residual_inf_norm = 0.0_dp
        integer :: lapack_info = 0
        logical :: pivoting_used = .false.
    end type lapack_tridiagonal_diagnostics

    type, public :: lapack_tridiagonal_factorization
        private
        integer :: n = 0
        real(dp), allocatable :: original_lower(:)
        real(dp), allocatable :: original_diagonal(:)
        real(dp), allocatable :: original_upper(:)
        real(dp), allocatable :: dl(:)
        real(dp), allocatable :: d(:)
        real(dp), allocatable :: du(:)
        real(dp), allocatable :: du2(:)
        integer, allocatable :: ipiv(:)
    end type lapack_tridiagonal_factorization

    interface
        subroutine dgttrf(n, dl, d, du, du2, ipiv, info)
            import :: dp
            integer, intent(in) :: n
            real(dp), intent(inout) :: dl(*), d(*), du(*), du2(*)
            integer, intent(out) :: ipiv(*)
            integer, intent(out) :: info
        end subroutine dgttrf

        subroutine dgttrs(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info)
            import :: dp
            character(len=1), intent(in) :: trans
            integer, intent(in) :: n, nrhs, ldb
            real(dp), intent(in) :: dl(*), d(*), du(*), du2(*)
            integer, intent(in) :: ipiv(*)
            real(dp), intent(inout) :: b(ldb, *)
            integer, intent(out) :: info
        end subroutine dgttrs
    end interface

    public :: factor_tridiagonal_lapack
    public :: solve_factored_tridiagonal_lapack
    public :: solve_tridiagonal_lapack

contains

    subroutine zero_diagnostics(diagnostics)
        type(lapack_tridiagonal_diagnostics), intent(out) :: diagnostics

        diagnostics%matrix_inf_norm = 0.0_dp
        diagnostics%min_abs_u_diagonal = 0.0_dp
        diagnostics%residual_inf_norm = 0.0_dp
        diagnostics%lapack_info = 0
        diagnostics%pivoting_used = .false.
    end subroutine zero_diagnostics


    subroutine empty_solution(solution)
        real(dp), allocatable, intent(out) :: solution(:)
        allocate(solution(0))
    end subroutine empty_solution


    subroutine reset_factorization(factor)
        type(lapack_tridiagonal_factorization), intent(inout) :: factor

        factor%n = 0
        if (allocated(factor%original_lower)) deallocate(factor%original_lower)
        if (allocated(factor%original_diagonal)) deallocate(factor%original_diagonal)
        if (allocated(factor%original_upper)) deallocate(factor%original_upper)
        if (allocated(factor%dl)) deallocate(factor%dl)
        if (allocated(factor%d)) deallocate(factor%d)
        if (allocated(factor%du)) deallocate(factor%du)
        if (allocated(factor%du2)) deallocate(factor%du2)
        if (allocated(factor%ipiv)) deallocate(factor%ipiv)
    end subroutine reset_factorization


    subroutine factor_tridiagonal_lapack(lower, diagonal, upper, factor, ierr, diagnostics)
        real(dp), intent(in) :: lower(:), diagonal(:), upper(:)
        type(lapack_tridiagonal_factorization), intent(inout) :: factor
        integer, intent(out) :: ierr
        type(lapack_tridiagonal_diagnostics), intent(out) :: diagnostics

        real(dp) :: row_norm
        integer :: i, info, n

        call reset_factorization(factor)
        call zero_diagnostics(diagnostics)
        ierr = LAPACK_TRI_OK
        n = size(diagonal)

        if (n == 0) then
            ierr = LAPACK_TRI_ERR_EMPTY
            return
        end if
        if (size(lower) /= n .or. size(upper) /= n) then
            ierr = LAPACK_TRI_ERR_DIMENSION
            return
        end if
        if (.not. all(ieee_is_finite(lower)) .or. .not. all(ieee_is_finite(diagonal)) .or. &
            .not. all(ieee_is_finite(upper))) then
            ierr = LAPACK_TRI_ERR_NONFINITE
            return
        end if
        if (lower(1) /= 0.0_dp .or. upper(n) /= 0.0_dp) then
            ierr = LAPACK_TRI_ERR_BAND_STRUCTURE
            return
        end if

        do i = 1, n
            row_norm = abs(diagonal(i))
            if (i > 1) row_norm = row_norm + abs(lower(i))
            if (i < n) row_norm = row_norm + abs(upper(i))
            diagnostics%matrix_inf_norm = max(diagnostics%matrix_inf_norm, row_norm)
        end do

        factor%n = n
        allocate(factor%original_lower(n), factor%original_diagonal(n), factor%original_upper(n))
        factor%original_lower = lower
        factor%original_diagonal = diagonal
        factor%original_upper = upper

        allocate(factor%d(n), factor%ipiv(n))
        factor%d = diagonal
        factor%ipiv = 0

        if (n == 1) then
            if (diagonal(1) == 0.0_dp) then
                diagnostics%lapack_info = 1
                ierr = LAPACK_TRI_ERR_SINGULAR
                call reset_factorization(factor)
                return
            end if
            allocate(factor%dl(0), factor%du(0), factor%du2(0))
            factor%ipiv(1) = 1
            diagnostics%min_abs_u_diagonal = abs(diagonal(1))
            return
        end if

        allocate(factor%dl(n - 1), factor%du(n - 1), factor%du2(max(1, n - 2)))
        factor%dl = lower(2:n)
        factor%du = upper(1:n - 1)
        factor%du2 = 0.0_dp

        call dgttrf(n, factor%dl, factor%d, factor%du, factor%du2, factor%ipiv, info)
        diagnostics%lapack_info = info
        if (info > 0) then
            ierr = LAPACK_TRI_ERR_SINGULAR
            call reset_factorization(factor)
            return
        else if (info < 0) then
            ierr = LAPACK_TRI_ERR_LAPACK
            call reset_factorization(factor)
            return
        end if

        diagnostics%min_abs_u_diagonal = minval(abs(factor%d))
        diagnostics%pivoting_used = any(factor%ipiv /= [(i, i = 1, n)])
    end subroutine factor_tridiagonal_lapack


    subroutine solve_factored_tridiagonal_lapack(factor, rhs, solution, ierr, diagnostics)
        type(lapack_tridiagonal_factorization), intent(in) :: factor
        real(dp), intent(in) :: rhs(:)
        real(dp), allocatable, intent(out) :: solution(:)
        integer, intent(out) :: ierr
        type(lapack_tridiagonal_diagnostics), intent(out) :: diagnostics

        real(dp), allocatable :: work(:,:), residual(:)
        real(dp) :: row_norm
        integer :: i, info, n

        call zero_diagnostics(diagnostics)
        ierr = LAPACK_TRI_OK
        n = factor%n

        if (n <= 0 .or. .not. allocated(factor%d)) then
            ierr = LAPACK_TRI_ERR_UNFACTORED
            call empty_solution(solution)
            return
        end if
        if (size(rhs) /= n) then
            ierr = LAPACK_TRI_ERR_DIMENSION
            call empty_solution(solution)
            return
        end if
        if (.not. all(ieee_is_finite(rhs))) then
            ierr = LAPACK_TRI_ERR_NONFINITE
            call empty_solution(solution)
            return
        end if

        do i = 1, n
            row_norm = abs(factor%original_diagonal(i))
            if (i > 1) row_norm = row_norm + abs(factor%original_lower(i))
            if (i < n) row_norm = row_norm + abs(factor%original_upper(i))
            diagnostics%matrix_inf_norm = max(diagnostics%matrix_inf_norm, row_norm)
        end do
        diagnostics%min_abs_u_diagonal = minval(abs(factor%d))
        diagnostics%pivoting_used = any(factor%ipiv /= [(i, i = 1, n)])

        allocate(work(n, 1))
        work(:, 1) = rhs

        if (n == 1) then
            work(1, 1) = work(1, 1) / factor%d(1)
            info = 0
        else
            call dgttrs('N', n, 1, factor%dl, factor%d, factor%du, factor%du2, &
                        factor%ipiv, work, n, info)
        end if
        diagnostics%lapack_info = info
        if (info /= 0) then
            ierr = LAPACK_TRI_ERR_LAPACK
            call empty_solution(solution)
            return
        end if

        allocate(solution(n))
        solution = work(:, 1)
        if (.not. all(ieee_is_finite(solution))) then
            deallocate(solution)
            call empty_solution(solution)
            ierr = LAPACK_TRI_ERR_NONFINITE
            return
        end if

        allocate(residual(n))
        residual = factor%original_diagonal * solution - rhs
        do i = 2, n
            residual(i) = residual(i) + factor%original_lower(i) * solution(i - 1)
        end do
        do i = 1, n - 1
            residual(i) = residual(i) + factor%original_upper(i) * solution(i + 1)
        end do
        diagnostics%residual_inf_norm = maxval(abs(residual))
    end subroutine solve_factored_tridiagonal_lapack


    subroutine solve_tridiagonal_lapack(lower, diagonal, upper, rhs, solution, ierr, diagnostics)
        real(dp), intent(in) :: lower(:), diagonal(:), upper(:), rhs(:)
        real(dp), allocatable, intent(out) :: solution(:)
        integer, intent(out) :: ierr
        type(lapack_tridiagonal_diagnostics), intent(out) :: diagnostics

        type(lapack_tridiagonal_factorization) :: factor
        type(lapack_tridiagonal_diagnostics) :: factor_diagnostics

        call factor_tridiagonal_lapack(lower, diagonal, upper, factor, ierr, factor_diagnostics)
        if (ierr /= LAPACK_TRI_OK) then
            diagnostics = factor_diagnostics
            call empty_solution(solution)
            return
        end if

        call solve_factored_tridiagonal_lapack(factor, rhs, solution, ierr, diagnostics)
    end subroutine solve_tridiagonal_lapack

end module climate_lapack_tridiagonal
