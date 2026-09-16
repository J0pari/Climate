program test_lapack_tridiagonal
    use climate_lapack_tridiagonal, only: dp, LAPACK_TRI_OK, &
        lapack_tridiagonal_diagnostics, lapack_tridiagonal_factorization, &
        factor_tridiagonal_lapack, solve_factored_tridiagonal_lapack, solve_tridiagonal_lapack
    use climate_linear_implicit, only: IMPLICIT_OK, IMPLICIT_ERR_PIVOT, &
        linear_solve_diagnostics, solve_tridiagonal
    implicit none

    call test_matches_reference_without_pivoting()
    call test_partial_pivoting_solves_reference_failure()
    call test_factorization_reuse()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_matches_reference_without_pivoting()
        real(dp) :: lower(3), diagonal(3), upper(3), rhs(3)
        real(dp), allocatable :: lapack_solution(:), reference_solution(:)
        type(lapack_tridiagonal_diagnostics) :: lapack_diagnostics
        type(linear_solve_diagnostics) :: reference_diagnostics
        integer :: ierr_lapack, ierr_reference

        lower = [0.0_dp, -1.0_dp, -1.0_dp]
        diagonal = [2.0_dp, 2.0_dp, 2.0_dp]
        upper = [-1.0_dp, -1.0_dp, 0.0_dp]
        rhs = [0.0_dp, 0.0_dp, 4.0_dp]

        call solve_tridiagonal_lapack(lower, diagonal, upper, rhs, &
                                      lapack_solution, ierr_lapack, lapack_diagnostics)
        call solve_tridiagonal(lower, diagonal, upper, rhs, 0.0_dp, 1.0e-14_dp, &
                               reference_solution, ierr_reference, reference_diagnostics)

        call require(ierr_lapack == LAPACK_TRI_OK, 'LAPACK tridiagonal solve must succeed')
        call require(ierr_reference == IMPLICIT_OK, 'reference tridiagonal solve must succeed')
        call require(maxval(abs(lapack_solution - reference_solution)) < 1.0e-13_dp, &
                     'LAPACK and reference solves must agree when pivoting is unnecessary')
        call require(.not. lapack_diagnostics%pivoting_used, &
                     'well-behaved reference system should not require LAPACK row interchange')
        call require(lapack_diagnostics%residual_inf_norm < 1.0e-13_dp, &
                     'LAPACK backend must expose a small residual')
    end subroutine test_matches_reference_without_pivoting


    subroutine test_partial_pivoting_solves_reference_failure()
        real(dp) :: lower(2), diagonal(2), upper(2), rhs(2)
        real(dp), allocatable :: lapack_solution(:), reference_solution(:)
        type(lapack_tridiagonal_diagnostics) :: lapack_diagnostics
        type(linear_solve_diagnostics) :: reference_diagnostics
        integer :: ierr_lapack, ierr_reference

        ! A = [[0,1],[1,1]], x=[1,1], rhs=[1,2].
        ! A zero first diagonal requires a row interchange. A Thomas solver
        ! without pivoting must reject it; LAPACK DGTTRF must pivot and solve it.
        lower = [0.0_dp, 1.0_dp]
        diagonal = [0.0_dp, 1.0_dp]
        upper = [1.0_dp, 0.0_dp]
        rhs = [1.0_dp, 2.0_dp]

        call solve_tridiagonal(lower, diagonal, upper, rhs, 0.0_dp, 0.0_dp, &
                               reference_solution, ierr_reference, reference_diagnostics)
        call require(ierr_reference == IMPLICIT_ERR_PIVOT, &
                     'reference Thomas solver must fail closed when pivoting is required')
        call require(size(reference_solution) == 0, &
                     'failed reference solve must not emit a solution')

        call solve_tridiagonal_lapack(lower, diagonal, upper, rhs, &
                                      lapack_solution, ierr_lapack, lapack_diagnostics)
        call require(ierr_lapack == LAPACK_TRI_OK, &
                     'LAPACK partial-pivoting backend must solve the pivoting system')
        call require(lapack_diagnostics%pivoting_used, &
                     'LAPACK diagnostics must report that row interchange occurred')
        call require(maxval(abs(lapack_solution - [1.0_dp, 1.0_dp])) < 1.0e-13_dp, &
                     'LAPACK pivoting solve must recover the exact solution')
        call require(lapack_diagnostics%residual_inf_norm < 1.0e-13_dp, &
                     'pivoted LAPACK solve must retain residual evidence')
    end subroutine test_partial_pivoting_solves_reference_failure


    subroutine test_factorization_reuse()
        real(dp) :: lower(3), diagonal(3), upper(3), rhs_a(3), rhs_b(3)
        real(dp), allocatable :: solution_a(:), solution_b(:)
        type(lapack_tridiagonal_factorization) :: factor
        type(lapack_tridiagonal_diagnostics) :: factor_diagnostics, solve_diagnostics
        integer :: ierr

        lower = [0.0_dp, -1.0_dp, -1.0_dp]
        diagonal = [2.0_dp, 2.0_dp, 2.0_dp]
        upper = [-1.0_dp, -1.0_dp, 0.0_dp]
        rhs_a = [0.0_dp, 0.0_dp, 4.0_dp]
        rhs_b = [2.0_dp, 0.0_dp, 0.0_dp]

        call factor_tridiagonal_lapack(lower, diagonal, upper, factor, ierr, factor_diagnostics)
        call require(ierr == LAPACK_TRI_OK, 'LAPACK factorization must succeed')

        call solve_factored_tridiagonal_lapack(factor, rhs_a, solution_a, ierr, solve_diagnostics)
        call require(ierr == LAPACK_TRI_OK, 'first LAPACK reused solve must succeed')
        call require(maxval(abs(solution_a - [1.0_dp, 2.0_dp, 3.0_dp])) < 1.0e-13_dp, &
                     'first LAPACK reused solve must recover the known solution')

        call solve_factored_tridiagonal_lapack(factor, rhs_b, solution_b, ierr, solve_diagnostics)
        call require(ierr == LAPACK_TRI_OK, 'second LAPACK reused solve must succeed')
        call require(maxval(abs(solution_b - [1.5_dp, 1.0_dp, 0.5_dp])) < 1.0e-13_dp, &
                     'LAPACK factorization must be reusable across right-hand sides')
    end subroutine test_factorization_reuse

end program test_lapack_tridiagonal
