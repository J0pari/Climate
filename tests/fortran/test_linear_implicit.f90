program test_linear_implicit
    use climate_linear_implicit, only: dp, IMPLICIT_OK, IMPLICIT_ERR_PIVOT, &
        IMPLICIT_ERR_PARAMETER, linear_solve_diagnostics, tridiagonal_factorization, &
        factor_tridiagonal, solve_factored_tridiagonal, solve_tridiagonal, theta_tridiagonal_step
    implicit none

    call test_known_tridiagonal_system()
    call test_factorization_reuse()
    call test_scale_aware_pivot_policy()
    call test_elimination_pivot_failure()
    call test_crank_nicolson_scalar_decay()
    call test_constant_diffusion_nullspace()
    call test_invalid_theta_fails_closed()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_known_tridiagonal_system()
        real(dp) :: lower(3), diagonal(3), upper(3), rhs(3)
        real(dp), allocatable :: solution(:)
        type(linear_solve_diagnostics) :: diagnostics
        integer :: ierr

        lower = [0.0_dp, -1.0_dp, -1.0_dp]
        diagonal = [2.0_dp, 2.0_dp, 2.0_dp]
        upper = [-1.0_dp, -1.0_dp, 0.0_dp]
        rhs = [0.0_dp, 0.0_dp, 4.0_dp]

        call solve_tridiagonal(lower, diagonal, upper, rhs, 0.0_dp, 1.0e-14_dp, &
                               solution, ierr, diagnostics)

        call require(ierr == IMPLICIT_OK, 'known tridiagonal solve must succeed')
        call require(maxval(abs(solution - [1.0_dp, 2.0_dp, 3.0_dp])) < 1.0e-13_dp, &
                     'Thomas solve must recover the known solution')
        call require(diagnostics%min_abs_pivot > 1.0_dp, &
                     'solver must report the encountered pivot margin')
        call require(diagnostics%min_scaled_pivot > 0.25_dp, &
                     'solver must expose pivot scale relative to matrix infinity norm')
        call require(diagnostics%residual_inf_norm < 1.0e-13_dp, &
                     'solver must report a small residual for the known system')
    end subroutine test_known_tridiagonal_system


    subroutine test_factorization_reuse()
        real(dp) :: lower(3), diagonal(3), upper(3), rhs_a(3), rhs_b(3)
        real(dp), allocatable :: solution_a(:), solution_b(:)
        type(tridiagonal_factorization) :: factor
        type(linear_solve_diagnostics) :: factor_diagnostics, solve_diagnostics
        integer :: ierr

        lower = [0.0_dp, -1.0_dp, -1.0_dp]
        diagonal = [2.0_dp, 2.0_dp, 2.0_dp]
        upper = [-1.0_dp, -1.0_dp, 0.0_dp]
        rhs_a = [0.0_dp, 0.0_dp, 4.0_dp]
        rhs_b = [2.0_dp, 0.0_dp, 0.0_dp]

        call factor_tridiagonal(lower, diagonal, upper, 0.0_dp, 1.0e-14_dp, &
                                factor, ierr, factor_diagnostics)
        call require(ierr == IMPLICIT_OK, 'reusable tridiagonal factorization must succeed')

        call solve_factored_tridiagonal(factor, rhs_a, solution_a, ierr, solve_diagnostics)
        call require(ierr == IMPLICIT_OK, 'first factored solve must succeed')
        call require(maxval(abs(solution_a - [1.0_dp, 2.0_dp, 3.0_dp])) < 1.0e-13_dp, &
                     'first reused solve must recover the known solution')

        call solve_factored_tridiagonal(factor, rhs_b, solution_b, ierr, solve_diagnostics)
        call require(ierr == IMPLICIT_OK, 'second factored solve must succeed')
        call require(maxval(abs(solution_b - [1.5_dp, 1.0_dp, 0.5_dp])) < 1.0e-13_dp, &
                     'one factorization must support a distinct right-hand side')
        call require(solve_diagnostics%residual_inf_norm < 1.0e-13_dp, &
                     'reused solve must retain residual diagnostics')
    end subroutine test_factorization_reuse


    subroutine test_scale_aware_pivot_policy()
        real(dp) :: lower(2), diagonal(2), upper(2), rhs(2)
        real(dp), allocatable :: solution(:)
        type(linear_solve_diagnostics) :: diagnostics
        integer :: ierr

        lower = [0.0_dp, 0.0_dp]
        diagonal = [1.0e-12_dp, 1.0_dp]
        upper = [0.0_dp, 0.0_dp]
        rhs = [1.0e-12_dp, 1.0_dp]

        call solve_tridiagonal(lower, diagonal, upper, rhs, 0.0_dp, 1.0e-10_dp, &
                               solution, ierr, diagnostics)

        call require(ierr == IMPLICIT_ERR_PIVOT, &
                     'relative pivot policy must reject a scale-small pivot')
        call require(size(solution) == 0, 'rejected scaled pivot must not emit a solution')
        call require(abs(diagnostics%matrix_inf_norm - 1.0_dp) < 1.0e-15_dp, &
                     'pivot policy must scale against the matrix infinity norm')
    end subroutine test_scale_aware_pivot_policy


    subroutine test_elimination_pivot_failure()
        real(dp) :: lower(2), diagonal(2), upper(2), rhs(2)
        real(dp), allocatable :: solution(:)
        type(linear_solve_diagnostics) :: diagnostics
        integer :: ierr

        lower = [0.0_dp, 1.0_dp]
        diagonal = [1.0_dp, 1.0_dp]
        upper = [1.0_dp, 0.0_dp]
        rhs = [2.0_dp, 2.0_dp]

        call solve_tridiagonal(lower, diagonal, upper, rhs, 0.0_dp, 1.0e-14_dp, &
                               solution, ierr, diagnostics)

        call require(ierr == IMPLICIT_ERR_PIVOT, &
                     'zero pivot created by elimination must fail closed')
        call require(size(solution) == 0, 'pivot failure must not emit a solution')
        call require(diagnostics%min_abs_pivot == 0.0_dp, &
                     'pivot diagnostic must expose the zero elimination pivot')
    end subroutine test_elimination_pivot_failure


    subroutine test_crank_nicolson_scalar_decay()
        real(dp) :: lower(1), diagonal(1), upper(1), state(1), source(1)
        real(dp), allocatable :: result(:)
        real(dp) :: expected
        type(linear_solve_diagnostics) :: diagnostics
        integer :: ierr

        lower = [0.0_dp]
        diagonal = [-2.0_dp]
        upper = [0.0_dp]
        state = [1.0_dp]
        source = [0.0_dp]
        expected = 0.9_dp / 1.1_dp

        call theta_tridiagonal_step(lower, diagonal, upper, state, source, 0.1_dp, 0.5_dp, &
                                    0.0_dp, 1.0e-14_dp, result, ierr, diagnostics)

        call require(ierr == IMPLICIT_OK, 'Crank-Nicolson scalar decay must succeed')
        call require(abs(result(1) - expected) < 1.0e-14_dp, &
                     'theta=0.5 must match the exact Crank-Nicolson rational update')
        call require(diagnostics%residual_inf_norm < 1.0e-14_dp, &
                     'Crank-Nicolson solve must report its algebraic residual')
    end subroutine test_crank_nicolson_scalar_decay


    subroutine test_constant_diffusion_nullspace()
        real(dp) :: lower(4), diagonal(4), upper(4), state(4), source(4)
        real(dp), allocatable :: result(:)
        type(linear_solve_diagnostics) :: diagnostics
        integer :: ierr

        lower = [0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
        diagonal = [-1.0_dp, -2.0_dp, -2.0_dp, -1.0_dp]
        upper = [1.0_dp, 1.0_dp, 1.0_dp, 0.0_dp]
        state = [3.0_dp, 3.0_dp, 3.0_dp, 3.0_dp]
        source = 0.0_dp

        call theta_tridiagonal_step(lower, diagonal, upper, state, source, 5.0_dp, 1.0_dp, &
                                    0.0_dp, 1.0e-14_dp, result, ierr, diagnostics)

        call require(ierr == IMPLICIT_OK, 'backward-Euler diffusion nullspace test must succeed')
        call require(maxval(abs(result - state)) < 1.0e-13_dp, &
                     'constant state must remain invariant under a zero-flux diffusion operator')
    end subroutine test_constant_diffusion_nullspace


    subroutine test_invalid_theta_fails_closed()
        real(dp) :: lower(1), diagonal(1), upper(1), state(1), source(1)
        real(dp), allocatable :: result(:)
        type(linear_solve_diagnostics) :: diagnostics
        integer :: ierr

        lower = [0.0_dp]
        diagonal = [-1.0_dp]
        upper = [0.0_dp]
        state = [1.0_dp]
        source = [0.0_dp]

        call theta_tridiagonal_step(lower, diagonal, upper, state, source, 0.1_dp, 1.1_dp, &
                                    0.0_dp, 1.0e-14_dp, result, ierr, diagnostics)

        call require(ierr == IMPLICIT_ERR_PARAMETER, 'theta outside [0,1] must be rejected')
        call require(size(result) == 0, 'invalid theta must not emit a state')
    end subroutine test_invalid_theta_fails_closed

end program test_linear_implicit
