program test_vertical_diffusion_lapack
    use climate_vertical_diffusion, only: dp, VBC_ZERO_FLUX, VDIFF_OK, &
        height_diffusion_boundary, height_diffusion_operator, build_height_diffusion_operator
    use climate_lapack_tridiagonal, only: LAPACK_TRI_OK, lapack_tridiagonal_diagnostics, &
        solve_tridiagonal_lapack
    implicit none

    call test_backward_euler_zero_flux_conservation()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_backward_euler_zero_flux_conservation()
        real(dp) :: thickness(3), face_diffusivity(4), state(3)
        real(dp) :: lower(3), diagonal(3), upper(3), rhs(3)
        real(dp), allocatable :: next_state(:)
        real(dp) :: dt, mass_before, mass_after, mean_before
        real(dp) :: variance_before, variance_after
        type(height_diffusion_boundary) :: bottom, top
        type(height_diffusion_operator) :: operator
        type(lapack_tridiagonal_diagnostics) :: diagnostics
        integer :: ierr

        thickness = [1.0_dp, 2.0_dp, 0.5_dp]
        face_diffusivity = [0.0_dp, 3.0_dp, 1.0_dp, 0.0_dp]
        state = [1.0_dp, -2.0_dp, 3.0_dp]
        bottom = height_diffusion_boundary(VBC_ZERO_FLUX, 0.0_dp)
        top = height_diffusion_boundary(VBC_ZERO_FLUX, 0.0_dp)
        dt = 10.0_dp

        call build_height_diffusion_operator(thickness, face_diffusivity, bottom, top, operator, ierr)
        call require(ierr == VDIFF_OK, 'zero-flux operator construction must succeed before LAPACK solve')

        lower = -dt * operator%lower_s_inv
        diagonal = 1.0_dp - dt * operator%diagonal_s_inv
        upper = -dt * operator%upper_s_inv
        rhs = state + dt * operator%source_per_s

        mass_before = dot_product(thickness, state)
        mean_before = mass_before / sum(thickness)
        variance_before = dot_product(thickness, (state - mean_before)**2)

        call solve_tridiagonal_lapack(lower, diagonal, upper, rhs, next_state, ierr, diagnostics)
        call require(ierr == LAPACK_TRI_OK, 'LAPACK backward-Euler diffusion solve must succeed')
        call require(diagnostics%residual_inf_norm < 1.0e-12_dp, &
                     'LAPACK diffusion solve must retain a small algebraic residual')

        mass_after = dot_product(thickness, next_state)
        variance_after = dot_product(thickness, (next_state - mean_before)**2)

        call require(abs(mass_after - mass_before) < 1.0e-12_dp, &
                     'backward-Euler zero-flux diffusion must preserve weighted column inventory')
        call require(variance_after <= variance_before + 1.0e-12_dp, &
                     'implicit diffusion must not increase weighted variance about the conserved mean')
    end subroutine test_backward_euler_zero_flux_conservation

end program test_vertical_diffusion_lapack
