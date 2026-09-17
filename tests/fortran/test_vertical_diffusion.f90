program test_vertical_diffusion
    use climate_vertical_diffusion, only: dp, VBC_ZERO_FLUX, VBC_PRESCRIBED_FLUX, &
        VBC_PRESCRIBED_VALUE, VDIFF_OK, VDIFF_ERR_DIFFUSIVITY, VDIFF_ERR_BOUNDARY, &
        height_diffusion_boundary, height_diffusion_operator, &
        build_height_diffusion_operator, evaluate_height_diffusion
    implicit none

    call test_nonuniform_zero_flux_conservation_and_dissipation()
    call test_prescribed_flux_budget_and_sign()
    call test_dirichlet_linear_steady_state()
    call test_invalid_boundary_kind_fails_closed()
    call test_negative_diffusivity_fails_closed()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_nonuniform_zero_flux_conservation_and_dissipation()
        real(dp) :: thickness(3), face_diffusivity(4), state(3), constant_state(3)
        real(dp), allocatable :: tendency(:)
        real(dp) :: weighted_quadratic
        type(height_diffusion_boundary) :: bottom, top
        type(height_diffusion_operator) :: operator
        integer :: ierr

        thickness = [1.0_dp, 2.0_dp, 0.5_dp]
        face_diffusivity = [0.0_dp, 3.0_dp, 1.0_dp, 0.0_dp]
        bottom = height_diffusion_boundary(VBC_ZERO_FLUX, 0.0_dp)
        top = height_diffusion_boundary(VBC_ZERO_FLUX, 0.0_dp)

        call build_height_diffusion_operator(thickness, face_diffusivity, bottom, top, operator, ierr)
        call require(ierr == VDIFF_OK, 'nonuniform zero-flux operator construction must succeed')
        call require(operator%diagnostics%constant_nullspace_expected, &
                     'zero-flux boundaries must declare the homogeneous constant nullspace')
        call require(operator%diagnostics%conservative_homogeneous_operator_expected, &
                     'zero-flux boundaries must declare homogeneous column conservation')
        call require(operator%diagnostics%constant_nullspace_defect_s_inv < 1.0e-14_dp, &
                     'assembled zero-flux operator must preserve constants')
        call require(operator%diagnostics%weighted_conservation_defect_m_s < 1.0e-14_dp, &
                     'finite-volume operator must conserve the thickness-weighted column integral')

        state = [1.0_dp, -2.0_dp, 3.0_dp]
        call evaluate_height_diffusion(operator, state, tendency, ierr)
        call require(ierr == VDIFF_OK, 'zero-flux diffusion evaluation must succeed')
        call require(abs(dot_product(thickness, tendency)) < 1.0e-13_dp, &
                     'zero-flux diffusion must not change the weighted column integral')

        weighted_quadratic = dot_product(thickness * state, tendency)
        call require(weighted_quadratic <= 1.0e-13_dp, &
                     'nonnegative diffusivity must make the weighted diffusion quadratic form non-positive')

        constant_state = 4.0_dp
        call evaluate_height_diffusion(operator, constant_state, tendency, ierr)
        call require(ierr == VDIFF_OK, 'constant-state diffusion evaluation must succeed')
        call require(maxval(abs(tendency)) < 1.0e-14_dp, &
                     'constant state must be an exact null mode for zero-flux diffusion')
    end subroutine test_nonuniform_zero_flux_conservation_and_dissipation


    subroutine test_prescribed_flux_budget_and_sign()
        real(dp) :: thickness(2), face_diffusivity(3), state(2)
        real(dp), allocatable :: tendency(:)
        type(height_diffusion_boundary) :: bottom, top
        type(height_diffusion_operator) :: operator
        integer :: ierr

        thickness = [1.0_dp, 2.0_dp]
        face_diffusivity = 0.0_dp
        state = [7.0_dp, -4.0_dp]
        bottom = height_diffusion_boundary(VBC_PRESCRIBED_FLUX, 2.0_dp)
        top = height_diffusion_boundary(VBC_PRESCRIBED_FLUX, 0.5_dp)

        call build_height_diffusion_operator(thickness, face_diffusivity, bottom, top, operator, ierr)
        call require(ierr == VDIFF_OK, 'prescribed-flux operator construction must succeed')
        call evaluate_height_diffusion(operator, state, tendency, ierr)
        call require(ierr == VDIFF_OK, 'prescribed-flux diffusion evaluation must succeed')
        call require(abs(dot_product(thickness, tendency) - 1.5_dp) < 1.0e-14_dp, &
                     'positive upward physical flux must change inventory by q_bottom - q_top')
        call require(abs(tendency(1) - 2.0_dp) < 1.0e-14_dp, &
                     'positive bottom upward flux must enter the lowest cell')
        call require(abs(tendency(2) + 0.25_dp) < 1.0e-14_dp, &
                     'positive top upward flux must leave the highest cell')
    end subroutine test_prescribed_flux_budget_and_sign


    subroutine test_dirichlet_linear_steady_state()
        integer, parameter :: n = 4
        real(dp) :: thickness(n), face_diffusivity(n + 1), state(n)
        real(dp), allocatable :: tendency(:)
        type(height_diffusion_boundary) :: bottom, top
        type(height_diffusion_operator) :: operator
        integer :: ierr

        thickness = 0.25_dp
        face_diffusivity = 1.0_dp
        state = [0.125_dp, 0.375_dp, 0.625_dp, 0.875_dp]
        bottom = height_diffusion_boundary(VBC_PRESCRIBED_VALUE, 0.0_dp)
        top = height_diffusion_boundary(VBC_PRESCRIBED_VALUE, 1.0_dp)

        call build_height_diffusion_operator(thickness, face_diffusivity, bottom, top, operator, ierr)
        call require(ierr == VDIFF_OK, 'fixed-value operator construction must succeed')
        call require(.not. operator%diagnostics%constant_nullspace_expected, &
                     'fixed-value boundaries must not claim a constant homogeneous nullspace')

        call evaluate_height_diffusion(operator, state, tendency, ierr)
        call require(ierr == VDIFF_OK, 'fixed-value steady-state evaluation must succeed')
        call require(maxval(abs(tendency)) < 1.0e-13_dp, &
                     'linear profile between fixed boundary values must be a discrete steady state')
    end subroutine test_dirichlet_linear_steady_state


    subroutine test_invalid_boundary_kind_fails_closed()
        real(dp) :: thickness(2), face_diffusivity(3)
        type(height_diffusion_boundary) :: bottom, top
        type(height_diffusion_operator) :: operator
        integer :: ierr

        thickness = [1.0_dp, 1.0_dp]
        face_diffusivity = 0.0_dp
        bottom = height_diffusion_boundary(99, 0.0_dp)
        top = height_diffusion_boundary(VBC_ZERO_FLUX, 0.0_dp)

        call build_height_diffusion_operator(thickness, face_diffusivity, bottom, top, operator, ierr)
        call require(ierr == VDIFF_ERR_BOUNDARY, 'undeclared boundary kind must be rejected')
        call require(operator%n == 0, 'invalid boundary must not emit a usable operator')
    end subroutine test_invalid_boundary_kind_fails_closed


    subroutine test_negative_diffusivity_fails_closed()
        real(dp) :: thickness(2), face_diffusivity(3)
        type(height_diffusion_boundary) :: bottom, top
        type(height_diffusion_operator) :: operator
        integer :: ierr

        thickness = [1.0_dp, 1.0_dp]
        face_diffusivity = [0.0_dp, -1.0_dp, 0.0_dp]
        bottom = height_diffusion_boundary(VBC_ZERO_FLUX, 0.0_dp)
        top = height_diffusion_boundary(VBC_ZERO_FLUX, 0.0_dp)

        call build_height_diffusion_operator(thickness, face_diffusivity, bottom, top, operator, ierr)
        call require(ierr == VDIFF_ERR_DIFFUSIVITY, 'negative diffusivity must be rejected')
        call require(operator%n == 0, 'invalid diffusivity must not emit a usable operator')
    end subroutine test_negative_diffusivity_fails_closed

end program test_vertical_diffusion
