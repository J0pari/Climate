program test_pressure_coordinate_continuity
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use climate_pressure_coordinate_continuity, only: &
        integrate_pressure_velocity, evaluate_pressure_continuity, &
        CONTINUITY_OK, CONTINUITY_ERR_PRESSURE_ORDER
    implicit none

    integer, parameter :: dp = real64

    call test_constant_divergence_column()
    call test_refinement_invariance()
    call test_closed_column_constraint()
    call test_residual_detects_inconsistent_omega()
    call test_invalid_pressure_order()

contains

    subroutine test_constant_divergence_column()
        real(dp) :: pressure(3), divergence(2), omega(3)
        real(dp) :: layer_residual(2), column_integral, column_residual
        integer :: ierr

        pressure = [100000.0_dp, 80000.0_dp, 50000.0_dp]
        divergence = [1.0e-5_dp, 1.0e-5_dp]

        call integrate_pressure_velocity(pressure, divergence, 0.0_dp, &
            omega, column_integral, ierr)
        call assert_int_equal(ierr, CONTINUITY_OK, 'constant divergence integration status')
        call assert_close(omega(1), 0.0_dp, 1.0e-13_dp, 'lower-boundary omega')
        call assert_close(omega(2), 0.2_dp, 1.0e-13_dp, 'first-interface omega')
        call assert_close(omega(3), 0.5_dp, 1.0e-13_dp, 'top-interface omega')
        call assert_close(column_integral, 0.5_dp, 1.0e-13_dp, &
            'column pressure-integrated divergence')

        call evaluate_pressure_continuity(pressure, divergence, omega, &
            layer_residual, column_residual, ierr)
        call assert_int_equal(ierr, CONTINUITY_OK, 'constant divergence residual status')
        call assert_close(maxval(abs(layer_residual)), 0.0_dp, 1.0e-13_dp, &
            'consistent omega has zero layer residual')
        call assert_close(column_residual, 0.0_dp, 1.0e-13_dp, &
            'consistent omega has zero column residual')
    end subroutine test_constant_divergence_column


    subroutine test_refinement_invariance()
        real(dp) :: coarse_pressure(2), coarse_divergence(1), coarse_omega(2)
        real(dp) :: fine_pressure(3), fine_divergence(2), fine_omega(3)
        real(dp) :: coarse_integral, fine_integral
        integer :: ierr

        coarse_pressure = [100000.0_dp, 50000.0_dp]
        coarse_divergence = [1.25e-5_dp]
        fine_pressure = [100000.0_dp, 75000.0_dp, 50000.0_dp]
        fine_divergence = [1.25e-5_dp, 1.25e-5_dp]

        call integrate_pressure_velocity(coarse_pressure, coarse_divergence, -0.1_dp, &
            coarse_omega, coarse_integral, ierr)
        call assert_int_equal(ierr, CONTINUITY_OK, 'coarse continuity status')
        call integrate_pressure_velocity(fine_pressure, fine_divergence, -0.1_dp, &
            fine_omega, fine_integral, ierr)
        call assert_int_equal(ierr, CONTINUITY_OK, 'refined continuity status')

        call assert_close(fine_omega(3), coarse_omega(2), 1.0e-13_dp, &
            'top omega invariant under constant-divergence refinement')
        call assert_close(fine_integral, coarse_integral, 1.0e-13_dp, &
            'pressure-integrated divergence invariant under refinement')
    end subroutine test_refinement_invariance


    subroutine test_closed_column_constraint()
        real(dp) :: pressure(3), divergence(2), omega(3), column_integral
        integer :: ierr

        pressure = [100000.0_dp, 80000.0_dp, 50000.0_dp]
        divergence(1) = 1.0e-5_dp
        divergence(2) = -(pressure(1) - pressure(2)) * divergence(1) / &
            (pressure(2) - pressure(3))

        call integrate_pressure_velocity(pressure, divergence, 0.0_dp, &
            omega, column_integral, ierr)
        call assert_int_equal(ierr, CONTINUITY_OK, 'closed-column continuity status')
        call assert_close(column_integral, 0.0_dp, 1.0e-13_dp, &
            'closed-column divergence integral')
        call assert_close(omega(3), 0.0_dp, 1.0e-13_dp, &
            'zero lower and upper omega require pressure-integrated divergence closure')
    end subroutine test_closed_column_constraint


    subroutine test_residual_detects_inconsistent_omega()
        real(dp) :: pressure(3), divergence(2), omega(3)
        real(dp) :: layer_residual(2), column_residual
        integer :: ierr

        pressure = [100000.0_dp, 80000.0_dp, 50000.0_dp]
        divergence = [1.0e-5_dp, 1.0e-5_dp]
        omega = [0.0_dp, 0.2_dp, 0.55_dp]

        call evaluate_pressure_continuity(pressure, divergence, omega, &
            layer_residual, column_residual, ierr)
        call assert_int_equal(ierr, CONTINUITY_OK, 'inconsistent omega residual status')
        call assert_close(layer_residual(1), 0.0_dp, 1.0e-13_dp, &
            'unchanged layer remains consistent')
        call assert_close(layer_residual(2), 0.05_dp, 1.0e-13_dp, &
            'perturbed interface produces local residual')
        call assert_close(column_residual, 0.05_dp, 1.0e-13_dp, &
            'perturbed interface produces column residual')
    end subroutine test_residual_detects_inconsistent_omega


    subroutine test_invalid_pressure_order()
        real(dp) :: pressure(3), divergence(2), omega(3), column_integral
        integer :: ierr

        pressure = [100000.0_dp, 80000.0_dp, 90000.0_dp]
        divergence = [1.0e-5_dp, 1.0e-5_dp]
        omega = 42.0_dp
        column_integral = 42.0_dp

        call integrate_pressure_velocity(pressure, divergence, 0.0_dp, &
            omega, column_integral, ierr)
        call assert_int_equal(ierr, CONTINUITY_ERR_PRESSURE_ORDER, &
            'pressure interfaces must strictly decrease upward')
        call assert_close(maxval(abs(omega)), 0.0_dp, 1.0e-13_dp, &
            'failed integration clears omega output')
        call assert_close(column_integral, 0.0_dp, 1.0e-13_dp, &
            'failed integration clears column diagnostic')
    end subroutine test_invalid_pressure_order


    subroutine assert_int_equal(actual, expected, label)
        integer, intent(in) :: actual, expected
        character(len=*), intent(in) :: label

        if (actual /= expected) then
            write(error_unit, '(A,2(1X,I0))') trim(label), actual, expected
            error stop 1
        end if
    end subroutine assert_int_equal


    subroutine assert_close(actual, expected, tolerance, label)
        real(dp), intent(in) :: actual, expected, tolerance
        character(len=*), intent(in) :: label
        real(dp) :: scale

        scale = max(1.0_dp, abs(expected))
        if (abs(actual - expected) > tolerance * scale) then
            write(error_unit, '(A,3(1X,ES24.16))') trim(label), &
                actual, expected, tolerance
            error stop 1
        end if
    end subroutine assert_close

end program test_pressure_coordinate_continuity
