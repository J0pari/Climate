program test_sigma_coordinate
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_sigma_coordinate, only: &
        sigma_interfaces_to_pressure, &
        SIGMA_COORDINATE_OK, SIGMA_COORDINATE_ERR_SIZE, &
        SIGMA_COORDINATE_ERR_NONFINITE, SIGMA_COORDINATE_ERR_SURFACE_PRESSURE, &
        SIGMA_COORDINATE_ERR_RANGE, SIGMA_COORDINATE_ERR_SURFACE_ANCHOR, &
        SIGMA_COORDINATE_ERR_ORDER
    use climate_pressure_coordinate_grid, only: &
        validate_pressure_interfaces, PRESSURE_GRID_OK
    use climate_pressure_coordinate_hydrostatics, only: &
        hydrostatic_parameters, integrate_pressure_column, HYDRO_OK
    implicit none

    integer, parameter :: dp = real64
    type(hydrostatic_parameters) :: hydro_parameters

    call test_sigma_mapping_reduces_to_pressure_coordinates()
    call test_layer_mass_uses_existing_pressure_hydrostatics(hydro_parameters)
    call test_invalid_shapes_fail_closed()
    call test_nonfinite_values_fail_closed()
    call test_surface_pressure_domain()
    call test_sigma_domain_and_surface_anchor()
    call test_sigma_orientation()

contains

    subroutine test_sigma_mapping_reduces_to_pressure_coordinates()
        real(dp) :: sigma(4), pressure(4), expected(4)
        integer :: ierr, pressure_ierr

        sigma = [1.0_dp, 0.8_dp, 0.5_dp, 0.2_dp]
        expected = [100000.0_dp, 80000.0_dp, 50000.0_dp, 20000.0_dp]
        call sigma_interfaces_to_pressure(sigma, 100000.0_dp, pressure, ierr)

        call assert_int_equal(ierr, SIGMA_COORDINATE_OK, 'sigma mapping status')
        call assert_array_close(pressure, expected, 0.0_dp, &
            'sigma mapping reproduces declared pressure interfaces')
        call validate_pressure_interfaces(pressure, pressure_ierr)
        call assert_int_equal(pressure_ierr, PRESSURE_GRID_OK, &
            'mapped sigma interfaces satisfy pressure-grid contract')
    end subroutine test_sigma_mapping_reduces_to_pressure_coordinates


    subroutine test_layer_mass_uses_existing_pressure_hydrostatics(parameters)
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp) :: sigma(4), pressure(4), temperature(3)
        real(dp) :: mass(3), dphi(3), dz(3), phi(4), expected_mass
        integer :: ierr

        sigma = [1.0_dp, 0.8_dp, 0.5_dp, 0.2_dp]
        temperature = [290.0_dp, 270.0_dp, 250.0_dp]
        call sigma_interfaces_to_pressure(sigma, 100000.0_dp, pressure, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_OK, 'sigma hydro mapping status')

        call integrate_pressure_column(pressure, temperature, parameters, &
            mass, dphi, dz, phi, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'sigma pressure hydrostatic status')
        expected_mass = 100000.0_dp * (sigma(1) - sigma(4)) / &
            parameters%gravity_m_s2
        call assert_close(sum(mass), expected_mass, 1.0e-12_dp, &
            'sigma layer mass delegates to pressure hydrostatics')
    end subroutine test_layer_mass_uses_existing_pressure_hydrostatics


    subroutine test_invalid_shapes_fail_closed()
        real(dp) :: sigma_short(1), pressure_short(1)
        real(dp) :: sigma(3), pressure_wrong(2)
        integer :: ierr

        sigma_short = [1.0_dp]
        pressure_short = -1.0_dp
        call sigma_interfaces_to_pressure(sigma_short, 100000.0_dp, &
            pressure_short, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_SIZE, &
            'sigma needs at least two interfaces')
        call assert_true(all(pressure_short == 0.0_dp), &
            'failed short mapping clears output')

        sigma = [1.0_dp, 0.5_dp, 0.2_dp]
        pressure_wrong = -1.0_dp
        call sigma_interfaces_to_pressure(sigma, 100000.0_dp, pressure_wrong, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_SIZE, &
            'sigma output shape must match input')
        call assert_true(all(pressure_wrong == 0.0_dp), &
            'failed shape mapping clears output')
    end subroutine test_invalid_shapes_fail_closed


    subroutine test_nonfinite_values_fail_closed()
        real(dp) :: sigma(2), pressure(2), nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        sigma = [1.0_dp, nan_value]
        call sigma_interfaces_to_pressure(sigma, 100000.0_dp, pressure, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_NONFINITE, &
            'nonfinite sigma rejected')

        sigma = [1.0_dp, 0.5_dp]
        call sigma_interfaces_to_pressure(sigma, nan_value, pressure, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_NONFINITE, &
            'nonfinite surface pressure rejected')
    end subroutine test_nonfinite_values_fail_closed


    subroutine test_surface_pressure_domain()
        real(dp) :: sigma(2), pressure(2)
        integer :: ierr

        sigma = [1.0_dp, 0.5_dp]
        call sigma_interfaces_to_pressure(sigma, 0.0_dp, pressure, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_SURFACE_PRESSURE, &
            'surface pressure must be positive')
    end subroutine test_surface_pressure_domain


    subroutine test_sigma_domain_and_surface_anchor()
        real(dp) :: pressure(2)
        integer :: ierr

        call sigma_interfaces_to_pressure([1.0_dp, 0.0_dp], 100000.0_dp, &
            pressure, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_RANGE, &
            'zero top sigma is outside finite-positive pressure domain')

        call sigma_interfaces_to_pressure([1.1_dp, 0.5_dp], 100000.0_dp, &
            pressure, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_RANGE, &
            'sigma above one rejected')

        call sigma_interfaces_to_pressure([0.9_dp, 0.5_dp], 100000.0_dp, &
            pressure, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_SURFACE_ANCHOR, &
            'first sigma interface must be the surface')
    end subroutine test_sigma_domain_and_surface_anchor


    subroutine test_sigma_orientation()
        real(dp) :: pressure(3)
        integer :: ierr

        call sigma_interfaces_to_pressure([1.0_dp, 0.6_dp, 0.7_dp], &
            100000.0_dp, pressure, ierr)
        call assert_int_equal(ierr, SIGMA_COORDINATE_ERR_ORDER, &
            'sigma interfaces strictly decrease upward')
    end subroutine test_sigma_orientation


    subroutine assert_array_close(actual, expected, tolerance, label)
        real(dp), intent(in) :: actual(:), expected(:), tolerance
        character(len=*), intent(in) :: label

        if (size(actual) /= size(expected) .or. &
            any(abs(actual - expected) > tolerance)) then
            write(error_unit, '(A)') trim(label)
            error stop 1
        end if
    end subroutine assert_array_close


    subroutine assert_close(actual, expected, relative_tolerance, label)
        real(dp), intent(in) :: actual, expected, relative_tolerance
        character(len=*), intent(in) :: label
        real(dp) :: scale

        scale = max(abs(expected), 1.0_dp)
        if (abs(actual - expected) > relative_tolerance * scale) then
            write(error_unit, '(A,2(1X,ES24.16))') trim(label), actual, expected
            error stop 1
        end if
    end subroutine assert_close


    subroutine assert_int_equal(actual, expected, label)
        integer, intent(in) :: actual, expected
        character(len=*), intent(in) :: label

        if (actual /= expected) then
            write(error_unit, '(A,2(1X,I0))') trim(label), actual, expected
            error stop 1
        end if
    end subroutine assert_int_equal


    subroutine assert_true(condition, label)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label

        if (.not. condition) then
            write(error_unit, '(A)') trim(label)
            error stop 1
        end if
    end subroutine assert_true

end program test_sigma_coordinate
