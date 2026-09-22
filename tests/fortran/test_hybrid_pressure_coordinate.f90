program test_hybrid_pressure_coordinate
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_hybrid_pressure_coordinate, only: &
        hybrid_interfaces_to_pressure, &
        HYBRID_COORDINATE_OK, HYBRID_COORDINATE_ERR_SIZE, &
        HYBRID_COORDINATE_ERR_SOURCE, HYBRID_COORDINATE_ERR_NONFINITE, &
        HYBRID_COORDINATE_ERR_SURFACE_PRESSURE, HYBRID_COORDINATE_ERR_A_RANGE, &
        HYBRID_COORDINATE_ERR_B_RANGE, HYBRID_COORDINATE_ERR_SURFACE_ANCHOR, &
        HYBRID_COORDINATE_ERR_PRESSURE_GRID
    use climate_sigma_coordinate, only: sigma_interfaces_to_pressure, &
        SIGMA_COORDINATE_OK
    use climate_pressure_coordinate_hydrostatics, only: &
        hydrostatic_parameters, integrate_pressure_column, HYDRO_OK
    implicit none

    integer, parameter :: dp = real64
    character(len=*), parameter :: SOURCE_ID = &
        'fixture:model-level-hybrid-coefficients/v1'
    type(hydrostatic_parameters) :: hydro_parameters

    call test_sigma_limit()
    call test_hybrid_surface_pressure_dependence()
    call test_layer_mass_uses_pressure_hydrostatics(hydro_parameters)
    call test_source_identity_is_required()
    call test_invalid_shapes_fail_closed()
    call test_nonfinite_and_surface_pressure_fail_closed()
    call test_coefficient_domains()
    call test_surface_anchor_is_coefficient_identity()
    call test_mapped_pressure_contract_fails_closed()

contains

    subroutine test_sigma_limit()
        real(dp) :: a_pa(4), b(4), hybrid_pressure(4), sigma_pressure(4)
        integer :: ierr, sigma_ierr

        a_pa = 0.0_dp
        b = [1.0_dp, 0.8_dp, 0.5_dp, 0.2_dp]
        call hybrid_interfaces_to_pressure( &
            a_pa, b, 100000.0_dp, SOURCE_ID, hybrid_pressure, ierr)
        call sigma_interfaces_to_pressure(b, 100000.0_dp, sigma_pressure, sigma_ierr)

        call assert_int_equal(ierr, HYBRID_COORDINATE_OK, 'hybrid sigma-limit status')
        call assert_int_equal(sigma_ierr, SIGMA_COORDINATE_OK, &
            'sigma reference status')
        call assert_array_close(hybrid_pressure, sigma_pressure, 0.0_dp, &
            'A=0 hybrid coefficients reduce exactly to sigma mapping')
    end subroutine test_sigma_limit


    subroutine test_hybrid_surface_pressure_dependence()
        real(dp) :: a_pa(4), b(4), pressure_a(4), pressure_b(4), expected_delta(4)
        integer :: ierr

        a_pa = [0.0_dp, 5000.0_dp, 12000.0_dp, 10000.0_dp]
        b = [1.0_dp, 0.75_dp, 0.35_dp, 0.0_dp]
        call hybrid_interfaces_to_pressure( &
            a_pa, b, 100000.0_dp, SOURCE_ID, pressure_a, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_OK, &
            'hybrid first surface-pressure status')
        call hybrid_interfaces_to_pressure( &
            a_pa, b, 90000.0_dp, SOURCE_ID, pressure_b, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_OK, &
            'hybrid second surface-pressure status')

        expected_delta = b * 10000.0_dp
        call assert_array_close(pressure_a - pressure_b, expected_delta, 0.0_dp, &
            'B coefficients are the fixed-coefficient dp/dps metric')
        call assert_close(pressure_a(4), 10000.0_dp, 0.0_dp, &
            'B=0 top interface is fixed pressure')
    end subroutine test_hybrid_surface_pressure_dependence


    subroutine test_layer_mass_uses_pressure_hydrostatics(parameters)
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp) :: a_pa(4), b(4), pressure(4), temperature(3)
        real(dp) :: mass(3), dphi(3), dz(3), phi(4), expected_mass
        integer :: ierr

        a_pa = [0.0_dp, 5000.0_dp, 12000.0_dp, 10000.0_dp]
        b = [1.0_dp, 0.75_dp, 0.35_dp, 0.0_dp]
        temperature = [290.0_dp, 270.0_dp, 250.0_dp]
        call hybrid_interfaces_to_pressure( &
            a_pa, b, 100000.0_dp, SOURCE_ID, pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_OK, 'hybrid hydro mapping status')
        call integrate_pressure_column(pressure, temperature, parameters, &
            mass, dphi, dz, phi, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'hybrid pressure hydrostatic status')

        expected_mass = (pressure(1) - pressure(4)) / parameters%gravity_m_s2
        call assert_close(sum(mass), expected_mass, 1.0e-12_dp, &
            'hybrid layer mass delegates to pressure hydrostatics')
    end subroutine test_layer_mass_uses_pressure_hydrostatics


    subroutine test_source_identity_is_required()
        real(dp) :: pressure(2)
        integer :: ierr

        call hybrid_interfaces_to_pressure( &
            [0.0_dp, 10000.0_dp], [1.0_dp, 0.0_dp], 100000.0_dp, &
            '   ', pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_SOURCE, &
            'hybrid coefficients require a named source identity')
        call assert_true(all(pressure == 0.0_dp), &
            'missing coefficient identity clears output')
    end subroutine test_source_identity_is_required


    subroutine test_invalid_shapes_fail_closed()
        real(dp) :: pressure(2)
        integer :: ierr

        call hybrid_interfaces_to_pressure( &
            [0.0_dp], [1.0_dp], 100000.0_dp, SOURCE_ID, pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_SIZE, &
            'hybrid needs at least two interfaces')

        call hybrid_interfaces_to_pressure( &
            [0.0_dp, 10000.0_dp], [1.0_dp], 100000.0_dp, SOURCE_ID, &
            pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_SIZE, &
            'hybrid A and B shapes must match')
    end subroutine test_invalid_shapes_fail_closed


    subroutine test_nonfinite_and_surface_pressure_fail_closed()
        real(dp) :: pressure(2), nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        call hybrid_interfaces_to_pressure( &
            [0.0_dp, nan_value], [1.0_dp, 0.0_dp], 100000.0_dp, SOURCE_ID, &
            pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_NONFINITE, &
            'nonfinite hybrid coefficient rejected')

        call hybrid_interfaces_to_pressure( &
            [0.0_dp, 10000.0_dp], [1.0_dp, 0.0_dp], nan_value, SOURCE_ID, &
            pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_NONFINITE, &
            'nonfinite hybrid surface pressure rejected')

        call hybrid_interfaces_to_pressure( &
            [0.0_dp, 10000.0_dp], [1.0_dp, 0.0_dp], 0.0_dp, SOURCE_ID, &
            pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_SURFACE_PRESSURE, &
            'hybrid surface pressure must be positive')
    end subroutine test_nonfinite_and_surface_pressure_fail_closed


    subroutine test_coefficient_domains()
        real(dp) :: pressure(2)
        integer :: ierr

        call hybrid_interfaces_to_pressure( &
            [0.0_dp, -1.0_dp], [1.0_dp, 0.0_dp], 100000.0_dp, SOURCE_ID, &
            pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_A_RANGE, &
            'hybrid A coefficients must be nonnegative pressure offsets')

        call hybrid_interfaces_to_pressure( &
            [0.0_dp, 10000.0_dp], [1.0_dp, 1.1_dp], 100000.0_dp, SOURCE_ID, &
            pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_B_RANGE, &
            'hybrid B coefficients must remain dimensionless fractions')
    end subroutine test_coefficient_domains


    subroutine test_surface_anchor_is_coefficient_identity()
        real(dp) :: pressure(2)
        integer :: ierr

        call hybrid_interfaces_to_pressure( &
            [1000.0_dp, 10000.0_dp], [0.99_dp, 0.0_dp], 100000.0_dp, &
            SOURCE_ID, pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_SURFACE_ANCHOR, &
            'surface interface must be A=0 and B=1 independently of ps')
    end subroutine test_surface_anchor_is_coefficient_identity


    subroutine test_mapped_pressure_contract_fails_closed()
        real(dp) :: pressure(3)
        integer :: ierr

        call hybrid_interfaces_to_pressure( &
            [0.0_dp, 90000.0_dp, 10000.0_dp], [1.0_dp, 0.2_dp, 0.0_dp], &
            100000.0_dp, SOURCE_ID, pressure, ierr)
        call assert_int_equal(ierr, HYBRID_COORDINATE_ERR_PRESSURE_GRID, &
            'mapped hybrid interfaces must strictly decrease upward')
        call assert_true(all(pressure == 0.0_dp), &
            'invalid mapped pressure grid clears output')
    end subroutine test_mapped_pressure_contract_fails_closed


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

end program test_hybrid_pressure_coordinate
