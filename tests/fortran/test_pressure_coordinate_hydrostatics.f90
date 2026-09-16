program test_pressure_coordinate_hydrostatics
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use climate_pressure_coordinate_hydrostatics, only: &
        hydrostatic_parameters, compute_hydrostatic_layer, integrate_pressure_column, &
        HYDRO_OK, HYDRO_ERR_SHAPE, HYDRO_ERR_PRESSURE_ORDER
    use climate_moist_vapor_algebra, only: &
        moist_vapor_parameters, compute_virtual_temperature_from_mixing_ratio, MOIST_OK
    implicit none

    integer, parameter :: dp = real64
    type(hydrostatic_parameters) :: parameters
    type(moist_vapor_parameters) :: moist_parameters

    call test_single_layer_identity(parameters)
    call test_refinement_invariance(parameters)
    call test_column_mass_telescopes(parameters)
    call test_moist_virtual_temperature_effect(parameters, moist_parameters)
    call test_invalid_pressure_order(parameters)
    call test_shape_failure(parameters)

contains

    subroutine test_single_layer_identity(parameters)
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp) :: mass, dphi, dz, expected_dphi
        integer :: ierr

        call compute_hydrostatic_layer(100000.0_dp, 50000.0_dp, 280.0_dp, &
            parameters, mass, dphi, dz, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'single layer status')

        expected_dphi = parameters%dry_air_gas_constant_j_kg_k * 280.0_dp * log(2.0_dp)
        call assert_close(mass, 50000.0_dp / parameters%gravity_m_s2, &
            1.0e-12_dp, 'hydrostatic layer mass')
        call assert_close(dphi, expected_dphi, 1.0e-12_dp, &
            'hypsometric geopotential thickness')
        call assert_close(dz, expected_dphi / parameters%gravity_m_s2, &
            1.0e-12_dp, 'geometric thickness')
    end subroutine test_single_layer_identity


    subroutine test_refinement_invariance(parameters)
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp) :: coarse_mass, coarse_dphi, coarse_dz
        real(dp) :: fine_mass(2), fine_dphi(2), fine_dz(2), phi(3)
        real(dp) :: pressures(3), temperatures(2)
        integer :: ierr

        call compute_hydrostatic_layer(100000.0_dp, 25000.0_dp, 280.0_dp, &
            parameters, coarse_mass, coarse_dphi, coarse_dz, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'coarse layer status')

        pressures = [100000.0_dp, 50000.0_dp, 25000.0_dp]
        temperatures = [280.0_dp, 280.0_dp]
        call integrate_pressure_column(pressures, temperatures, parameters, &
            fine_mass, fine_dphi, fine_dz, phi, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'refined column status')

        call assert_close(sum(fine_mass), coarse_mass, 1.0e-12_dp, &
            'mass invariant under layer refinement')
        call assert_close(sum(fine_dphi), coarse_dphi, 1.0e-12_dp, &
            'geopotential invariant under isothermal refinement')
        call assert_close(sum(fine_dz), coarse_dz, 1.0e-12_dp, &
            'height invariant under isothermal refinement')
        call assert_close(phi(3), coarse_dphi, 1.0e-12_dp, &
            'cumulative geopotential matches coarse layer')
    end subroutine test_refinement_invariance


    subroutine test_column_mass_telescopes(parameters)
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp) :: pressures(4), temperatures(3)
        real(dp) :: mass(3), dphi(3), dz(3), phi(4)
        real(dp) :: expected_total_mass
        integer :: ierr

        pressures = [100000.0_dp, 80000.0_dp, 50000.0_dp, 20000.0_dp]
        temperatures = [290.0_dp, 270.0_dp, 240.0_dp]
        call integrate_pressure_column(pressures, temperatures, parameters, &
            mass, dphi, dz, phi, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'variable-temperature column status')

        expected_total_mass = (pressures(1) - pressures(4)) / parameters%gravity_m_s2
        call assert_close(sum(mass), expected_total_mass, 1.0e-12_dp, &
            'layer masses telescope to pressure-column mass')
        call assert_true(all(dphi > 0.0_dp), 'all geopotential layers positive')
        call assert_true(all(dz > 0.0_dp), 'all geometric layers positive')
        call assert_true(phi(4) > phi(3) .and. phi(3) > phi(2) .and. &
            phi(2) > phi(1), 'relative geopotential increases upward')
    end subroutine test_column_mass_telescopes


    subroutine test_moist_virtual_temperature_effect(parameters, moist_parameters)
        type(hydrostatic_parameters), intent(in) :: parameters
        type(moist_vapor_parameters), intent(in) :: moist_parameters
        real(dp) :: virtual_temperature, dry_mass, dry_dphi, dry_dz
        real(dp) :: moist_mass, moist_dphi, moist_dz
        integer :: ierr, moist_ierr

        call compute_virtual_temperature_from_mixing_ratio(290.0_dp, 0.02_dp, &
            moist_parameters, virtual_temperature, moist_ierr)
        call assert_int_equal(moist_ierr, MOIST_OK, 'virtual temperature status')
        call assert_true(virtual_temperature > 290.0_dp, &
            'water vapor raises virtual temperature')

        call compute_hydrostatic_layer(100000.0_dp, 80000.0_dp, 290.0_dp, &
            parameters, dry_mass, dry_dphi, dry_dz, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'dry layer status')
        call compute_hydrostatic_layer(100000.0_dp, 80000.0_dp, &
            virtual_temperature, parameters, moist_mass, moist_dphi, moist_dz, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'moist layer status')

        call assert_close(moist_mass, dry_mass, 1.0e-12_dp, &
            'pressure-coordinate layer mass is temperature independent')
        call assert_true(moist_dphi > dry_dphi, &
            'higher virtual temperature increases geopotential thickness')
        call assert_true(moist_dz > dry_dz, &
            'higher virtual temperature increases geometric thickness')
    end subroutine test_moist_virtual_temperature_effect


    subroutine test_invalid_pressure_order(parameters)
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp) :: mass, dphi, dz
        integer :: ierr

        call compute_hydrostatic_layer(80000.0_dp, 90000.0_dp, 280.0_dp, &
            parameters, mass, dphi, dz, ierr)
        call assert_int_equal(ierr, HYDRO_ERR_PRESSURE_ORDER, &
            'pressure interfaces must descend upward')
        call assert_close(mass, 0.0_dp, 0.0_dp, 'failed layer mass cleared')
        call assert_close(dphi, 0.0_dp, 0.0_dp, 'failed layer geopotential cleared')
        call assert_close(dz, 0.0_dp, 0.0_dp, 'failed layer height cleared')
    end subroutine test_invalid_pressure_order


    subroutine test_shape_failure(parameters)
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp) :: pressures(3), temperatures(1)
        real(dp) :: mass(1), dphi(1), dz(1), phi(2)
        integer :: ierr

        pressures = [100000.0_dp, 80000.0_dp, 50000.0_dp]
        temperatures = [280.0_dp]
        call integrate_pressure_column(pressures, temperatures, parameters, &
            mass, dphi, dz, phi, ierr)
        call assert_int_equal(ierr, HYDRO_ERR_SHAPE, &
            'interface count must equal layer count plus one')
    end subroutine test_shape_failure


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

end program test_pressure_coordinate_hydrostatics
