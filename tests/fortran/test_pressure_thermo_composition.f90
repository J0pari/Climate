program test_pressure_thermo_composition
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use climate_dry_thermodynamics, only: &
        dry_thermo_parameters, compute_dry_air_density, &
        compute_isothermal_hydrostatic_thickness, THERMO_OK
    use climate_moist_vapor_algebra, only: &
        moist_vapor_parameters, compute_virtual_temperature_from_mixing_ratio, &
        compute_moist_vapor_density, MOIST_OK
    use climate_pressure_coordinate_hydrostatics, only: &
        hydrostatic_parameters, compute_hydrostatic_layer, HYDRO_OK
    implicit none

    integer, parameter :: dp = real64

    call test_default_parameter_authority_is_composable()
    call test_zero_vapor_density_matches_dry_density()
    call test_isothermal_hydrostatic_thickness_matches_pressure_kernel()

contains

    subroutine test_default_parameter_authority_is_composable()
        type(dry_thermo_parameters) :: dry
        type(moist_vapor_parameters) :: moist
        type(hydrostatic_parameters) :: hydro

        dry = dry_thermo_parameters()
        moist = moist_vapor_parameters()
        hydro = hydrostatic_parameters()

        call assert_close(dry%gas_constant_j_kg_k, &
            moist%dry_air_gas_constant_j_kg_k, 0.0_dp, &
            'dry-air gas constant agrees across default parameter sets')
        call assert_close(dry%gas_constant_j_kg_k, &
            hydro%dry_air_gas_constant_j_kg_k, 0.0_dp, &
            'hydrostatic gas constant agrees with dry thermodynamics')
        call assert_close(dry%gravity_m_s2, hydro%gravity_m_s2, 0.0_dp, &
            'gravity agrees across default parameter sets')
    end subroutine test_default_parameter_authority_is_composable


    subroutine test_zero_vapor_density_matches_dry_density()
        type(dry_thermo_parameters) :: dry
        type(moist_vapor_parameters) :: moist
        real(dp) :: pressure_pa, temperature_k
        real(dp) :: dry_density, moist_density, virtual_temperature
        integer :: ierr

        dry = dry_thermo_parameters()
        moist = moist_vapor_parameters()
        pressure_pa = 85000.0_dp
        temperature_k = 280.0_dp

        call compute_virtual_temperature_from_mixing_ratio( &
            temperature_k, 0.0_dp, moist, virtual_temperature, ierr)
        call assert_int_equal(ierr, MOIST_OK, 'zero-vapor virtual temperature status')
        call assert_close(virtual_temperature, temperature_k, 2.0e-15_dp, &
            'zero-vapor virtual temperature reduces to dry temperature')

        call compute_dry_air_density(pressure_pa, temperature_k, dry, &
            dry_density, ierr)
        call assert_int_equal(ierr, THERMO_OK, 'dry density status')

        call compute_moist_vapor_density(pressure_pa, temperature_k, 0.0_dp, &
            moist, moist_density, ierr)
        call assert_int_equal(ierr, MOIST_OK, 'zero-vapor moist density status')
        call assert_close(moist_density, dry_density, 2.0e-15_dp, &
            'zero-vapor moist density reduces to dry density')
    end subroutine test_zero_vapor_density_matches_dry_density


    subroutine test_isothermal_hydrostatic_thickness_matches_pressure_kernel()
        type(dry_thermo_parameters) :: dry
        type(hydrostatic_parameters) :: hydro
        real(dp) :: lower_pressure_pa, upper_pressure_pa, temperature_k
        real(dp) :: dry_thickness_m, mass_per_area_kg_m2
        real(dp) :: geopotential_thickness_m2_s2, hydro_thickness_m
        integer :: ierr

        dry = dry_thermo_parameters()
        hydro = hydrostatic_parameters()
        lower_pressure_pa = 100000.0_dp
        upper_pressure_pa = 70000.0_dp
        temperature_k = 275.0_dp

        call compute_isothermal_hydrostatic_thickness( &
            lower_pressure_pa, upper_pressure_pa, temperature_k, dry, &
            dry_thickness_m, ierr)
        call assert_int_equal(ierr, THERMO_OK, 'dry hydrostatic thickness status')

        call compute_hydrostatic_layer( &
            lower_pressure_pa, upper_pressure_pa, temperature_k, hydro, &
            mass_per_area_kg_m2, geopotential_thickness_m2_s2, &
            hydro_thickness_m, ierr)
        call assert_int_equal(ierr, HYDRO_OK, 'pressure hydrostatic layer status')
        call assert_close(hydro_thickness_m, dry_thickness_m, 2.0e-15_dp, &
            'isothermal thickness agrees across thermodynamic and pressure kernels')
    end subroutine test_isothermal_hydrostatic_thickness_matches_pressure_kernel


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

end program test_pressure_thermo_composition
