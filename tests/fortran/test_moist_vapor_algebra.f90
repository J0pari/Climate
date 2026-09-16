program test_moist_vapor_algebra
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_moist_vapor_algebra, only: dp, moist_vapor_parameters, MOIST_OK, &
        MOIST_ERR_NONFINITE, MOIST_ERR_PARAMETERS, MOIST_ERR_PARTIAL_PRESSURE, &
        MOIST_ERR_MIXING_RATIO, MOIST_ERR_SPECIFIC_HUMIDITY, &
        compute_mixing_ratio_from_vapor_pressure, compute_vapor_pressure_from_mixing_ratio, &
        compute_specific_humidity_from_mixing_ratio, compute_mixing_ratio_from_specific_humidity, &
        compute_virtual_temperature_from_mixing_ratio, compute_moist_vapor_density
    use climate_dry_thermodynamics, only: dry_thermo_parameters, compute_dry_air_density
    implicit none

    call test_pressure_mixing_ratio_round_trip()
    call test_specific_humidity_round_trip()
    call test_dry_virtual_temperature_limit()
    call test_virtual_temperature_mixture_identity()
    call test_density_cross_module_dry_limit()
    call test_vapor_reduces_density()
    call test_parameter_sensitivity()
    call test_invalid_inputs_fail_closed()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_pressure_mixing_ratio_round_trip()
        type(moist_vapor_parameters) :: parameters
        real(dp) :: mixing_ratio, recovered_vapor_pressure
        integer :: ierr

        call compute_mixing_ratio_from_vapor_pressure(2000.0_dp, 90000.0_dp, parameters, mixing_ratio, ierr)
        call require(ierr == MOIST_OK, 'vapor-pressure to mixing-ratio transform must succeed')
        call compute_vapor_pressure_from_mixing_ratio(mixing_ratio, 90000.0_dp, parameters, &
                                                      recovered_vapor_pressure, ierr)
        call require(ierr == MOIST_OK, 'mixing-ratio to vapor-pressure transform must succeed')
        call require(abs(recovered_vapor_pressure - 2000.0_dp) < 2.0e-13_dp * 2000.0_dp, &
                     'vapor pressure and mixing ratio transforms must round trip')
    end subroutine test_pressure_mixing_ratio_round_trip


    subroutine test_specific_humidity_round_trip()
        real(dp) :: specific_humidity, recovered_mixing_ratio
        integer :: ierr

        call compute_specific_humidity_from_mixing_ratio(0.018_dp, specific_humidity, ierr)
        call require(ierr == MOIST_OK, 'mixing-ratio to specific-humidity transform must succeed')
        call compute_mixing_ratio_from_specific_humidity(specific_humidity, recovered_mixing_ratio, ierr)
        call require(ierr == MOIST_OK, 'specific-humidity to mixing-ratio transform must succeed')
        call require(abs(recovered_mixing_ratio - 0.018_dp) < 2.0e-15_dp, &
                     'mixing ratio and specific humidity transforms must round trip')
    end subroutine test_specific_humidity_round_trip


    subroutine test_dry_virtual_temperature_limit()
        type(moist_vapor_parameters) :: parameters
        real(dp) :: virtual_temperature
        integer :: ierr

        call compute_virtual_temperature_from_mixing_ratio(290.0_dp, 0.0_dp, parameters, &
                                                           virtual_temperature, ierr)
        call require(ierr == MOIST_OK, 'dry virtual-temperature limit must succeed')
        call require(abs(virtual_temperature - 290.0_dp) < 1.0e-13_dp, &
                     'virtual temperature must equal physical temperature in the dry limit')
    end subroutine test_dry_virtual_temperature_limit


    subroutine test_virtual_temperature_mixture_identity()
        type(moist_vapor_parameters) :: parameters
        real(dp), parameter :: temperature_k = 300.0_dp, mixing_ratio = 0.020_dp
        real(dp) :: virtual_temperature, specific_humidity, mixture_gas_constant, expected
        integer :: ierr

        call compute_virtual_temperature_from_mixing_ratio(temperature_k, mixing_ratio, parameters, &
                                                           virtual_temperature, ierr)
        call require(ierr == MOIST_OK, 'moist virtual temperature must succeed')
        call compute_specific_humidity_from_mixing_ratio(mixing_ratio, specific_humidity, ierr)
        call require(ierr == MOIST_OK, 'specific humidity for mixture identity must succeed')

        mixture_gas_constant = (1.0_dp - specific_humidity) * parameters%dry_air_gas_constant_j_kg_k + &
            specific_humidity * parameters%water_vapor_gas_constant_j_kg_k
        expected = temperature_k * mixture_gas_constant / parameters%dry_air_gas_constant_j_kg_k
        call require(abs(virtual_temperature - expected) < 2.0e-13_dp * expected, &
                     'virtual-temperature formula must equal the ideal-gas mixture identity')
        call require(virtual_temperature > temperature_k, &
                     'water vapor must raise virtual temperature relative to dry air at fixed T')
    end subroutine test_virtual_temperature_mixture_identity


    subroutine test_density_cross_module_dry_limit()
        type(moist_vapor_parameters) :: moist_parameters
        type(dry_thermo_parameters) :: dry_parameters
        real(dp) :: moist_density, dry_density
        integer :: ierr

        dry_parameters%gas_constant_j_kg_k = moist_parameters%dry_air_gas_constant_j_kg_k
        call compute_moist_vapor_density(95000.0_dp, 285.0_dp, 0.0_dp, moist_parameters, moist_density, ierr)
        call require(ierr == MOIST_OK, 'moist-density dry limit must succeed')
        call compute_dry_air_density(95000.0_dp, 285.0_dp, dry_parameters, dry_density, ierr)
        call require(ierr == 0, 'canonical dry-density comparison must succeed')
        call require(abs(moist_density - dry_density) < 2.0e-14_dp * dry_density, &
                     'moist-vapor density must reduce to canonical dry density when mixing ratio is zero')
    end subroutine test_density_cross_module_dry_limit


    subroutine test_vapor_reduces_density()
        type(moist_vapor_parameters) :: parameters
        real(dp) :: dry_density, moist_density
        integer :: ierr

        call compute_moist_vapor_density(100000.0_dp, 300.0_dp, 0.0_dp, parameters, dry_density, ierr)
        call require(ierr == MOIST_OK, 'dry density through moist module must succeed')
        call compute_moist_vapor_density(100000.0_dp, 300.0_dp, 0.025_dp, parameters, moist_density, ierr)
        call require(ierr == MOIST_OK, 'moist density must succeed')
        call require(moist_density < dry_density, &
                     'adding water vapor at fixed pressure and temperature must reduce ideal-gas density')
    end subroutine test_vapor_reduces_density


    subroutine test_parameter_sensitivity()
        type(moist_vapor_parameters) :: baseline, modified
        real(dp) :: baseline_ratio, modified_ratio
        integer :: ierr

        modified = baseline
        modified%water_vapor_gas_constant_j_kg_k = 480.0_dp
        call compute_mixing_ratio_from_vapor_pressure(2500.0_dp, 100000.0_dp, baseline, baseline_ratio, ierr)
        call require(ierr == MOIST_OK, 'baseline gas constants must succeed')
        call compute_mixing_ratio_from_vapor_pressure(2500.0_dp, 100000.0_dp, modified, modified_ratio, ierr)
        call require(ierr == MOIST_OK, 'modified vapor gas constant must succeed')
        call require(abs(modified_ratio - baseline_ratio) > 1.0e-8_dp, &
                     'declared gas constants must demonstrably affect the vapor-pressure transform')
    end subroutine test_parameter_sensitivity


    subroutine test_invalid_inputs_fail_closed()
        type(moist_vapor_parameters) :: parameters, invalid_parameters
        real(dp) :: result, nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
        call compute_mixing_ratio_from_vapor_pressure(nan_value, 100000.0_dp, parameters, result, ierr)
        call require(ierr == MOIST_ERR_NONFINITE .and. abs(result) < tiny(1.0_dp), &
                     'non-finite vapor pressure must fail closed')

        call compute_mixing_ratio_from_vapor_pressure(100000.0_dp, 100000.0_dp, parameters, result, ierr)
        call require(ierr == MOIST_ERR_PARTIAL_PRESSURE .and. abs(result) < tiny(1.0_dp), &
                     'vapor pressure at or above total pressure must fail closed')

        call compute_specific_humidity_from_mixing_ratio(-0.1_dp, result, ierr)
        call require(ierr == MOIST_ERR_MIXING_RATIO .and. abs(result) < tiny(1.0_dp), &
                     'negative mixing ratio must fail closed')

        call compute_mixing_ratio_from_specific_humidity(1.0_dp, result, ierr)
        call require(ierr == MOIST_ERR_SPECIFIC_HUMIDITY .and. abs(result) < tiny(1.0_dp), &
                     'specific humidity outside [0,1) must fail closed')

        invalid_parameters = parameters
        invalid_parameters%water_vapor_gas_constant_j_kg_k = &
            invalid_parameters%dry_air_gas_constant_j_kg_k
        call compute_virtual_temperature_from_mixing_ratio(280.0_dp, 0.01_dp, invalid_parameters, result, ierr)
        call require(ierr == MOIST_ERR_PARAMETERS .and. abs(result) < tiny(1.0_dp), &
                     'invalid gas-constant ordering must fail closed')
    end subroutine test_invalid_inputs_fail_closed

end program test_moist_vapor_algebra
