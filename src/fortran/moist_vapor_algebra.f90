module climate_moist_vapor_algebra
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: MOIST_OK = 0
    integer, parameter, public :: MOIST_ERR_NONFINITE = 1
    integer, parameter, public :: MOIST_ERR_PARAMETERS = 2
    integer, parameter, public :: MOIST_ERR_PRESSURE = 3
    integer, parameter, public :: MOIST_ERR_PARTIAL_PRESSURE = 4
    integer, parameter, public :: MOIST_ERR_MIXING_RATIO = 5
    integer, parameter, public :: MOIST_ERR_SPECIFIC_HUMIDITY = 6
    integer, parameter, public :: MOIST_ERR_TEMPERATURE = 7
    integer, parameter, public :: MOIST_ERR_RESULT = 8

    type, public :: moist_vapor_parameters
        ! Conventional ideal-gas reference values in SI. The ratio Rd/Rv is
        ! derived at runtime so callers may use a different declared parameter
        ! set without hidden duplicate constants.
        real(dp) :: dry_air_gas_constant_j_kg_k = 287.05_dp
        real(dp) :: water_vapor_gas_constant_j_kg_k = 461.5_dp
    end type moist_vapor_parameters

    public :: compute_mixing_ratio_from_vapor_pressure
    public :: compute_vapor_pressure_from_mixing_ratio
    public :: compute_specific_humidity_from_mixing_ratio
    public :: compute_mixing_ratio_from_specific_humidity
    public :: compute_virtual_temperature_from_mixing_ratio
    public :: compute_moist_vapor_density

contains

    logical function valid_parameters(parameters)
        type(moist_vapor_parameters), intent(in) :: parameters

        valid_parameters = &
            ieee_is_finite(parameters%dry_air_gas_constant_j_kg_k) .and. &
            ieee_is_finite(parameters%water_vapor_gas_constant_j_kg_k) .and. &
            parameters%dry_air_gas_constant_j_kg_k > 0.0_dp .and. &
            parameters%water_vapor_gas_constant_j_kg_k > parameters%dry_air_gas_constant_j_kg_k
    end function valid_parameters


    subroutine compute_mixing_ratio_from_vapor_pressure(vapor_pressure_pa, total_pressure_pa, &
                                                        parameters, mixing_ratio, ierr)
        real(dp), intent(in) :: vapor_pressure_pa, total_pressure_pa
        type(moist_vapor_parameters), intent(in) :: parameters
        real(dp), intent(out) :: mixing_ratio
        integer, intent(out) :: ierr

        real(dp) :: epsilon

        mixing_ratio = 0.0_dp
        ierr = MOIST_OK

        if (.not. valid_parameters(parameters)) then
            ierr = MOIST_ERR_PARAMETERS
            return
        end if
        if (.not. ieee_is_finite(vapor_pressure_pa) .or. .not. ieee_is_finite(total_pressure_pa)) then
            ierr = MOIST_ERR_NONFINITE
            return
        end if
        if (total_pressure_pa <= 0.0_dp) then
            ierr = MOIST_ERR_PRESSURE
            return
        end if
        if (vapor_pressure_pa < 0.0_dp .or. vapor_pressure_pa >= total_pressure_pa) then
            ierr = MOIST_ERR_PARTIAL_PRESSURE
            return
        end if

        epsilon = parameters%dry_air_gas_constant_j_kg_k / parameters%water_vapor_gas_constant_j_kg_k
        mixing_ratio = epsilon * vapor_pressure_pa / (total_pressure_pa - vapor_pressure_pa)

        if (.not. ieee_is_finite(mixing_ratio) .or. mixing_ratio < 0.0_dp) then
            mixing_ratio = 0.0_dp
            ierr = MOIST_ERR_RESULT
        end if
    end subroutine compute_mixing_ratio_from_vapor_pressure


    subroutine compute_vapor_pressure_from_mixing_ratio(mixing_ratio, total_pressure_pa, &
                                                        parameters, vapor_pressure_pa, ierr)
        real(dp), intent(in) :: mixing_ratio, total_pressure_pa
        type(moist_vapor_parameters), intent(in) :: parameters
        real(dp), intent(out) :: vapor_pressure_pa
        integer, intent(out) :: ierr

        real(dp) :: epsilon

        vapor_pressure_pa = 0.0_dp
        ierr = MOIST_OK

        if (.not. valid_parameters(parameters)) then
            ierr = MOIST_ERR_PARAMETERS
            return
        end if
        if (.not. ieee_is_finite(mixing_ratio) .or. .not. ieee_is_finite(total_pressure_pa)) then
            ierr = MOIST_ERR_NONFINITE
            return
        end if
        if (total_pressure_pa <= 0.0_dp) then
            ierr = MOIST_ERR_PRESSURE
            return
        end if
        if (mixing_ratio < 0.0_dp) then
            ierr = MOIST_ERR_MIXING_RATIO
            return
        end if

        epsilon = parameters%dry_air_gas_constant_j_kg_k / parameters%water_vapor_gas_constant_j_kg_k
        vapor_pressure_pa = total_pressure_pa * mixing_ratio / (epsilon + mixing_ratio)

        if (.not. ieee_is_finite(vapor_pressure_pa) .or. vapor_pressure_pa < 0.0_dp .or. &
            vapor_pressure_pa >= total_pressure_pa) then
            vapor_pressure_pa = 0.0_dp
            ierr = MOIST_ERR_RESULT
        end if
    end subroutine compute_vapor_pressure_from_mixing_ratio


    subroutine compute_specific_humidity_from_mixing_ratio(mixing_ratio, specific_humidity, ierr)
        real(dp), intent(in) :: mixing_ratio
        real(dp), intent(out) :: specific_humidity
        integer, intent(out) :: ierr

        specific_humidity = 0.0_dp
        ierr = MOIST_OK

        if (.not. ieee_is_finite(mixing_ratio)) then
            ierr = MOIST_ERR_NONFINITE
            return
        end if
        if (mixing_ratio < 0.0_dp) then
            ierr = MOIST_ERR_MIXING_RATIO
            return
        end if

        specific_humidity = mixing_ratio / (1.0_dp + mixing_ratio)
        if (.not. ieee_is_finite(specific_humidity) .or. &
            specific_humidity < 0.0_dp .or. specific_humidity >= 1.0_dp) then
            specific_humidity = 0.0_dp
            ierr = MOIST_ERR_RESULT
        end if
    end subroutine compute_specific_humidity_from_mixing_ratio


    subroutine compute_mixing_ratio_from_specific_humidity(specific_humidity, mixing_ratio, ierr)
        real(dp), intent(in) :: specific_humidity
        real(dp), intent(out) :: mixing_ratio
        integer, intent(out) :: ierr

        mixing_ratio = 0.0_dp
        ierr = MOIST_OK

        if (.not. ieee_is_finite(specific_humidity)) then
            ierr = MOIST_ERR_NONFINITE
            return
        end if
        if (specific_humidity < 0.0_dp .or. specific_humidity >= 1.0_dp) then
            ierr = MOIST_ERR_SPECIFIC_HUMIDITY
            return
        end if

        mixing_ratio = specific_humidity / (1.0_dp - specific_humidity)
        if (.not. ieee_is_finite(mixing_ratio) .or. mixing_ratio < 0.0_dp) then
            mixing_ratio = 0.0_dp
            ierr = MOIST_ERR_RESULT
        end if
    end subroutine compute_mixing_ratio_from_specific_humidity


    subroutine compute_virtual_temperature_from_mixing_ratio(temperature_k, mixing_ratio, parameters, &
                                                             virtual_temperature_k, ierr)
        real(dp), intent(in) :: temperature_k, mixing_ratio
        type(moist_vapor_parameters), intent(in) :: parameters
        real(dp), intent(out) :: virtual_temperature_k
        integer, intent(out) :: ierr

        real(dp) :: epsilon

        virtual_temperature_k = 0.0_dp
        ierr = MOIST_OK

        if (.not. valid_parameters(parameters)) then
            ierr = MOIST_ERR_PARAMETERS
            return
        end if
        if (.not. ieee_is_finite(temperature_k) .or. .not. ieee_is_finite(mixing_ratio)) then
            ierr = MOIST_ERR_NONFINITE
            return
        end if
        if (temperature_k <= 0.0_dp) then
            ierr = MOIST_ERR_TEMPERATURE
            return
        end if
        if (mixing_ratio < 0.0_dp) then
            ierr = MOIST_ERR_MIXING_RATIO
            return
        end if

        epsilon = parameters%dry_air_gas_constant_j_kg_k / parameters%water_vapor_gas_constant_j_kg_k
        virtual_temperature_k = temperature_k * (mixing_ratio + epsilon) / &
            (epsilon * (1.0_dp + mixing_ratio))

        if (.not. ieee_is_finite(virtual_temperature_k) .or. virtual_temperature_k <= 0.0_dp) then
            virtual_temperature_k = 0.0_dp
            ierr = MOIST_ERR_RESULT
        end if
    end subroutine compute_virtual_temperature_from_mixing_ratio


    subroutine compute_moist_vapor_density(total_pressure_pa, temperature_k, mixing_ratio, parameters, &
                                           density_kg_m3, ierr)
        real(dp), intent(in) :: total_pressure_pa, temperature_k, mixing_ratio
        type(moist_vapor_parameters), intent(in) :: parameters
        real(dp), intent(out) :: density_kg_m3
        integer, intent(out) :: ierr

        real(dp) :: virtual_temperature_k

        density_kg_m3 = 0.0_dp
        ierr = MOIST_OK

        if (.not. ieee_is_finite(total_pressure_pa)) then
            ierr = MOIST_ERR_NONFINITE
            return
        end if
        if (total_pressure_pa <= 0.0_dp) then
            ierr = MOIST_ERR_PRESSURE
            return
        end if

        call compute_virtual_temperature_from_mixing_ratio(temperature_k, mixing_ratio, parameters, &
                                                           virtual_temperature_k, ierr)
        if (ierr /= MOIST_OK) return

        density_kg_m3 = total_pressure_pa / &
            (parameters%dry_air_gas_constant_j_kg_k * virtual_temperature_k)
        if (.not. ieee_is_finite(density_kg_m3) .or. density_kg_m3 <= 0.0_dp) then
            density_kg_m3 = 0.0_dp
            ierr = MOIST_ERR_RESULT
        end if
    end subroutine compute_moist_vapor_density

end module climate_moist_vapor_algebra
