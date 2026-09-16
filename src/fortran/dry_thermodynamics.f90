module climate_dry_thermodynamics
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: THERMO_OK = 0
    integer, parameter, public :: THERMO_ERR_NONFINITE = 1
    integer, parameter, public :: THERMO_ERR_PARAMETERS = 2
    integer, parameter, public :: THERMO_ERR_PRESSURE = 3
    integer, parameter, public :: THERMO_ERR_TEMPERATURE = 4
    integer, parameter, public :: THERMO_ERR_PRESSURE_ORDER = 5
    integer, parameter, public :: THERMO_ERR_RESULT = 6

    type, public :: dry_thermo_parameters
        ! Conventional dry-air reference values in SI. They are defaults, not
        ! hidden universal constants: callers may supply a different parameter
        ! set, and every public operation validates the supplied set.
        real(dp) :: reference_pressure_pa = 100000.0_dp
        real(dp) :: gas_constant_j_kg_k = 287.05_dp
        real(dp) :: heat_capacity_cp_j_kg_k = 1004.0_dp
        real(dp) :: gravity_m_s2 = 9.80665_dp
    end type dry_thermo_parameters

    public :: compute_exner
    public :: compute_potential_temperature
    public :: compute_temperature_from_potential
    public :: compute_dry_air_density
    public :: compute_isothermal_hydrostatic_thickness

contains

    logical function valid_parameters(parameters)
        type(dry_thermo_parameters), intent(in) :: parameters

        valid_parameters = &
            ieee_is_finite(parameters%reference_pressure_pa) .and. &
            ieee_is_finite(parameters%gas_constant_j_kg_k) .and. &
            ieee_is_finite(parameters%heat_capacity_cp_j_kg_k) .and. &
            ieee_is_finite(parameters%gravity_m_s2) .and. &
            parameters%reference_pressure_pa > 0.0_dp .and. &
            parameters%gas_constant_j_kg_k > 0.0_dp .and. &
            parameters%heat_capacity_cp_j_kg_k > parameters%gas_constant_j_kg_k .and. &
            parameters%gravity_m_s2 > 0.0_dp
    end function valid_parameters


    subroutine compute_exner(pressure_pa, parameters, exner, ierr)
        real(dp), intent(in) :: pressure_pa
        type(dry_thermo_parameters), intent(in) :: parameters
        real(dp), intent(out) :: exner
        integer, intent(out) :: ierr

        real(dp) :: kappa

        exner = 0.0_dp
        ierr = THERMO_OK

        if (.not. valid_parameters(parameters)) then
            ierr = THERMO_ERR_PARAMETERS
            return
        end if
        if (.not. ieee_is_finite(pressure_pa)) then
            ierr = THERMO_ERR_NONFINITE
            return
        end if
        if (pressure_pa <= 0.0_dp) then
            ierr = THERMO_ERR_PRESSURE
            return
        end if

        kappa = parameters%gas_constant_j_kg_k / parameters%heat_capacity_cp_j_kg_k
        exner = (pressure_pa / parameters%reference_pressure_pa)**kappa

        if (.not. ieee_is_finite(exner) .or. exner <= 0.0_dp) then
            exner = 0.0_dp
            ierr = THERMO_ERR_RESULT
        end if
    end subroutine compute_exner


    subroutine compute_potential_temperature(temperature_k, pressure_pa, parameters, &
                                             potential_temperature_k, ierr)
        real(dp), intent(in) :: temperature_k, pressure_pa
        type(dry_thermo_parameters), intent(in) :: parameters
        real(dp), intent(out) :: potential_temperature_k
        integer, intent(out) :: ierr

        real(dp) :: exner

        potential_temperature_k = 0.0_dp
        ierr = THERMO_OK

        if (.not. ieee_is_finite(temperature_k)) then
            ierr = THERMO_ERR_NONFINITE
            return
        end if
        if (temperature_k <= 0.0_dp) then
            ierr = THERMO_ERR_TEMPERATURE
            return
        end if

        call compute_exner(pressure_pa, parameters, exner, ierr)
        if (ierr /= THERMO_OK) return

        potential_temperature_k = temperature_k / exner
        if (.not. ieee_is_finite(potential_temperature_k) .or. potential_temperature_k <= 0.0_dp) then
            potential_temperature_k = 0.0_dp
            ierr = THERMO_ERR_RESULT
        end if
    end subroutine compute_potential_temperature


    subroutine compute_temperature_from_potential(potential_temperature_k, pressure_pa, parameters, &
                                                  temperature_k, ierr)
        real(dp), intent(in) :: potential_temperature_k, pressure_pa
        type(dry_thermo_parameters), intent(in) :: parameters
        real(dp), intent(out) :: temperature_k
        integer, intent(out) :: ierr

        real(dp) :: exner

        temperature_k = 0.0_dp
        ierr = THERMO_OK

        if (.not. ieee_is_finite(potential_temperature_k)) then
            ierr = THERMO_ERR_NONFINITE
            return
        end if
        if (potential_temperature_k <= 0.0_dp) then
            ierr = THERMO_ERR_TEMPERATURE
            return
        end if

        call compute_exner(pressure_pa, parameters, exner, ierr)
        if (ierr /= THERMO_OK) return

        temperature_k = potential_temperature_k * exner
        if (.not. ieee_is_finite(temperature_k) .or. temperature_k <= 0.0_dp) then
            temperature_k = 0.0_dp
            ierr = THERMO_ERR_RESULT
        end if
    end subroutine compute_temperature_from_potential


    subroutine compute_dry_air_density(pressure_pa, temperature_k, parameters, density_kg_m3, ierr)
        real(dp), intent(in) :: pressure_pa, temperature_k
        type(dry_thermo_parameters), intent(in) :: parameters
        real(dp), intent(out) :: density_kg_m3
        integer, intent(out) :: ierr

        density_kg_m3 = 0.0_dp
        ierr = THERMO_OK

        if (.not. valid_parameters(parameters)) then
            ierr = THERMO_ERR_PARAMETERS
            return
        end if
        if (.not. ieee_is_finite(pressure_pa) .or. .not. ieee_is_finite(temperature_k)) then
            ierr = THERMO_ERR_NONFINITE
            return
        end if
        if (pressure_pa <= 0.0_dp) then
            ierr = THERMO_ERR_PRESSURE
            return
        end if
        if (temperature_k <= 0.0_dp) then
            ierr = THERMO_ERR_TEMPERATURE
            return
        end if

        density_kg_m3 = pressure_pa / (parameters%gas_constant_j_kg_k * temperature_k)
        if (.not. ieee_is_finite(density_kg_m3) .or. density_kg_m3 <= 0.0_dp) then
            density_kg_m3 = 0.0_dp
            ierr = THERMO_ERR_RESULT
        end if
    end subroutine compute_dry_air_density


    subroutine compute_isothermal_hydrostatic_thickness(lower_pressure_pa, upper_pressure_pa, &
                                                        temperature_k, parameters, thickness_m, ierr)
        real(dp), intent(in) :: lower_pressure_pa, upper_pressure_pa, temperature_k
        type(dry_thermo_parameters), intent(in) :: parameters
        real(dp), intent(out) :: thickness_m
        integer, intent(out) :: ierr

        thickness_m = 0.0_dp
        ierr = THERMO_OK

        if (.not. valid_parameters(parameters)) then
            ierr = THERMO_ERR_PARAMETERS
            return
        end if
        if (.not. ieee_is_finite(lower_pressure_pa) .or. &
            .not. ieee_is_finite(upper_pressure_pa) .or. &
            .not. ieee_is_finite(temperature_k)) then
            ierr = THERMO_ERR_NONFINITE
            return
        end if
        if (lower_pressure_pa <= 0.0_dp .or. upper_pressure_pa <= 0.0_dp) then
            ierr = THERMO_ERR_PRESSURE
            return
        end if
        if (lower_pressure_pa <= upper_pressure_pa) then
            ierr = THERMO_ERR_PRESSURE_ORDER
            return
        end if
        if (temperature_k <= 0.0_dp) then
            ierr = THERMO_ERR_TEMPERATURE
            return
        end if

        thickness_m = parameters%gas_constant_j_kg_k * temperature_k / parameters%gravity_m_s2 * &
            log(lower_pressure_pa / upper_pressure_pa)
        if (.not. ieee_is_finite(thickness_m) .or. thickness_m <= 0.0_dp) then
            thickness_m = 0.0_dp
            ierr = THERMO_ERR_RESULT
        end if
    end subroutine compute_isothermal_hydrostatic_thickness

end module climate_dry_thermodynamics
