module climate_saturation_moisture
    use, intrinsic :: iso_fortran_env, only: real64
    use climate_saturation_vapor_pressure, only: compute_saturation_vapor_pressure, SAT_OK
    use climate_moist_vapor_algebra, only: moist_vapor_parameters, &
        compute_mixing_ratio_from_vapor_pressure, compute_specific_humidity_from_mixing_ratio, MOIST_OK
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: SATMOIST_OK = 0
    integer, parameter, public :: SATMOIST_ERR_SATURATION_PROVIDER = 1
    integer, parameter, public :: SATMOIST_ERR_MIXING_RATIO = 2
    integer, parameter, public :: SATMOIST_ERR_SPECIFIC_HUMIDITY = 3

    type, public :: saturation_moisture_state
        real(dp) :: vapor_pressure_pa = 0.0_dp
        real(dp) :: mixing_ratio = 0.0_dp
        real(dp) :: specific_humidity = 0.0_dp
        integer :: saturation_ierr = SAT_OK
        integer :: mixing_ratio_ierr = MOIST_OK
        integer :: specific_humidity_ierr = MOIST_OK
    end type saturation_moisture_state

    public :: compute_saturation_moisture_state

contains

    subroutine compute_saturation_moisture_state(temperature_k, total_pressure_pa, phase, parameters, state, ierr)
        real(dp), intent(in) :: temperature_k, total_pressure_pa
        integer, intent(in) :: phase
        type(moist_vapor_parameters), intent(in) :: parameters
        type(saturation_moisture_state), intent(out) :: state
        integer, intent(out) :: ierr

        state = saturation_moisture_state()
        ierr = SATMOIST_OK

        call compute_saturation_vapor_pressure(temperature_k, phase, state%vapor_pressure_pa, state%saturation_ierr)
        if (state%saturation_ierr /= SAT_OK) then
            ierr = SATMOIST_ERR_SATURATION_PROVIDER
            return
        end if

        call compute_mixing_ratio_from_vapor_pressure(state%vapor_pressure_pa, total_pressure_pa, &
                                                      parameters, state%mixing_ratio, state%mixing_ratio_ierr)
        if (state%mixing_ratio_ierr /= MOIST_OK) then
            ierr = SATMOIST_ERR_MIXING_RATIO
            return
        end if

        call compute_specific_humidity_from_mixing_ratio(state%mixing_ratio, &
                                                         state%specific_humidity, state%specific_humidity_ierr)
        if (state%specific_humidity_ierr /= MOIST_OK) then
            ierr = SATMOIST_ERR_SPECIFIC_HUMIDITY
            return
        end if
    end subroutine compute_saturation_moisture_state

end module climate_saturation_moisture
