module climate_saturation_vapor_pressure
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: SAT_OK = 0
    integer, parameter, public :: SAT_ERR_NONFINITE = 1
    integer, parameter, public :: SAT_ERR_PHASE = 2
    integer, parameter, public :: SAT_ERR_TEMPERATURE_RANGE = 3
    integer, parameter, public :: SAT_ERR_RESULT = 4

    integer, parameter, public :: SAT_PHASE_LIQUID = 1
    integer, parameter, public :: SAT_PHASE_ICE = 2

    real(dp), parameter, public :: WATER_TRIPLE_POINT_K = 273.16_dp
    real(dp), parameter, public :: LIQUID_MIN_TEMPERATURE_K = 123.0_dp
    real(dp), parameter, public :: LIQUID_MAX_TEMPERATURE_K = 332.0_dp
    real(dp), parameter, public :: ICE_MIN_TEMPERATURE_K = 110.0_dp
    real(dp), parameter, public :: ICE_MAX_TEMPERATURE_K = WATER_TRIPLE_POINT_K

    public :: compute_saturation_vapor_pressure

contains

    subroutine compute_saturation_vapor_pressure(temperature_k, phase, vapor_pressure_pa, ierr)
        real(dp), intent(in) :: temperature_k
        integer, intent(in) :: phase
        real(dp), intent(out) :: vapor_pressure_pa
        integer, intent(out) :: ierr

        real(dp) :: log_pressure

        vapor_pressure_pa = 0.0_dp
        ierr = SAT_OK

        if (.not. ieee_is_finite(temperature_k)) then
            ierr = SAT_ERR_NONFINITE
            return
        end if

        select case (phase)
        case (SAT_PHASE_LIQUID)
            ! Murphy & Koop (2005), QJRMS 131, 1539-1565, Eq. (10),
            ! doi:10.1256/qj.04.94. The published validity interval is
            ! 123 K < T < 332 K. This includes supercooled liquid water.
            if (temperature_k <= LIQUID_MIN_TEMPERATURE_K .or. &
                temperature_k >= LIQUID_MAX_TEMPERATURE_K) then
                ierr = SAT_ERR_TEMPERATURE_RANGE
                return
            end if

            log_pressure = 54.842763_dp - 6763.22_dp / temperature_k - &
                4.210_dp * log(temperature_k) + 0.000367_dp * temperature_k + &
                tanh(0.0415_dp * (temperature_k - 218.8_dp)) * ( &
                    53.878_dp - 1331.22_dp / temperature_k - &
                    9.44523_dp * log(temperature_k) + 0.014025_dp * temperature_k)

        case (SAT_PHASE_ICE)
            ! Murphy & Koop (2005), Eq. (7), for hexagonal ice, documented
            ! for T > 110 K. The canonical API additionally restricts use to
            ! T <= the water triple point: selecting metastable/non-stable ice
            ! above that boundary must be an explicit higher-level policy.
            if (temperature_k <= ICE_MIN_TEMPERATURE_K .or. &
                temperature_k > ICE_MAX_TEMPERATURE_K) then
                ierr = SAT_ERR_TEMPERATURE_RANGE
                return
            end if

            log_pressure = 9.550426_dp - 5723.265_dp / temperature_k + &
                3.53068_dp * log(temperature_k) - 0.00728332_dp * temperature_k

        case default
            ierr = SAT_ERR_PHASE
            return
        end select

        vapor_pressure_pa = exp(log_pressure)
        if (.not. ieee_is_finite(vapor_pressure_pa) .or. vapor_pressure_pa <= 0.0_dp) then
            vapor_pressure_pa = 0.0_dp
            ierr = SAT_ERR_RESULT
        end if
    end subroutine compute_saturation_vapor_pressure

end module climate_saturation_vapor_pressure
