program test_saturation_vapor_pressure
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_saturation_vapor_pressure, only: dp, SAT_OK, SAT_ERR_NONFINITE, &
        SAT_ERR_PHASE, SAT_ERR_TEMPERATURE_RANGE, SAT_PHASE_LIQUID, SAT_PHASE_ICE, &
        WATER_TRIPLE_POINT_K, compute_saturation_vapor_pressure
    use climate_moist_vapor_algebra, only: moist_vapor_parameters, MOIST_OK, &
        compute_mixing_ratio_from_vapor_pressure, compute_vapor_pressure_from_mixing_ratio
    implicit none

    call test_published_formula_fixtures()
    call test_monotonicity()
    call test_supercooled_phase_difference()
    call test_triple_point_agreement()
    call test_composition_with_vapor_algebra()
    call test_phase_and_range_fail_closed()
    call test_nonfinite_fails_closed()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require

    subroutine require_relative(actual, expected, tolerance, message)
        real(dp), intent(in) :: actual, expected, tolerance
        character(len=*), intent(in) :: message
        real(dp) :: scale

        scale = max(abs(expected), 1.0_dp)
        call require(abs(actual - expected) <= tolerance * scale, message)
    end subroutine require_relative

    subroutine test_published_formula_fixtures()
        real(dp) :: pressure
        integer :: ierr

        ! Independent decimal fixtures were generated directly from the
        ! published Murphy-Koop equations, outside the Fortran implementation.
        call compute_saturation_vapor_pressure(250.0_dp, SAT_PHASE_LIQUID, pressure, ierr)
        call require(ierr == SAT_OK, '250 K liquid-water saturation pressure must succeed')
        call require_relative(pressure, 95.30126979027628_dp, 5.0e-13_dp, &
            '250 K liquid-water fixture must match the published parameterization')

        call compute_saturation_vapor_pressure(250.0_dp, SAT_PHASE_ICE, pressure, ierr)
        call require(ierr == SAT_OK, '250 K ice saturation pressure must succeed')
        call require_relative(pressure, 76.02389003659836_dp, 5.0e-13_dp, &
            '250 K ice fixture must match the published parameterization')

        call compute_saturation_vapor_pressure(300.0_dp, SAT_PHASE_LIQUID, pressure, ierr)
        call require(ierr == SAT_OK, '300 K liquid-water saturation pressure must succeed')
        call require_relative(pressure, 3536.7644130514645_dp, 5.0e-13_dp, &
            '300 K liquid-water fixture must match the published parameterization')
    end subroutine test_published_formula_fixtures

    subroutine test_monotonicity()
        real(dp) :: p1, p2, p3
        integer :: ierr

        call compute_saturation_vapor_pressure(230.0_dp, SAT_PHASE_LIQUID, p1, ierr)
        call require(ierr == SAT_OK, 'first liquid monotonicity point must succeed')
        call compute_saturation_vapor_pressure(250.0_dp, SAT_PHASE_LIQUID, p2, ierr)
        call require(ierr == SAT_OK, 'second liquid monotonicity point must succeed')
        call compute_saturation_vapor_pressure(273.15_dp, SAT_PHASE_LIQUID, p3, ierr)
        call require(ierr == SAT_OK, 'third liquid monotonicity point must succeed')
        call require(p1 < p2 .and. p2 < p3, &
            'liquid-water saturation pressure must increase over the tested temperature interval')

        call compute_saturation_vapor_pressure(180.0_dp, SAT_PHASE_ICE, p1, ierr)
        call require(ierr == SAT_OK, 'first ice monotonicity point must succeed')
        call compute_saturation_vapor_pressure(230.0_dp, SAT_PHASE_ICE, p2, ierr)
        call require(ierr == SAT_OK, 'second ice monotonicity point must succeed')
        call compute_saturation_vapor_pressure(250.0_dp, SAT_PHASE_ICE, p3, ierr)
        call require(ierr == SAT_OK, 'third ice monotonicity point must succeed')
        call require(p1 < p2 .and. p2 < p3, &
            'ice saturation pressure must increase over the tested temperature interval')
    end subroutine test_monotonicity

    subroutine test_supercooled_phase_difference()
        real(dp) :: p_liquid, p_ice
        integer :: ierr

        call compute_saturation_vapor_pressure(250.0_dp, SAT_PHASE_LIQUID, p_liquid, ierr)
        call require(ierr == SAT_OK, 'supercooled liquid pressure must succeed')
        call compute_saturation_vapor_pressure(250.0_dp, SAT_PHASE_ICE, p_ice, ierr)
        call require(ierr == SAT_OK, 'ice pressure at 250 K must succeed')
        call require(p_liquid > p_ice, &
            'supercooled liquid and ice must remain distinct phase conventions')
    end subroutine test_supercooled_phase_difference

    subroutine test_triple_point_agreement()
        real(dp) :: p_liquid, p_ice
        integer :: ierr

        call compute_saturation_vapor_pressure(WATER_TRIPLE_POINT_K, SAT_PHASE_LIQUID, p_liquid, ierr)
        call require(ierr == SAT_OK, 'liquid formula at the triple point must succeed')
        call compute_saturation_vapor_pressure(WATER_TRIPLE_POINT_K, SAT_PHASE_ICE, p_ice, ierr)
        call require(ierr == SAT_OK, 'ice formula at the triple point must succeed')
        call require(abs(p_liquid - p_ice) < 5.0e-5_dp, &
            'liquid and ice parameterizations must meet closely at the water triple point')
        call require_relative(p_liquid, 611.6570436443282_dp, 5.0e-13_dp, &
            'triple-point liquid fixture must remain stable')
    end subroutine test_triple_point_agreement

    subroutine test_composition_with_vapor_algebra()
        type(moist_vapor_parameters) :: parameters
        real(dp) :: e_liquid, e_roundtrip, mixing_ratio
        integer :: ierr, moist_ierr

        call compute_saturation_vapor_pressure(250.0_dp, SAT_PHASE_LIQUID, e_liquid, ierr)
        call require(ierr == SAT_OK, 'saturation provider must succeed before algebra composition')
        call compute_mixing_ratio_from_vapor_pressure(e_liquid, 80000.0_dp, parameters, &
            mixing_ratio, moist_ierr)
        call require(moist_ierr == MOIST_OK, &
            'published saturation pressure must compose with vapor-pressure/mixing-ratio algebra')
        call require(mixing_ratio > 0.0_dp, 'saturation mixing ratio must be positive')
        call compute_vapor_pressure_from_mixing_ratio(mixing_ratio, 80000.0_dp, parameters, &
            e_roundtrip, moist_ierr)
        call require(moist_ierr == MOIST_OK, 'saturation mixing-ratio round trip must succeed')
        call require_relative(e_roundtrip, e_liquid, 5.0e-14_dp, &
            'vapor algebra must recover the supplied saturation pressure')
    end subroutine test_composition_with_vapor_algebra

    subroutine test_phase_and_range_fail_closed()
        real(dp) :: pressure
        integer :: ierr

        call compute_saturation_vapor_pressure(250.0_dp, 999, pressure, ierr)
        call require(ierr == SAT_ERR_PHASE, 'unknown phase selector must fail closed')
        call require(pressure == 0.0_dp, 'unknown phase must not emit a pressure')

        call compute_saturation_vapor_pressure(123.0_dp, SAT_PHASE_LIQUID, pressure, ierr)
        call require(ierr == SAT_ERR_TEMPERATURE_RANGE, &
            'liquid lower published boundary is open and must fail at exactly 123 K')
        call require(pressure == 0.0_dp, 'out-of-range liquid call must not emit a pressure')

        call compute_saturation_vapor_pressure(332.0_dp, SAT_PHASE_LIQUID, pressure, ierr)
        call require(ierr == SAT_ERR_TEMPERATURE_RANGE, &
            'liquid upper published boundary is open and must fail at exactly 332 K')

        call compute_saturation_vapor_pressure(110.0_dp, SAT_PHASE_ICE, pressure, ierr)
        call require(ierr == SAT_ERR_TEMPERATURE_RANGE, &
            'ice published lower boundary is open and must fail at exactly 110 K')

        call compute_saturation_vapor_pressure(274.0_dp, SAT_PHASE_ICE, pressure, ierr)
        call require(ierr == SAT_ERR_TEMPERATURE_RANGE, &
            'canonical stable-ice policy must reject temperatures above the triple point')
    end subroutine test_phase_and_range_fail_closed

    subroutine test_nonfinite_fails_closed()
        real(dp) :: nan_value, pressure
        integer :: ierr

        nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
        call compute_saturation_vapor_pressure(nan_value, SAT_PHASE_LIQUID, pressure, ierr)
        call require(ierr == SAT_ERR_NONFINITE, 'non-finite temperature must fail closed')
        call require(pressure == 0.0_dp, 'non-finite input must not emit a pressure')
    end subroutine test_nonfinite_fails_closed

end program test_saturation_vapor_pressure
