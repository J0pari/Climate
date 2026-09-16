program test_saturation_moisture
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use climate_saturation_vapor_pressure, only: &
        compute_saturation_vapor_pressure, SAT_OK, SAT_ERR_PHASE, &
        SAT_PHASE_LIQUID, SAT_PHASE_ICE
    use climate_moist_vapor_algebra, only: &
        moist_vapor_parameters, MOIST_OK, MOIST_ERR_PARTIAL_PRESSURE, &
        compute_mixing_ratio_from_vapor_pressure, &
        compute_specific_humidity_from_mixing_ratio, &
        compute_vapor_pressure_from_mixing_ratio
    use climate_saturation_moisture, only: &
        saturation_moisture_state, compute_saturation_moisture_state, &
        SATMOIST_OK, SATMOIST_ERR_SATURATION_PROVIDER, &
        SATMOIST_ERR_MIXING_RATIO
    implicit none

    integer, parameter :: dp = real64
    type(moist_vapor_parameters) :: parameters

    call test_direct_composition(parameters)
    call test_phase_ordering(parameters)
    call test_pressure_dependence(parameters)
    call test_provider_failure_is_preserved(parameters)
    call test_algebra_failure_is_preserved(parameters)
    call test_vapor_pressure_round_trip(parameters)

contains

    subroutine test_direct_composition(parameters)
        type(moist_vapor_parameters), intent(in) :: parameters
        type(saturation_moisture_state) :: state
        real(dp) :: vapor_pressure_pa, mixing_ratio, specific_humidity
        integer :: ierr, sat_ierr, mix_ierr, q_ierr

        call compute_saturation_moisture_state(293.15_dp, 100000.0_dp, &
            SAT_PHASE_LIQUID, parameters, state, ierr)
        call assert_int_equal(ierr, SATMOIST_OK, 'nominal composition status')

        call compute_saturation_vapor_pressure(293.15_dp, SAT_PHASE_LIQUID, &
            vapor_pressure_pa, sat_ierr)
        call compute_mixing_ratio_from_vapor_pressure(vapor_pressure_pa, &
            100000.0_dp, parameters, mixing_ratio, mix_ierr)
        call compute_specific_humidity_from_mixing_ratio(mixing_ratio, &
            specific_humidity, q_ierr)

        call assert_int_equal(sat_ierr, SAT_OK, 'direct saturation status')
        call assert_int_equal(mix_ierr, MOIST_OK, 'direct mixing status')
        call assert_int_equal(q_ierr, MOIST_OK, 'direct humidity status')
        call assert_close(state%vapor_pressure_pa, vapor_pressure_pa, &
            1.0e-12_dp, 'composed vapor pressure')
        call assert_close(state%mixing_ratio, mixing_ratio, 1.0e-14_dp, &
            'composed mixing ratio')
        call assert_close(state%specific_humidity, specific_humidity, &
            1.0e-14_dp, 'composed specific humidity')
    end subroutine test_direct_composition


    subroutine test_phase_ordering(parameters)
        type(moist_vapor_parameters), intent(in) :: parameters
        type(saturation_moisture_state) :: liquid, ice
        integer :: ierr

        call compute_saturation_moisture_state(260.0_dp, 80000.0_dp, &
            SAT_PHASE_LIQUID, parameters, liquid, ierr)
        call assert_int_equal(ierr, SATMOIST_OK, 'supercooled liquid status')

        call compute_saturation_moisture_state(260.0_dp, 80000.0_dp, &
            SAT_PHASE_ICE, parameters, ice, ierr)
        call assert_int_equal(ierr, SATMOIST_OK, 'ice status')

        call assert_true(liquid%vapor_pressure_pa > ice%vapor_pressure_pa, &
            'liquid saturation pressure exceeds ice below freezing')
        call assert_true(liquid%mixing_ratio > ice%mixing_ratio, &
            'phase pressure ordering propagates to mixing ratio')
        call assert_true(liquid%specific_humidity > ice%specific_humidity, &
            'phase pressure ordering propagates to specific humidity')
    end subroutine test_phase_ordering


    subroutine test_pressure_dependence(parameters)
        type(moist_vapor_parameters), intent(in) :: parameters
        type(saturation_moisture_state) :: high_pressure, low_pressure
        integer :: ierr

        call compute_saturation_moisture_state(280.0_dp, 100000.0_dp, &
            SAT_PHASE_LIQUID, parameters, high_pressure, ierr)
        call assert_int_equal(ierr, SATMOIST_OK, 'high-pressure status')

        call compute_saturation_moisture_state(280.0_dp, 70000.0_dp, &
            SAT_PHASE_LIQUID, parameters, low_pressure, ierr)
        call assert_int_equal(ierr, SATMOIST_OK, 'low-pressure status')

        call assert_close(high_pressure%vapor_pressure_pa, &
            low_pressure%vapor_pressure_pa, 1.0e-12_dp, &
            'saturation vapor pressure is independent of ambient pressure')
        call assert_true(low_pressure%mixing_ratio > high_pressure%mixing_ratio, &
            'saturation mixing ratio increases as total pressure decreases')
    end subroutine test_pressure_dependence


    subroutine test_provider_failure_is_preserved(parameters)
        type(moist_vapor_parameters), intent(in) :: parameters
        type(saturation_moisture_state) :: state
        integer :: ierr

        call compute_saturation_moisture_state(260.0_dp, 80000.0_dp, 99, &
            parameters, state, ierr)

        call assert_int_equal(ierr, SATMOIST_ERR_SATURATION_PROVIDER, &
            'invalid phase adapter status')
        call assert_int_equal(state%saturation_ierr, SAT_ERR_PHASE, &
            'invalid phase provider status')
        call assert_int_equal(state%mixing_ratio_ierr, MOIST_OK, &
            'downstream mixing stage was not executed')
        call assert_int_equal(state%specific_humidity_ierr, MOIST_OK, &
            'downstream humidity stage was not executed')
        call assert_close(state%vapor_pressure_pa, 0.0_dp, 0.0_dp, &
            'failed provider leaves no vapor pressure')
    end subroutine test_provider_failure_is_preserved


    subroutine test_algebra_failure_is_preserved(parameters)
        type(moist_vapor_parameters), intent(in) :: parameters
        type(saturation_moisture_state) :: state
        integer :: ierr

        call compute_saturation_moisture_state(300.0_dp, 3000.0_dp, &
            SAT_PHASE_LIQUID, parameters, state, ierr)

        call assert_int_equal(ierr, SATMOIST_ERR_MIXING_RATIO, &
            'partial pressure exceeding total pressure fails in algebra stage')
        call assert_int_equal(state%saturation_ierr, SAT_OK, &
            'saturation provider succeeded before algebra failure')
        call assert_int_equal(state%mixing_ratio_ierr, &
            MOIST_ERR_PARTIAL_PRESSURE, 'algebra error provenance')
        call assert_int_equal(state%specific_humidity_ierr, MOIST_OK, &
            'specific humidity stage was not executed')
    end subroutine test_algebra_failure_is_preserved


    subroutine test_vapor_pressure_round_trip(parameters)
        type(moist_vapor_parameters), intent(in) :: parameters
        type(saturation_moisture_state) :: state
        real(dp) :: recovered_vapor_pressure_pa
        integer :: ierr, moist_ierr

        call compute_saturation_moisture_state(275.0_dp, 90000.0_dp, &
            SAT_PHASE_LIQUID, parameters, state, ierr)
        call assert_int_equal(ierr, SATMOIST_OK, 'round-trip setup')

        call compute_vapor_pressure_from_mixing_ratio(state%mixing_ratio, &
            90000.0_dp, parameters, recovered_vapor_pressure_pa, moist_ierr)
        call assert_int_equal(moist_ierr, MOIST_OK, 'round-trip algebra status')
        call assert_close(recovered_vapor_pressure_pa, state%vapor_pressure_pa, &
            1.0e-10_dp, 'mixing-ratio inverse recovers phase provider pressure')
    end subroutine test_vapor_pressure_round_trip


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

end program test_saturation_moisture
