program test_pressure_energy_exchange
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_pressure_energy_exchange, only: &
        pressure_energy_exchange, evaluate_pressure_energy_exchange, &
        PRESSURE_ENERGY_OK, PRESSURE_ENERGY_ERR_NONFINITE, &
        PRESSURE_ENERGY_ERR_SPECIFIC_VOLUME
    implicit none

    integer, parameter :: dp = real64

    call test_exchange_closure()
    call test_vertical_pressure_work_signs()
    call test_local_geopotential_tendency_passthrough()
    call test_invalid_specific_volume()
    call test_nonfinite_input()

contains

    subroutine test_exchange_closure()
        type(pressure_energy_exchange) :: exchange
        integer :: ierr

        call evaluate_pressure_energy_exchange(10.0_dp, -4.0_dp, &
            1.0e-3_dp, -2.0e-3_dp, -0.2_dp, 0.8_dp, 0.0_dp, exchange, ierr)

        call assert_int_equal(ierr, PRESSURE_ENERGY_OK, 'exchange status')
        call assert_close(exchange%horizontal_geopotential_advection_w_kg, &
            0.018_dp, 1.0e-14_dp, 'horizontal geopotential advection')
        call assert_close(exchange%kinetic_pressure_gradient_power_w_kg, &
            -0.018_dp, 1.0e-14_dp, 'pressure-gradient kinetic power')
        call assert_close(exchange%enthalpy_pressure_work_w_kg, &
            -0.16_dp, 1.0e-14_dp, 'ascending pressure work cools enthalpy')
        call assert_close(exchange%geopotential_material_tendency_w_kg, &
            0.178_dp, 1.0e-14_dp, 'geopotential material tendency')
        call assert_close(exchange%closure_residual_w_kg, &
            0.0_dp, 1.0e-14_dp, 'pressure-coordinate energy exchange closes')
    end subroutine test_exchange_closure


    subroutine test_vertical_pressure_work_signs()
        type(pressure_energy_exchange) :: ascending, descending
        integer :: ierr

        call evaluate_pressure_energy_exchange(0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
            -0.5_dp, 0.9_dp, 0.0_dp, ascending, ierr)
        call assert_int_equal(ierr, PRESSURE_ENERGY_OK, 'ascending status')
        call assert_true(ascending%enthalpy_pressure_work_w_kg < 0.0_dp, &
            'omega below zero gives adiabatic enthalpy cooling')
        call assert_true(ascending%geopotential_material_tendency_w_kg > 0.0_dp, &
            'ascending motion raises geopotential contribution')

        call evaluate_pressure_energy_exchange(0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
            0.5_dp, 0.9_dp, 0.0_dp, descending, ierr)
        call assert_int_equal(ierr, PRESSURE_ENERGY_OK, 'descending status')
        call assert_true(descending%enthalpy_pressure_work_w_kg > 0.0_dp, &
            'omega above zero gives adiabatic enthalpy warming')
        call assert_true(descending%geopotential_material_tendency_w_kg < 0.0_dp, &
            'descending motion lowers geopotential contribution')
    end subroutine test_vertical_pressure_work_signs


    subroutine test_local_geopotential_tendency_passthrough()
        type(pressure_energy_exchange) :: exchange
        real(dp) :: resolved_total
        integer :: ierr

        call evaluate_pressure_energy_exchange(7.0_dp, 3.0_dp, -8.0e-4_dp, &
            4.0e-4_dp, 0.15_dp, 0.75_dp, 2.5e-3_dp, exchange, ierr)
        call assert_int_equal(ierr, PRESSURE_ENERGY_OK, 'local tendency status')

        resolved_total = exchange%kinetic_pressure_gradient_power_w_kg + &
            exchange%enthalpy_pressure_work_w_kg + &
            exchange%geopotential_material_tendency_w_kg
        call assert_close(resolved_total, 2.5e-3_dp, 1.0e-13_dp, &
            'resolved exchange sums to explicit local geopotential tendency')
    end subroutine test_local_geopotential_tendency_passthrough


    subroutine test_invalid_specific_volume()
        type(pressure_energy_exchange) :: exchange
        integer :: ierr

        call evaluate_pressure_energy_exchange(1.0_dp, 0.0_dp, 0.01_dp, 0.0_dp, &
            0.0_dp, 0.0_dp, 0.0_dp, exchange, ierr)
        call assert_int_equal(ierr, PRESSURE_ENERGY_ERR_SPECIFIC_VOLUME, &
            'specific volume must be positive')
        call assert_close(exchange%closure_residual_w_kg, 0.0_dp, 0.0_dp, &
            'failed exchange is cleared')
    end subroutine test_invalid_specific_volume


    subroutine test_nonfinite_input()
        type(pressure_energy_exchange) :: exchange
        real(dp) :: nan_value
        integer :: ierr

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        call evaluate_pressure_energy_exchange(nan_value, 0.0_dp, 0.0_dp, 0.0_dp, &
            0.0_dp, 1.0_dp, 0.0_dp, exchange, ierr)
        call assert_int_equal(ierr, PRESSURE_ENERGY_ERR_NONFINITE, &
            'nonfinite exchange input rejected')
    end subroutine test_nonfinite_input


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

end program test_pressure_energy_exchange
