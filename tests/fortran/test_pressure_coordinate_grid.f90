program test_pressure_coordinate_grid
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_pressure_coordinate_grid, only: &
        validate_pressure_layer, validate_pressure_interfaces, &
        PRESSURE_GRID_OK, PRESSURE_GRID_ERR_SIZE, &
        PRESSURE_GRID_ERR_NONFINITE, PRESSURE_GRID_ERR_PRESSURE, &
        PRESSURE_GRID_ERR_ORDER
    implicit none

    integer, parameter :: dp = real64

    call test_valid_layer_and_column()
    call test_invalid_size()
    call test_nonfinite_pressure()
    call test_nonpositive_pressure()
    call test_invalid_order()

contains

    subroutine test_valid_layer_and_column()
        real(dp) :: pressure(4)
        integer :: ierr

        pressure = [100000.0_dp, 85000.0_dp, 70000.0_dp, 50000.0_dp]

        call validate_pressure_layer(pressure(1), pressure(2), ierr)
        call assert_int_equal(ierr, PRESSURE_GRID_OK, 'valid pressure layer')

        call validate_pressure_interfaces(pressure, ierr)
        call assert_int_equal(ierr, PRESSURE_GRID_OK, 'valid pressure column')
    end subroutine test_valid_layer_and_column


    subroutine test_invalid_size()
        real(dp) :: pressure(1)
        integer :: ierr

        pressure = [100000.0_dp]
        call validate_pressure_interfaces(pressure, ierr)
        call assert_int_equal(ierr, PRESSURE_GRID_ERR_SIZE, &
            'pressure column requires at least two interfaces')
    end subroutine test_invalid_size


    subroutine test_nonfinite_pressure()
        real(dp) :: pressure(2)
        integer :: ierr

        pressure = [100000.0_dp, ieee_value(0.0_dp, ieee_quiet_nan)]
        call validate_pressure_interfaces(pressure, ierr)
        call assert_int_equal(ierr, PRESSURE_GRID_ERR_NONFINITE, &
            'nonfinite pressure is rejected')
    end subroutine test_nonfinite_pressure


    subroutine test_nonpositive_pressure()
        real(dp) :: pressure(2)
        integer :: ierr

        pressure = [100000.0_dp, 0.0_dp]
        call validate_pressure_interfaces(pressure, ierr)
        call assert_int_equal(ierr, PRESSURE_GRID_ERR_PRESSURE, &
            'nonpositive pressure is rejected')
    end subroutine test_nonpositive_pressure


    subroutine test_invalid_order()
        real(dp) :: pressure(3)
        integer :: ierr

        pressure = [100000.0_dp, 80000.0_dp, 90000.0_dp]
        call validate_pressure_interfaces(pressure, ierr)
        call assert_int_equal(ierr, PRESSURE_GRID_ERR_ORDER, &
            'pressure interfaces strictly decrease upward')
    end subroutine test_invalid_order


    subroutine assert_int_equal(actual, expected, label)
        integer, intent(in) :: actual, expected
        character(len=*), intent(in) :: label

        if (actual /= expected) then
            write(error_unit, '(A,2(1X,I0))') trim(label), actual, expected
            error stop 1
        end if
    end subroutine assert_int_equal

end program test_pressure_coordinate_grid
