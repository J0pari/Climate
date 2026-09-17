module climate_pressure_coordinate_grid
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: PRESSURE_GRID_OK = 0
    integer, parameter, public :: PRESSURE_GRID_ERR_SIZE = 1
    integer, parameter, public :: PRESSURE_GRID_ERR_NONFINITE = 2
    integer, parameter, public :: PRESSURE_GRID_ERR_PRESSURE = 3
    integer, parameter, public :: PRESSURE_GRID_ERR_ORDER = 4

    public :: validate_pressure_layer
    public :: validate_pressure_interfaces

contains

    subroutine validate_pressure_layer(lower_pressure_pa, upper_pressure_pa, ierr)
        real(dp), intent(in) :: lower_pressure_pa, upper_pressure_pa
        integer, intent(out) :: ierr

        ierr = PRESSURE_GRID_OK

        if (.not. ieee_is_finite(lower_pressure_pa) .or. &
            .not. ieee_is_finite(upper_pressure_pa)) then
            ierr = PRESSURE_GRID_ERR_NONFINITE
            return
        end if
        if (lower_pressure_pa <= 0.0_dp .or. upper_pressure_pa <= 0.0_dp) then
            ierr = PRESSURE_GRID_ERR_PRESSURE
            return
        end if
        if (lower_pressure_pa <= upper_pressure_pa) then
            ierr = PRESSURE_GRID_ERR_ORDER
        end if
    end subroutine validate_pressure_layer


    subroutine validate_pressure_interfaces(interface_pressure_pa, ierr)
        real(dp), intent(in) :: interface_pressure_pa(:)
        integer, intent(out) :: ierr

        integer :: k, layer_ierr

        ierr = PRESSURE_GRID_OK
        if (size(interface_pressure_pa) < 2) then
            ierr = PRESSURE_GRID_ERR_SIZE
            return
        end if

        do k = 1, size(interface_pressure_pa) - 1
            call validate_pressure_layer(interface_pressure_pa(k), &
                interface_pressure_pa(k + 1), layer_ierr)
            if (layer_ierr /= PRESSURE_GRID_OK) then
                ierr = layer_ierr
                return
            end if
        end do
    end subroutine validate_pressure_interfaces

end module climate_pressure_coordinate_grid
