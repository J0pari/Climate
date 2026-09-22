module climate_sigma_coordinate
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use climate_pressure_coordinate_grid, only: &
        validate_pressure_interfaces, PRESSURE_GRID_OK
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: SIGMA_COORDINATE_OK = 0
    integer, parameter, public :: SIGMA_COORDINATE_ERR_SIZE = 1
    integer, parameter, public :: SIGMA_COORDINATE_ERR_NONFINITE = 2
    integer, parameter, public :: SIGMA_COORDINATE_ERR_SURFACE_PRESSURE = 3
    integer, parameter, public :: SIGMA_COORDINATE_ERR_RANGE = 4
    integer, parameter, public :: SIGMA_COORDINATE_ERR_SURFACE_ANCHOR = 5
    integer, parameter, public :: SIGMA_COORDINATE_ERR_ORDER = 6
    integer, parameter, public :: SIGMA_COORDINATE_ERR_PRESSURE_GRID = 7

    public :: sigma_interfaces_to_pressure

contains

    subroutine sigma_interfaces_to_pressure( &
        sigma_interface, surface_pressure_pa, interface_pressure_pa, ierr)
        real(dp), intent(in) :: sigma_interface(:)
        real(dp), intent(in) :: surface_pressure_pa
        real(dp), intent(out) :: interface_pressure_pa(:)
        integer, intent(out) :: ierr

        integer :: k, pressure_ierr

        interface_pressure_pa = 0.0_dp
        ierr = SIGMA_COORDINATE_OK

        if (size(sigma_interface) < 2 .or. &
            size(interface_pressure_pa) /= size(sigma_interface)) then
            ierr = SIGMA_COORDINATE_ERR_SIZE
            return
        end if
        if (.not. ieee_is_finite(surface_pressure_pa)) then
            ierr = SIGMA_COORDINATE_ERR_NONFINITE
            return
        end if
        if (surface_pressure_pa <= 0.0_dp) then
            ierr = SIGMA_COORDINATE_ERR_SURFACE_PRESSURE
            return
        end if
        if (.not. all(ieee_is_finite(sigma_interface))) then
            ierr = SIGMA_COORDINATE_ERR_NONFINITE
            return
        end if
        if (any(sigma_interface <= 0.0_dp) .or. any(sigma_interface > 1.0_dp)) then
            ierr = SIGMA_COORDINATE_ERR_RANGE
            return
        end if
        if (sigma_interface(1) /= 1.0_dp) then
            ierr = SIGMA_COORDINATE_ERR_SURFACE_ANCHOR
            return
        end if
        do k = 1, size(sigma_interface) - 1
            if (sigma_interface(k) <= sigma_interface(k + 1)) then
                ierr = SIGMA_COORDINATE_ERR_ORDER
                return
            end if
        end do

        interface_pressure_pa = sigma_interface * surface_pressure_pa
        call validate_pressure_interfaces(interface_pressure_pa, pressure_ierr)
        if (pressure_ierr /= PRESSURE_GRID_OK) then
            interface_pressure_pa = 0.0_dp
            ierr = SIGMA_COORDINATE_ERR_PRESSURE_GRID
        end if
    end subroutine sigma_interfaces_to_pressure

end module climate_sigma_coordinate
