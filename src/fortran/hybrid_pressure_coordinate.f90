module climate_hybrid_pressure_coordinate
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use climate_pressure_coordinate_grid, only: &
        validate_pressure_interfaces, PRESSURE_GRID_OK
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: HYBRID_COORDINATE_OK = 0
    integer, parameter, public :: HYBRID_COORDINATE_ERR_SIZE = 1
    integer, parameter, public :: HYBRID_COORDINATE_ERR_SOURCE = 2
    integer, parameter, public :: HYBRID_COORDINATE_ERR_NONFINITE = 3
    integer, parameter, public :: HYBRID_COORDINATE_ERR_SURFACE_PRESSURE = 4
    integer, parameter, public :: HYBRID_COORDINATE_ERR_A_RANGE = 5
    integer, parameter, public :: HYBRID_COORDINATE_ERR_B_RANGE = 6
    integer, parameter, public :: HYBRID_COORDINATE_ERR_SURFACE_ANCHOR = 7
    integer, parameter, public :: HYBRID_COORDINATE_ERR_PRESSURE_GRID = 8

    public :: hybrid_interfaces_to_pressure

contains

    subroutine hybrid_interfaces_to_pressure( &
        a_interface_pa, b_interface, surface_pressure_pa, coefficient_source_id, &
        interface_pressure_pa, ierr)
        real(dp), intent(in) :: a_interface_pa(:)
        real(dp), intent(in) :: b_interface(:)
        real(dp), intent(in) :: surface_pressure_pa
        character(len=*), intent(in) :: coefficient_source_id
        real(dp), intent(out) :: interface_pressure_pa(:)
        integer, intent(out) :: ierr

        integer :: pressure_ierr

        interface_pressure_pa = 0.0_dp
        ierr = HYBRID_COORDINATE_OK

        if (size(a_interface_pa) < 2 .or. &
            size(b_interface) /= size(a_interface_pa) .or. &
            size(interface_pressure_pa) /= size(a_interface_pa)) then
            ierr = HYBRID_COORDINATE_ERR_SIZE
            return
        end if
        if (len_trim(coefficient_source_id) == 0) then
            ierr = HYBRID_COORDINATE_ERR_SOURCE
            return
        end if
        if (.not. ieee_is_finite(surface_pressure_pa)) then
            ierr = HYBRID_COORDINATE_ERR_NONFINITE
            return
        end if
        if (surface_pressure_pa <= 0.0_dp) then
            ierr = HYBRID_COORDINATE_ERR_SURFACE_PRESSURE
            return
        end if
        if (.not. all(ieee_is_finite(a_interface_pa)) .or. &
            .not. all(ieee_is_finite(b_interface))) then
            ierr = HYBRID_COORDINATE_ERR_NONFINITE
            return
        end if
        if (any(a_interface_pa < 0.0_dp)) then
            ierr = HYBRID_COORDINATE_ERR_A_RANGE
            return
        end if
        if (any(b_interface < 0.0_dp) .or. any(b_interface > 1.0_dp)) then
            ierr = HYBRID_COORDINATE_ERR_B_RANGE
            return
        end if
        if (a_interface_pa(1) /= 0.0_dp .or. b_interface(1) /= 1.0_dp) then
            ierr = HYBRID_COORDINATE_ERR_SURFACE_ANCHOR
            return
        end if

        interface_pressure_pa = a_interface_pa + &
            b_interface * surface_pressure_pa
        call validate_pressure_interfaces(interface_pressure_pa, pressure_ierr)
        if (pressure_ierr /= PRESSURE_GRID_OK) then
            interface_pressure_pa = 0.0_dp
            ierr = HYBRID_COORDINATE_ERR_PRESSURE_GRID
        end if
    end subroutine hybrid_interfaces_to_pressure

end module climate_hybrid_pressure_coordinate
