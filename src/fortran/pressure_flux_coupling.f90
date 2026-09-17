module climate_pressure_flux_coupling
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use climate_pressure_coordinate_grid, only: &
        validate_pressure_interfaces, PRESSURE_GRID_OK, PRESSURE_GRID_ERR_SIZE, &
        PRESSURE_GRID_ERR_NONFINITE, PRESSURE_GRID_ERR_PRESSURE, &
        PRESSURE_GRID_ERR_ORDER
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: PRESSURE_FLUX_OK = 0
    integer, parameter, public :: PRESSURE_FLUX_ERR_SIZE = 1
    integer, parameter, public :: PRESSURE_FLUX_ERR_NONFINITE = 2
    integer, parameter, public :: PRESSURE_FLUX_ERR_PRESSURE = 3
    integer, parameter, public :: PRESSURE_FLUX_ERR_PRESSURE_ORDER = 4
    integer, parameter, public :: PRESSURE_FLUX_ERR_GEOMETRY = 5
    integer, parameter, public :: PRESSURE_FLUX_ERR_RESULT = 6

    public :: map_net_face_mass_flux_to_pressure_divergence

contains

    integer function map_pressure_grid_error(grid_ierr) result(ierr)
        integer, intent(in) :: grid_ierr

        select case (grid_ierr)
        case (PRESSURE_GRID_OK)
            ierr = PRESSURE_FLUX_OK
        case (PRESSURE_GRID_ERR_SIZE)
            ierr = PRESSURE_FLUX_ERR_SIZE
        case (PRESSURE_GRID_ERR_NONFINITE)
            ierr = PRESSURE_FLUX_ERR_NONFINITE
        case (PRESSURE_GRID_ERR_PRESSURE)
            ierr = PRESSURE_FLUX_ERR_PRESSURE
        case (PRESSURE_GRID_ERR_ORDER)
            ierr = PRESSURE_FLUX_ERR_PRESSURE_ORDER
        case default
            ierr = PRESSURE_FLUX_ERR_RESULT
        end select
    end function map_pressure_grid_error


    subroutine map_net_face_mass_flux_to_pressure_divergence( &
            interface_pressure_pa, cell_horizontal_area_m2, gravity_m_s2, &
            net_outward_face_mass_flux_kg_s, layer_mass_kg, &
            layer_pressure_mean_horizontal_divergence_s1, ierr)
        real(dp), intent(in) :: interface_pressure_pa(:)
        real(dp), intent(in) :: cell_horizontal_area_m2, gravity_m_s2
        real(dp), intent(in) :: net_outward_face_mass_flux_kg_s(:)
        real(dp), intent(out) :: layer_mass_kg(:)
        real(dp), intent(out) :: layer_pressure_mean_horizontal_divergence_s1(:)
        integer, intent(out) :: ierr

        integer :: grid_ierr, k, n_layers
        real(dp) :: delta_pressure_pa

        layer_mass_kg = 0.0_dp
        layer_pressure_mean_horizontal_divergence_s1 = 0.0_dp
        ierr = PRESSURE_FLUX_OK

        n_layers = size(net_outward_face_mass_flux_kg_s)
        if (n_layers < 1 .or. size(interface_pressure_pa) /= n_layers + 1 .or. &
            size(layer_mass_kg) /= n_layers .or. &
            size(layer_pressure_mean_horizontal_divergence_s1) /= n_layers) then
            ierr = PRESSURE_FLUX_ERR_SIZE
            return
        end if

        call validate_pressure_interfaces(interface_pressure_pa, grid_ierr)
        ierr = map_pressure_grid_error(grid_ierr)
        if (ierr /= PRESSURE_FLUX_OK) return

        if (.not. ieee_is_finite(cell_horizontal_area_m2) .or. &
            .not. ieee_is_finite(gravity_m_s2) .or. &
            .not. all(ieee_is_finite(net_outward_face_mass_flux_kg_s))) then
            ierr = PRESSURE_FLUX_ERR_NONFINITE
            return
        end if
        if (cell_horizontal_area_m2 <= 0.0_dp .or. gravity_m_s2 <= 0.0_dp) then
            ierr = PRESSURE_FLUX_ERR_GEOMETRY
            return
        end if

        ! The caller owns horizontal mesh topology and face orientation.  Each
        ! input entry is the signed sum of outward carrier-mass flux over all
        ! horizontal faces of one pressure layer.  For fixed pressure
        ! coordinates, hydrostatic layer mass is A * Delta p / g and
        ! div_h = net_outward_mass_flux / layer_mass.
        do k = 1, n_layers
            delta_pressure_pa = interface_pressure_pa(k) - interface_pressure_pa(k + 1)
            layer_mass_kg(k) = cell_horizontal_area_m2 * delta_pressure_pa / gravity_m_s2
            layer_pressure_mean_horizontal_divergence_s1(k) = &
                net_outward_face_mass_flux_kg_s(k) / layer_mass_kg(k)
        end do

        if (.not. all(ieee_is_finite(layer_mass_kg)) .or. &
            .not. all(ieee_is_finite(layer_pressure_mean_horizontal_divergence_s1)) .or. &
            any(layer_mass_kg <= 0.0_dp)) then
            layer_mass_kg = 0.0_dp
            layer_pressure_mean_horizontal_divergence_s1 = 0.0_dp
            ierr = PRESSURE_FLUX_ERR_RESULT
        end if
    end subroutine map_net_face_mass_flux_to_pressure_divergence

end module climate_pressure_flux_coupling
