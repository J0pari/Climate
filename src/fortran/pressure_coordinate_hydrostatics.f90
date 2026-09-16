module climate_pressure_coordinate_hydrostatics
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: HYDRO_OK = 0
    integer, parameter, public :: HYDRO_ERR_NONFINITE = 1
    integer, parameter, public :: HYDRO_ERR_PARAMETERS = 2
    integer, parameter, public :: HYDRO_ERR_SHAPE = 3
    integer, parameter, public :: HYDRO_ERR_PRESSURE = 4
    integer, parameter, public :: HYDRO_ERR_PRESSURE_ORDER = 5
    integer, parameter, public :: HYDRO_ERR_TEMPERATURE = 6
    integer, parameter, public :: HYDRO_ERR_RESULT = 7

    type, public :: hydrostatic_parameters
        real(dp) :: dry_air_gas_constant_j_kg_k = 287.05_dp
        real(dp) :: gravity_m_s2 = 9.80665_dp
    end type hydrostatic_parameters

    public :: compute_hydrostatic_layer
    public :: integrate_pressure_column

contains

    logical function valid_parameters(parameters)
        type(hydrostatic_parameters), intent(in) :: parameters

        valid_parameters = &
            ieee_is_finite(parameters%dry_air_gas_constant_j_kg_k) .and. &
            ieee_is_finite(parameters%gravity_m_s2) .and. &
            parameters%dry_air_gas_constant_j_kg_k > 0.0_dp .and. &
            parameters%gravity_m_s2 > 0.0_dp
    end function valid_parameters


    subroutine compute_hydrostatic_layer(lower_pressure_pa, upper_pressure_pa, &
                                         mean_virtual_temperature_k, parameters, &
                                         mass_per_area_kg_m2, geopotential_thickness_m2_s2, &
                                         geometric_thickness_m, ierr)
        real(dp), intent(in) :: lower_pressure_pa, upper_pressure_pa
        real(dp), intent(in) :: mean_virtual_temperature_k
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp), intent(out) :: mass_per_area_kg_m2
        real(dp), intent(out) :: geopotential_thickness_m2_s2
        real(dp), intent(out) :: geometric_thickness_m
        integer, intent(out) :: ierr

        mass_per_area_kg_m2 = 0.0_dp
        geopotential_thickness_m2_s2 = 0.0_dp
        geometric_thickness_m = 0.0_dp
        ierr = HYDRO_OK

        if (.not. valid_parameters(parameters)) then
            ierr = HYDRO_ERR_PARAMETERS
            return
        end if
        if (.not. ieee_is_finite(lower_pressure_pa) .or. &
            .not. ieee_is_finite(upper_pressure_pa) .or. &
            .not. ieee_is_finite(mean_virtual_temperature_k)) then
            ierr = HYDRO_ERR_NONFINITE
            return
        end if
        if (lower_pressure_pa <= 0.0_dp .or. upper_pressure_pa <= 0.0_dp) then
            ierr = HYDRO_ERR_PRESSURE
            return
        end if
        if (lower_pressure_pa <= upper_pressure_pa) then
            ierr = HYDRO_ERR_PRESSURE_ORDER
            return
        end if
        if (mean_virtual_temperature_k <= 0.0_dp) then
            ierr = HYDRO_ERR_TEMPERATURE
            return
        end if

        mass_per_area_kg_m2 = &
            (lower_pressure_pa - upper_pressure_pa) / parameters%gravity_m_s2
        geopotential_thickness_m2_s2 = parameters%dry_air_gas_constant_j_kg_k * &
            mean_virtual_temperature_k * log(lower_pressure_pa / upper_pressure_pa)
        geometric_thickness_m = geopotential_thickness_m2_s2 / parameters%gravity_m_s2

        if (.not. ieee_is_finite(mass_per_area_kg_m2) .or. &
            .not. ieee_is_finite(geopotential_thickness_m2_s2) .or. &
            .not. ieee_is_finite(geometric_thickness_m) .or. &
            mass_per_area_kg_m2 <= 0.0_dp .or. &
            geopotential_thickness_m2_s2 <= 0.0_dp .or. &
            geometric_thickness_m <= 0.0_dp) then
            mass_per_area_kg_m2 = 0.0_dp
            geopotential_thickness_m2_s2 = 0.0_dp
            geometric_thickness_m = 0.0_dp
            ierr = HYDRO_ERR_RESULT
        end if
    end subroutine compute_hydrostatic_layer


    subroutine integrate_pressure_column(interface_pressure_pa, &
                                         layer_mean_virtual_temperature_k, parameters, &
                                         layer_mass_per_area_kg_m2, &
                                         layer_geopotential_thickness_m2_s2, &
                                         layer_geometric_thickness_m, &
                                         interface_relative_geopotential_m2_s2, ierr)
        real(dp), intent(in) :: interface_pressure_pa(:)
        real(dp), intent(in) :: layer_mean_virtual_temperature_k(:)
        type(hydrostatic_parameters), intent(in) :: parameters
        real(dp), intent(out) :: layer_mass_per_area_kg_m2(:)
        real(dp), intent(out) :: layer_geopotential_thickness_m2_s2(:)
        real(dp), intent(out) :: layer_geometric_thickness_m(:)
        real(dp), intent(out) :: interface_relative_geopotential_m2_s2(:)
        integer, intent(out) :: ierr

        integer :: n_layers, k, layer_ierr

        layer_mass_per_area_kg_m2 = 0.0_dp
        layer_geopotential_thickness_m2_s2 = 0.0_dp
        layer_geometric_thickness_m = 0.0_dp
        interface_relative_geopotential_m2_s2 = 0.0_dp
        ierr = HYDRO_OK

        n_layers = size(layer_mean_virtual_temperature_k)
        if (n_layers < 1 .or. &
            size(interface_pressure_pa) /= n_layers + 1 .or. &
            size(layer_mass_per_area_kg_m2) /= n_layers .or. &
            size(layer_geopotential_thickness_m2_s2) /= n_layers .or. &
            size(layer_geometric_thickness_m) /= n_layers .or. &
            size(interface_relative_geopotential_m2_s2) /= n_layers + 1) then
            ierr = HYDRO_ERR_SHAPE
            return
        end if
        if (.not. valid_parameters(parameters)) then
            ierr = HYDRO_ERR_PARAMETERS
            return
        end if

        interface_relative_geopotential_m2_s2(1) = 0.0_dp
        do k = 1, n_layers
            call compute_hydrostatic_layer(interface_pressure_pa(k), &
                interface_pressure_pa(k + 1), layer_mean_virtual_temperature_k(k), &
                parameters, layer_mass_per_area_kg_m2(k), &
                layer_geopotential_thickness_m2_s2(k), &
                layer_geometric_thickness_m(k), layer_ierr)
            if (layer_ierr /= HYDRO_OK) then
                layer_mass_per_area_kg_m2 = 0.0_dp
                layer_geopotential_thickness_m2_s2 = 0.0_dp
                layer_geometric_thickness_m = 0.0_dp
                interface_relative_geopotential_m2_s2 = 0.0_dp
                ierr = layer_ierr
                return
            end if

            interface_relative_geopotential_m2_s2(k + 1) = &
                interface_relative_geopotential_m2_s2(k) + &
                layer_geopotential_thickness_m2_s2(k)
        end do
    end subroutine integrate_pressure_column

end module climate_pressure_coordinate_hydrostatics
