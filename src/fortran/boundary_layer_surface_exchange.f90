module climate_boundary_layer_surface_exchange
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use climate_thermodynamic_reference_values, only: &
        DRY_AIR_HEAT_CAPACITY_CP_J_KG_K
    use climate_component_exchange, only: &
        component_exchange_fluxes, component_exchange_budget, &
        integrate_component_exchange, COMPONENT_EXCHANGE_OK
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: BOUNDARY_LAYER_EXCHANGE_OK = 0
    integer, parameter, public :: BOUNDARY_LAYER_EXCHANGE_ERR_MASS = 1
    integer, parameter, public :: BOUNDARY_LAYER_EXCHANGE_ERR_HEAT_CAPACITY = 2
    integer, parameter, public :: BOUNDARY_LAYER_EXCHANGE_ERR_NONFINITE = 3
    integer, parameter, public :: BOUNDARY_LAYER_EXCHANGE_ERR_COMPONENT = 4

    type, public :: boundary_layer_exchange_parameters
        ! Dry-air sensible-heat capacity used only to convert the declared
        ! sensible-heat exchange into a dry-air temperature tendency.
        real(dp) :: dry_air_heat_capacity_j_kg_k
    end type boundary_layer_exchange_parameters

    type, public :: boundary_layer_surface_fluxes
        ! Positive values transfer the named extensive quantity from the
        ! caller-designated surface/component into the atmospheric layer.
        real(dp) :: sensible_heat_w_m2
        real(dp) :: water_mass_kg_m2_s
        real(dp) :: stress_x_n_m2
        real(dp) :: stress_y_n_m2
    end type boundary_layer_surface_fluxes

    type, public :: boundary_layer_tendencies
        real(dp) :: dry_air_temperature_k_s
        ! kg water / kg reference layer air / s. The layer mass is held fixed
        ! during this tendency conversion; mass-state updating is caller-owned.
        real(dp) :: water_mass_fraction_s_inv
        real(dp) :: wind_x_m_s2
        real(dp) :: wind_y_m_s2
    end type boundary_layer_tendencies

    public :: reference_boundary_layer_exchange_parameters
    public :: evaluate_boundary_layer_surface_exchange

contains

    function reference_boundary_layer_exchange_parameters() result(parameters)
        type(boundary_layer_exchange_parameters) :: parameters

        parameters%dry_air_heat_capacity_j_kg_k = &
            DRY_AIR_HEAT_CAPACITY_CP_J_KG_K
    end function reference_boundary_layer_exchange_parameters


    subroutine clear_tendencies(tendencies)
        type(boundary_layer_tendencies), intent(out) :: tendencies

        tendencies%dry_air_temperature_k_s = 0.0_dp
        tendencies%water_mass_fraction_s_inv = 0.0_dp
        tendencies%wind_x_m_s2 = 0.0_dp
        tendencies%wind_y_m_s2 = 0.0_dp
    end subroutine clear_tendencies


    subroutine evaluate_boundary_layer_surface_exchange( &
        fluxes, layer_air_mass_kg, area_m2, timestep_s, parameters, &
        tendencies, budget, ierr)
        type(boundary_layer_surface_fluxes), intent(in) :: fluxes
        real(dp), intent(in) :: layer_air_mass_kg
        real(dp), intent(in) :: area_m2
        real(dp), intent(in) :: timestep_s
        type(boundary_layer_exchange_parameters), intent(in) :: parameters
        type(boundary_layer_tendencies), intent(out) :: tendencies
        type(component_exchange_budget), intent(out) :: budget
        integer, intent(out) :: ierr

        type(component_exchange_fluxes) :: component_fluxes
        integer :: component_ierr
        real(dp) :: inverse_mass_time

        call clear_tendencies(tendencies)
        budget = component_exchange_budget()
        ierr = BOUNDARY_LAYER_EXCHANGE_OK

        if (.not. ieee_is_finite(layer_air_mass_kg) .or. &
            layer_air_mass_kg <= 0.0_dp) then
            ierr = BOUNDARY_LAYER_EXCHANGE_ERR_MASS
            return
        end if
        if (.not. ieee_is_finite(parameters%dry_air_heat_capacity_j_kg_k) .or. &
            parameters%dry_air_heat_capacity_j_kg_k <= 0.0_dp) then
            ierr = BOUNDARY_LAYER_EXCHANGE_ERR_HEAT_CAPACITY
            return
        end if
        if (.not. ieee_is_finite(fluxes%sensible_heat_w_m2) .or. &
            .not. ieee_is_finite(fluxes%water_mass_kg_m2_s) .or. &
            .not. ieee_is_finite(fluxes%stress_x_n_m2) .or. &
            .not. ieee_is_finite(fluxes%stress_y_n_m2)) then
            ierr = BOUNDARY_LAYER_EXCHANGE_ERR_NONFINITE
            return
        end if

        component_fluxes%energy_w_m2 = fluxes%sensible_heat_w_m2
        component_fluxes%water_mass_kg_m2_s = fluxes%water_mass_kg_m2_s
        component_fluxes%momentum_x_n_m2 = fluxes%stress_x_n_m2
        component_fluxes%momentum_y_n_m2 = fluxes%stress_y_n_m2
        call integrate_component_exchange( &
            component_fluxes, area_m2, timestep_s, budget, component_ierr)
        if (component_ierr /= COMPONENT_EXCHANGE_OK) then
            budget = component_exchange_budget()
            ierr = BOUNDARY_LAYER_EXCHANGE_ERR_COMPONENT
            return
        end if

        inverse_mass_time = 1.0_dp / (layer_air_mass_kg * timestep_s)
        if (.not. ieee_is_finite(inverse_mass_time)) then
            budget = component_exchange_budget()
            ierr = BOUNDARY_LAYER_EXCHANGE_ERR_NONFINITE
            return
        end if

        tendencies%dry_air_temperature_k_s = &
            budget%target_energy_j * inverse_mass_time / &
            parameters%dry_air_heat_capacity_j_kg_k
        tendencies%water_mass_fraction_s_inv = &
            budget%target_water_mass_kg * inverse_mass_time
        tendencies%wind_x_m_s2 = &
            budget%target_momentum_x_n_s * inverse_mass_time
        tendencies%wind_y_m_s2 = &
            budget%target_momentum_y_n_s * inverse_mass_time

        if (.not. ieee_is_finite(tendencies%dry_air_temperature_k_s) .or. &
            .not. ieee_is_finite(tendencies%water_mass_fraction_s_inv) .or. &
            .not. ieee_is_finite(tendencies%wind_x_m_s2) .or. &
            .not. ieee_is_finite(tendencies%wind_y_m_s2)) then
            call clear_tendencies(tendencies)
            budget = component_exchange_budget()
            ierr = BOUNDARY_LAYER_EXCHANGE_ERR_NONFINITE
        end if
    end subroutine evaluate_boundary_layer_surface_exchange

end module climate_boundary_layer_surface_exchange
