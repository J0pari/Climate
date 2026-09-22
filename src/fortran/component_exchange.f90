module climate_component_exchange
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: COMPONENT_EXCHANGE_OK = 0
    integer, parameter, public :: COMPONENT_EXCHANGE_ERR_AREA = 1
    integer, parameter, public :: COMPONENT_EXCHANGE_ERR_TIMESTEP = 2
    integer, parameter, public :: COMPONENT_EXCHANGE_ERR_NONFINITE = 3

    type, public :: component_exchange_fluxes
        ! Sign convention: positive flux transfers the extensive quantity from
        ! the caller-designated source component to the target component.
        real(dp) :: energy_w_m2 = 0.0_dp
        real(dp) :: water_mass_kg_m2_s = 0.0_dp
        real(dp) :: momentum_x_n_m2 = 0.0_dp
        real(dp) :: momentum_y_n_m2 = 0.0_dp
    end type component_exchange_fluxes

    type, public :: component_exchange_budget
        real(dp) :: source_energy_j = 0.0_dp
        real(dp) :: target_energy_j = 0.0_dp
        real(dp) :: source_water_mass_kg = 0.0_dp
        real(dp) :: target_water_mass_kg = 0.0_dp
        real(dp) :: source_momentum_x_n_s = 0.0_dp
        real(dp) :: target_momentum_x_n_s = 0.0_dp
        real(dp) :: source_momentum_y_n_s = 0.0_dp
        real(dp) :: target_momentum_y_n_s = 0.0_dp
    end type component_exchange_budget

    public :: integrate_component_exchange

contains

    subroutine integrate_component_exchange(fluxes, area_m2, timestep_s, budget, ierr)
        type(component_exchange_fluxes), intent(in) :: fluxes
        real(dp), intent(in) :: area_m2
        real(dp), intent(in) :: timestep_s
        type(component_exchange_budget), intent(out) :: budget
        integer, intent(out) :: ierr

        real(dp) :: area_time
        real(dp) :: energy_transfer, water_transfer
        real(dp) :: momentum_x_transfer, momentum_y_transfer

        budget = component_exchange_budget()
        ierr = COMPONENT_EXCHANGE_OK

        if (.not. ieee_is_finite(area_m2) .or. area_m2 <= 0.0_dp) then
            ierr = COMPONENT_EXCHANGE_ERR_AREA
            return
        end if
        if (.not. ieee_is_finite(timestep_s) .or. timestep_s <= 0.0_dp) then
            ierr = COMPONENT_EXCHANGE_ERR_TIMESTEP
            return
        end if
        if (.not. ieee_is_finite(fluxes%energy_w_m2) .or. &
            .not. ieee_is_finite(fluxes%water_mass_kg_m2_s) .or. &
            .not. ieee_is_finite(fluxes%momentum_x_n_m2) .or. &
            .not. ieee_is_finite(fluxes%momentum_y_n_m2)) then
            ierr = COMPONENT_EXCHANGE_ERR_NONFINITE
            return
        end if

        area_time = area_m2 * timestep_s
        if (.not. ieee_is_finite(area_time)) then
            ierr = COMPONENT_EXCHANGE_ERR_NONFINITE
            return
        end if

        energy_transfer = fluxes%energy_w_m2 * area_time
        water_transfer = fluxes%water_mass_kg_m2_s * area_time
        momentum_x_transfer = fluxes%momentum_x_n_m2 * area_time
        momentum_y_transfer = fluxes%momentum_y_n_m2 * area_time
        if (.not. ieee_is_finite(energy_transfer) .or. &
            .not. ieee_is_finite(water_transfer) .or. &
            .not. ieee_is_finite(momentum_x_transfer) .or. &
            .not. ieee_is_finite(momentum_y_transfer)) then
            budget = component_exchange_budget()
            ierr = COMPONENT_EXCHANGE_ERR_NONFINITE
            return
        end if

        budget%source_energy_j = -energy_transfer
        budget%target_energy_j = energy_transfer
        budget%source_water_mass_kg = -water_transfer
        budget%target_water_mass_kg = water_transfer
        budget%source_momentum_x_n_s = -momentum_x_transfer
        budget%target_momentum_x_n_s = momentum_x_transfer
        budget%source_momentum_y_n_s = -momentum_y_transfer
        budget%target_momentum_y_n_s = momentum_y_transfer
    end subroutine integrate_component_exchange

end module climate_component_exchange
