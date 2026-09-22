module climate_convective_column_redistribution
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: CONVECTIVE_REDISTRIBUTION_OK = 0
    integer, parameter, public :: CONVECTIVE_REDISTRIBUTION_ERR_EMPTY = 1
    integer, parameter, public :: CONVECTIVE_REDISTRIBUTION_ERR_DIMENSION = 2
    integer, parameter, public :: CONVECTIVE_REDISTRIBUTION_ERR_MASS = 3
    integer, parameter, public :: CONVECTIVE_REDISTRIBUTION_ERR_NONFINITE = 4

    type, public :: convective_column_budget_rate
        ! All quantities are per unit horizontal area. Positive interface flux
        ! is upward; column tendency is lower-boundary flux minus upper-boundary flux.
        real(dp) :: column_energy_tendency_w_m2
        real(dp) :: lower_energy_flux_w_m2
        real(dp) :: upper_energy_flux_w_m2
        real(dp) :: energy_closure_residual_w_m2
        real(dp) :: column_water_tendency_kg_m2_s
        real(dp) :: lower_water_flux_kg_m2_s
        real(dp) :: upper_water_flux_kg_m2_s
        real(dp) :: water_closure_residual_kg_m2_s
        real(dp) :: column_momentum_x_tendency_n_m2
        real(dp) :: lower_momentum_x_flux_n_m2
        real(dp) :: upper_momentum_x_flux_n_m2
        real(dp) :: momentum_x_closure_residual_n_m2
        real(dp) :: column_momentum_y_tendency_n_m2
        real(dp) :: lower_momentum_y_flux_n_m2
        real(dp) :: upper_momentum_y_flux_n_m2
        real(dp) :: momentum_y_closure_residual_n_m2
    end type convective_column_budget_rate

    public :: evaluate_convective_redistribution

contains

    subroutine clear_budget(budget)
        type(convective_column_budget_rate), intent(out) :: budget

        budget%column_energy_tendency_w_m2 = 0.0_dp
        budget%lower_energy_flux_w_m2 = 0.0_dp
        budget%upper_energy_flux_w_m2 = 0.0_dp
        budget%energy_closure_residual_w_m2 = 0.0_dp
        budget%column_water_tendency_kg_m2_s = 0.0_dp
        budget%lower_water_flux_kg_m2_s = 0.0_dp
        budget%upper_water_flux_kg_m2_s = 0.0_dp
        budget%water_closure_residual_kg_m2_s = 0.0_dp
        budget%column_momentum_x_tendency_n_m2 = 0.0_dp
        budget%lower_momentum_x_flux_n_m2 = 0.0_dp
        budget%upper_momentum_x_flux_n_m2 = 0.0_dp
        budget%momentum_x_closure_residual_n_m2 = 0.0_dp
        budget%column_momentum_y_tendency_n_m2 = 0.0_dp
        budget%lower_momentum_y_flux_n_m2 = 0.0_dp
        budget%upper_momentum_y_flux_n_m2 = 0.0_dp
        budget%momentum_y_closure_residual_n_m2 = 0.0_dp
    end subroutine clear_budget


    subroutine fail_outputs(energy_tendency_w_kg, water_tendency_s_inv, &
                            momentum_x_tendency_m_s2, momentum_y_tendency_m_s2, budget)
        real(dp), allocatable, intent(out) :: energy_tendency_w_kg(:)
        real(dp), allocatable, intent(out) :: water_tendency_s_inv(:)
        real(dp), allocatable, intent(out) :: momentum_x_tendency_m_s2(:)
        real(dp), allocatable, intent(out) :: momentum_y_tendency_m_s2(:)
        type(convective_column_budget_rate), intent(out) :: budget

        allocate(energy_tendency_w_kg(0), water_tendency_s_inv(0), &
                 momentum_x_tendency_m_s2(0), momentum_y_tendency_m_s2(0))
        call clear_budget(budget)
    end subroutine fail_outputs


    subroutine evaluate_convective_redistribution( &
        layer_air_mass_kg_m2, energy_flux_w_m2, water_flux_kg_m2_s, &
        momentum_x_flux_n_m2, momentum_y_flux_n_m2, &
        energy_tendency_w_kg, water_tendency_s_inv, &
        momentum_x_tendency_m_s2, momentum_y_tendency_m_s2, budget, ierr)
        real(dp), intent(in) :: layer_air_mass_kg_m2(:)
        real(dp), intent(in) :: energy_flux_w_m2(:)
        real(dp), intent(in) :: water_flux_kg_m2_s(:)
        real(dp), intent(in) :: momentum_x_flux_n_m2(:)
        real(dp), intent(in) :: momentum_y_flux_n_m2(:)
        real(dp), allocatable, intent(out) :: energy_tendency_w_kg(:)
        real(dp), allocatable, intent(out) :: water_tendency_s_inv(:)
        real(dp), allocatable, intent(out) :: momentum_x_tendency_m_s2(:)
        real(dp), allocatable, intent(out) :: momentum_y_tendency_m_s2(:)
        type(convective_column_budget_rate), intent(out) :: budget
        integer, intent(out) :: ierr

        integer :: i, n

        ierr = CONVECTIVE_REDISTRIBUTION_OK
        call clear_budget(budget)
        n = size(layer_air_mass_kg_m2)

        if (n == 0) then
            ierr = CONVECTIVE_REDISTRIBUTION_ERR_EMPTY
            call fail_outputs(energy_tendency_w_kg, water_tendency_s_inv, &
                momentum_x_tendency_m_s2, momentum_y_tendency_m_s2, budget)
            return
        end if
        if (size(energy_flux_w_m2) /= n + 1 .or. &
            size(water_flux_kg_m2_s) /= n + 1 .or. &
            size(momentum_x_flux_n_m2) /= n + 1 .or. &
            size(momentum_y_flux_n_m2) /= n + 1) then
            ierr = CONVECTIVE_REDISTRIBUTION_ERR_DIMENSION
            call fail_outputs(energy_tendency_w_kg, water_tendency_s_inv, &
                momentum_x_tendency_m_s2, momentum_y_tendency_m_s2, budget)
            return
        end if
        if (.not. all(ieee_is_finite(layer_air_mass_kg_m2)) .or. &
            any(layer_air_mass_kg_m2 <= 0.0_dp)) then
            ierr = CONVECTIVE_REDISTRIBUTION_ERR_MASS
            call fail_outputs(energy_tendency_w_kg, water_tendency_s_inv, &
                momentum_x_tendency_m_s2, momentum_y_tendency_m_s2, budget)
            return
        end if
        if (.not. all(ieee_is_finite(energy_flux_w_m2)) .or. &
            .not. all(ieee_is_finite(water_flux_kg_m2_s)) .or. &
            .not. all(ieee_is_finite(momentum_x_flux_n_m2)) .or. &
            .not. all(ieee_is_finite(momentum_y_flux_n_m2))) then
            ierr = CONVECTIVE_REDISTRIBUTION_ERR_NONFINITE
            call fail_outputs(energy_tendency_w_kg, water_tendency_s_inv, &
                momentum_x_tendency_m_s2, momentum_y_tendency_m_s2, budget)
            return
        end if

        allocate(energy_tendency_w_kg(n), water_tendency_s_inv(n), &
                 momentum_x_tendency_m_s2(n), momentum_y_tendency_m_s2(n))
        do i = 1, n
            energy_tendency_w_kg(i) = &
                (energy_flux_w_m2(i) - energy_flux_w_m2(i + 1)) / &
                layer_air_mass_kg_m2(i)
            water_tendency_s_inv(i) = &
                (water_flux_kg_m2_s(i) - water_flux_kg_m2_s(i + 1)) / &
                layer_air_mass_kg_m2(i)
            momentum_x_tendency_m_s2(i) = &
                (momentum_x_flux_n_m2(i) - momentum_x_flux_n_m2(i + 1)) / &
                layer_air_mass_kg_m2(i)
            momentum_y_tendency_m_s2(i) = &
                (momentum_y_flux_n_m2(i) - momentum_y_flux_n_m2(i + 1)) / &
                layer_air_mass_kg_m2(i)
        end do
        if (.not. all(ieee_is_finite(energy_tendency_w_kg)) .or. &
            .not. all(ieee_is_finite(water_tendency_s_inv)) .or. &
            .not. all(ieee_is_finite(momentum_x_tendency_m_s2)) .or. &
            .not. all(ieee_is_finite(momentum_y_tendency_m_s2))) then
            deallocate(energy_tendency_w_kg, water_tendency_s_inv, &
                       momentum_x_tendency_m_s2, momentum_y_tendency_m_s2)
            ierr = CONVECTIVE_REDISTRIBUTION_ERR_NONFINITE
            call fail_outputs(energy_tendency_w_kg, water_tendency_s_inv, &
                momentum_x_tendency_m_s2, momentum_y_tendency_m_s2, budget)
            return
        end if

        budget%lower_energy_flux_w_m2 = energy_flux_w_m2(1)
        budget%upper_energy_flux_w_m2 = energy_flux_w_m2(n + 1)
        budget%column_energy_tendency_w_m2 = &
            sum(layer_air_mass_kg_m2 * energy_tendency_w_kg)
        budget%energy_closure_residual_w_m2 = &
            budget%column_energy_tendency_w_m2 - &
            (budget%lower_energy_flux_w_m2 - budget%upper_energy_flux_w_m2)

        budget%lower_water_flux_kg_m2_s = water_flux_kg_m2_s(1)
        budget%upper_water_flux_kg_m2_s = water_flux_kg_m2_s(n + 1)
        budget%column_water_tendency_kg_m2_s = &
            sum(layer_air_mass_kg_m2 * water_tendency_s_inv)
        budget%water_closure_residual_kg_m2_s = &
            budget%column_water_tendency_kg_m2_s - &
            (budget%lower_water_flux_kg_m2_s - budget%upper_water_flux_kg_m2_s)

        budget%lower_momentum_x_flux_n_m2 = momentum_x_flux_n_m2(1)
        budget%upper_momentum_x_flux_n_m2 = momentum_x_flux_n_m2(n + 1)
        budget%column_momentum_x_tendency_n_m2 = &
            sum(layer_air_mass_kg_m2 * momentum_x_tendency_m_s2)
        budget%momentum_x_closure_residual_n_m2 = &
            budget%column_momentum_x_tendency_n_m2 - &
            (budget%lower_momentum_x_flux_n_m2 - budget%upper_momentum_x_flux_n_m2)

        budget%lower_momentum_y_flux_n_m2 = momentum_y_flux_n_m2(1)
        budget%upper_momentum_y_flux_n_m2 = momentum_y_flux_n_m2(n + 1)
        budget%column_momentum_y_tendency_n_m2 = &
            sum(layer_air_mass_kg_m2 * momentum_y_tendency_m_s2)
        budget%momentum_y_closure_residual_n_m2 = &
            budget%column_momentum_y_tendency_n_m2 - &
            (budget%lower_momentum_y_flux_n_m2 - budget%upper_momentum_y_flux_n_m2)
    end subroutine evaluate_convective_redistribution

end module climate_convective_column_redistribution
