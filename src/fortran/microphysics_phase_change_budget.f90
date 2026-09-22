module climate_microphysics_phase_change_budget
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: MICROPHYSICS_BUDGET_OK = 0
    integer, parameter, public :: MICROPHYSICS_BUDGET_ERR_EMPTY = 1
    integer, parameter, public :: MICROPHYSICS_BUDGET_ERR_DIMENSION = 2
    integer, parameter, public :: MICROPHYSICS_BUDGET_ERR_LATENT_HEAT = 3
    integer, parameter, public :: MICROPHYSICS_BUDGET_ERR_EXPORT = 4
    integer, parameter, public :: MICROPHYSICS_BUDGET_ERR_NONFINITE = 5

    type, public :: microphysics_column_budget_rate
        ! Positive phase_change_kg_m2_s means vapor -> condensate.
        ! Positive precipitation export leaves the represented atmospheric column.
        real(dp) :: vapor_tendency_kg_m2_s
        real(dp) :: condensate_tendency_kg_m2_s
        real(dp) :: total_water_tendency_kg_m2_s
        real(dp) :: precipitation_export_kg_m2_s
        real(dp) :: water_closure_residual_kg_m2_s
        real(dp) :: phase_change_energy_tendency_w_m2
        real(dp) :: precipitation_energy_export_w_m2
        real(dp) :: total_energy_tendency_w_m2
        real(dp) :: energy_closure_residual_w_m2
    end type microphysics_column_budget_rate

    public :: evaluate_microphysics_phase_change_budget

contains

    subroutine clear_budget(budget)
        type(microphysics_column_budget_rate), intent(out) :: budget

        budget%vapor_tendency_kg_m2_s = 0.0_dp
        budget%condensate_tendency_kg_m2_s = 0.0_dp
        budget%total_water_tendency_kg_m2_s = 0.0_dp
        budget%precipitation_export_kg_m2_s = 0.0_dp
        budget%water_closure_residual_kg_m2_s = 0.0_dp
        budget%phase_change_energy_tendency_w_m2 = 0.0_dp
        budget%precipitation_energy_export_w_m2 = 0.0_dp
        budget%total_energy_tendency_w_m2 = 0.0_dp
        budget%energy_closure_residual_w_m2 = 0.0_dp
    end subroutine clear_budget


    subroutine fail_outputs(vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
                            energy_tendency_w_m2, budget)
        real(dp), allocatable, intent(out) :: vapor_tendency_kg_m2_s(:)
        real(dp), allocatable, intent(out) :: condensate_tendency_kg_m2_s(:)
        real(dp), allocatable, intent(out) :: energy_tendency_w_m2(:)
        type(microphysics_column_budget_rate), intent(out) :: budget

        allocate(vapor_tendency_kg_m2_s(0), condensate_tendency_kg_m2_s(0), &
                 energy_tendency_w_m2(0))
        call clear_budget(budget)
    end subroutine fail_outputs


    subroutine evaluate_microphysics_phase_change_budget( &
        phase_change_kg_m2_s, precipitation_export_kg_m2_s, &
        precipitation_energy_export_w_m2, latent_heat_j_kg, &
        vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
        energy_tendency_w_m2, budget, ierr)
        real(dp), intent(in) :: phase_change_kg_m2_s(:)
        real(dp), intent(in) :: precipitation_export_kg_m2_s(:)
        real(dp), intent(in) :: precipitation_energy_export_w_m2(:)
        real(dp), intent(in) :: latent_heat_j_kg
        real(dp), allocatable, intent(out) :: vapor_tendency_kg_m2_s(:)
        real(dp), allocatable, intent(out) :: condensate_tendency_kg_m2_s(:)
        real(dp), allocatable, intent(out) :: energy_tendency_w_m2(:)
        type(microphysics_column_budget_rate), intent(out) :: budget
        integer, intent(out) :: ierr

        integer :: n

        ierr = MICROPHYSICS_BUDGET_OK
        call clear_budget(budget)
        n = size(phase_change_kg_m2_s)

        if (n == 0) then
            ierr = MICROPHYSICS_BUDGET_ERR_EMPTY
            call fail_outputs(vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
                energy_tendency_w_m2, budget)
            return
        end if
        if (size(precipitation_export_kg_m2_s) /= n .or. &
            size(precipitation_energy_export_w_m2) /= n) then
            ierr = MICROPHYSICS_BUDGET_ERR_DIMENSION
            call fail_outputs(vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
                energy_tendency_w_m2, budget)
            return
        end if
        if (.not. ieee_is_finite(latent_heat_j_kg) .or. latent_heat_j_kg <= 0.0_dp) then
            ierr = MICROPHYSICS_BUDGET_ERR_LATENT_HEAT
            call fail_outputs(vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
                energy_tendency_w_m2, budget)
            return
        end if
        if (.not. all(ieee_is_finite(phase_change_kg_m2_s)) .or. &
            .not. all(ieee_is_finite(precipitation_export_kg_m2_s)) .or. &
            .not. all(ieee_is_finite(precipitation_energy_export_w_m2))) then
            ierr = MICROPHYSICS_BUDGET_ERR_NONFINITE
            call fail_outputs(vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
                energy_tendency_w_m2, budget)
            return
        end if
        if (any(precipitation_export_kg_m2_s < 0.0_dp) .or. &
            any(precipitation_energy_export_w_m2 < 0.0_dp)) then
            ierr = MICROPHYSICS_BUDGET_ERR_EXPORT
            call fail_outputs(vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
                energy_tendency_w_m2, budget)
            return
        end if

        allocate(vapor_tendency_kg_m2_s(n), condensate_tendency_kg_m2_s(n), &
                 energy_tendency_w_m2(n))
        vapor_tendency_kg_m2_s = -phase_change_kg_m2_s
        condensate_tendency_kg_m2_s = &
            phase_change_kg_m2_s - precipitation_export_kg_m2_s
        energy_tendency_w_m2 = &
            latent_heat_j_kg * phase_change_kg_m2_s - &
            precipitation_energy_export_w_m2

        if (.not. all(ieee_is_finite(vapor_tendency_kg_m2_s)) .or. &
            .not. all(ieee_is_finite(condensate_tendency_kg_m2_s)) .or. &
            .not. all(ieee_is_finite(energy_tendency_w_m2))) then
            deallocate(vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
                       energy_tendency_w_m2)
            ierr = MICROPHYSICS_BUDGET_ERR_NONFINITE
            call fail_outputs(vapor_tendency_kg_m2_s, condensate_tendency_kg_m2_s, &
                energy_tendency_w_m2, budget)
            return
        end if

        budget%vapor_tendency_kg_m2_s = sum(vapor_tendency_kg_m2_s)
        budget%condensate_tendency_kg_m2_s = sum(condensate_tendency_kg_m2_s)
        budget%total_water_tendency_kg_m2_s = &
            budget%vapor_tendency_kg_m2_s + budget%condensate_tendency_kg_m2_s
        budget%precipitation_export_kg_m2_s = sum(precipitation_export_kg_m2_s)
        budget%water_closure_residual_kg_m2_s = &
            budget%total_water_tendency_kg_m2_s + &
            budget%precipitation_export_kg_m2_s

        budget%phase_change_energy_tendency_w_m2 = &
            latent_heat_j_kg * sum(phase_change_kg_m2_s)
        budget%precipitation_energy_export_w_m2 = &
            sum(precipitation_energy_export_w_m2)
        budget%total_energy_tendency_w_m2 = sum(energy_tendency_w_m2)
        budget%energy_closure_residual_w_m2 = &
            budget%total_energy_tendency_w_m2 - &
            (budget%phase_change_energy_tendency_w_m2 - &
             budget%precipitation_energy_export_w_m2)
    end subroutine evaluate_microphysics_phase_change_budget

end module climate_microphysics_phase_change_budget
