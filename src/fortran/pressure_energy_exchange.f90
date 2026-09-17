module climate_pressure_energy_exchange
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: PRESSURE_ENERGY_OK = 0
    integer, parameter, public :: PRESSURE_ENERGY_ERR_NONFINITE = 1
    integer, parameter, public :: PRESSURE_ENERGY_ERR_SPECIFIC_VOLUME = 2
    integer, parameter, public :: PRESSURE_ENERGY_ERR_RESULT = 3

    type, public :: pressure_energy_exchange
        ! All power terms are specific powers in W/kg = m^2/s^3.
        real(dp) :: horizontal_geopotential_advection_w_kg = 0.0_dp
        real(dp) :: kinetic_pressure_gradient_power_w_kg = 0.0_dp
        real(dp) :: enthalpy_pressure_work_w_kg = 0.0_dp
        real(dp) :: geopotential_material_tendency_w_kg = 0.0_dp
        real(dp) :: local_geopotential_tendency_w_kg = 0.0_dp
        real(dp) :: closure_residual_w_kg = 0.0_dp
    end type pressure_energy_exchange

    public :: evaluate_pressure_energy_exchange

contains

    subroutine evaluate_pressure_energy_exchange( &
            zonal_velocity_m_s, meridional_velocity_m_s, &
            geopotential_gradient_x_m_s2, geopotential_gradient_y_m_s2, &
            pressure_velocity_pa_s, specific_volume_m3_kg, &
            local_geopotential_tendency_m2_s3, exchange, ierr)
        real(dp), intent(in) :: zonal_velocity_m_s, meridional_velocity_m_s
        real(dp), intent(in) :: geopotential_gradient_x_m_s2, geopotential_gradient_y_m_s2
        real(dp), intent(in) :: pressure_velocity_pa_s, specific_volume_m3_kg
        real(dp), intent(in) :: local_geopotential_tendency_m2_s3
        type(pressure_energy_exchange), intent(out) :: exchange
        integer, intent(out) :: ierr

        real(dp) :: horizontal_advection

        exchange = pressure_energy_exchange()
        ierr = PRESSURE_ENERGY_OK

        if (.not. ieee_is_finite(zonal_velocity_m_s) .or. &
            .not. ieee_is_finite(meridional_velocity_m_s) .or. &
            .not. ieee_is_finite(geopotential_gradient_x_m_s2) .or. &
            .not. ieee_is_finite(geopotential_gradient_y_m_s2) .or. &
            .not. ieee_is_finite(pressure_velocity_pa_s) .or. &
            .not. ieee_is_finite(specific_volume_m3_kg) .or. &
            .not. ieee_is_finite(local_geopotential_tendency_m2_s3)) then
            ierr = PRESSURE_ENERGY_ERR_NONFINITE
            return
        end if
        if (specific_volume_m3_kg <= 0.0_dp) then
            ierr = PRESSURE_ENERGY_ERR_SPECIFIC_VOLUME
            return
        end if

        ! In pressure coordinates the horizontal pressure-gradient acceleration
        ! is -grad_p(Phi), so its kinetic-energy power is -v . grad_p(Phi).
        horizontal_advection = &
            zonal_velocity_m_s * geopotential_gradient_x_m_s2 + &
            meridional_velocity_m_s * geopotential_gradient_y_m_s2

        exchange%horizontal_geopotential_advection_w_kg = horizontal_advection
        exchange%kinetic_pressure_gradient_power_w_kg = -horizontal_advection

        ! The dry/moist equation of state remains caller-owned.  Given the
        ! caller's specific volume alpha, pressure-coordinate compressional
        ! work in the enthalpy equation is alpha * omega.
        exchange%enthalpy_pressure_work_w_kg = &
            specific_volume_m3_kg * pressure_velocity_pa_s

        ! Hydrostatic pressure coordinates give D Phi / Dt =
        ! partial_t Phi + v . grad_p(Phi) - alpha * omega.
        exchange%local_geopotential_tendency_w_kg = &
            local_geopotential_tendency_m2_s3
        exchange%geopotential_material_tendency_w_kg = &
            local_geopotential_tendency_m2_s3 + horizontal_advection - &
            exchange%enthalpy_pressure_work_w_kg

        ! The three exchange terms must collapse exactly to the explicit local
        ! geopotential tendency.  This diagnostic is the coupling contract;
        ! it does not choose a horizontal derivative, EOS, or time integrator.
        exchange%closure_residual_w_kg = &
            exchange%kinetic_pressure_gradient_power_w_kg + &
            exchange%enthalpy_pressure_work_w_kg + &
            exchange%geopotential_material_tendency_w_kg - &
            exchange%local_geopotential_tendency_w_kg

        if (.not. ieee_is_finite(exchange%horizontal_geopotential_advection_w_kg) .or. &
            .not. ieee_is_finite(exchange%kinetic_pressure_gradient_power_w_kg) .or. &
            .not. ieee_is_finite(exchange%enthalpy_pressure_work_w_kg) .or. &
            .not. ieee_is_finite(exchange%geopotential_material_tendency_w_kg) .or. &
            .not. ieee_is_finite(exchange%closure_residual_w_kg)) then
            exchange = pressure_energy_exchange()
            ierr = PRESSURE_ENERGY_ERR_RESULT
        end if
    end subroutine evaluate_pressure_energy_exchange

end module climate_pressure_energy_exchange
