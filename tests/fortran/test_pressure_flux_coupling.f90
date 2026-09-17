program test_pressure_flux_coupling
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_geophysical_reference_values, only: STANDARD_GRAVITY_M_S2
    use climate_pressure_flux_coupling
    use climate_conservative_transport, only: &
        advance_mass_and_tracer, transport_budget, TRANSPORT_OK
    use climate_pressure_coordinate_continuity, only: &
        integrate_pressure_velocity, CONTINUITY_OK
    use climate_pressure_energy_exchange, only: &
        evaluate_pressure_energy_exchange, pressure_energy_exchange, PRESSURE_ENERGY_OK
    implicit none

    call test_layer_mass_flux_divergence_identity()
    call test_transport_continuity_energy_composition()
    call test_invalid_geometry_fails_closed()

contains

    subroutine assert_near(actual, expected, relative_tolerance, label)
        real(real64), intent(in) :: actual, expected, relative_tolerance
        character(len=*), intent(in) :: label
        real(real64) :: scale

        scale = max(1.0_real64, abs(expected))
        if (abs(actual - expected) > relative_tolerance * scale) then
            write(*,*) 'FAIL ', trim(label), actual, expected
            error stop 1
        end if
    end subroutine assert_near

    subroutine assert_true(condition, label)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        if (.not. condition) then
            write(*,*) 'FAIL ', trim(label)
            error stop 1
        end if
    end subroutine assert_true

    subroutine test_layer_mass_flux_divergence_identity()
        real(real64) :: pressure(3), net_outward_flux(2), layer_mass(2), divergence(2)
        real(real64), parameter :: area_m2 = 2.0e6_real64
        integer :: ierr, k

        pressure = [100000.0_real64, 80000.0_real64, 50000.0_real64]
        net_outward_flux = [1000.0_real64, -500.0_real64]

        call map_net_face_mass_flux_to_pressure_divergence( &
            pressure, area_m2, STANDARD_GRAVITY_M_S2, net_outward_flux, &
            layer_mass, divergence, ierr)
        call assert_true(ierr == PRESSURE_FLUX_OK, 'layer mapping status')

        do k = 1, 2
            call assert_near( &
                layer_mass(k), &
                area_m2 * (pressure(k) - pressure(k + 1)) / STANDARD_GRAVITY_M_S2, &
                2.0e-15_real64, 'pressure layer mass')
            call assert_near( &
                layer_mass(k) * divergence(k), net_outward_flux(k), &
                2.0e-15_real64, 'mass flux divergence identity')
        end do
    end subroutine test_layer_mass_flux_divergence_identity

    subroutine test_transport_continuity_energy_composition()
        real(real64), parameter :: area_m2 = 1.0e4_real64
        real(real64), parameter :: dt_s = 10.0_real64
        real(real64), parameter :: tracer_fraction = 0.2_real64
        real(real64) :: pressure(2), face_flux(2), net_outward_flux(1)
        real(real64) :: layer_mass(1), divergence(1), tracer_mass(1)
        real(real64) :: face_tracer_fraction(2), zero_source(1)
        real(real64) :: next_mass(1), next_tracer(1)
        real(real64) :: interface_omega(2), column_divergence_pa_s
        real(real64) :: mass_tendency_kg_s, tracer_tendency_kg_s
        type(transport_budget) :: budget
        type(pressure_energy_exchange) :: exchange
        integer :: coupling_ierr, transport_ierr, continuity_ierr, energy_ierr

        pressure = [100000.0_real64, 90000.0_real64]
        ! Positive face flux follows the transport kernel's left-to-right
        ! orientation.  The same two authoritative face values are reduced to
        ! net outward flux for this one horizontal control volume.
        face_flux = [100.0_real64, 130.0_real64]
        net_outward_flux = [face_flux(2) - face_flux(1)]

        call map_net_face_mass_flux_to_pressure_divergence( &
            pressure, area_m2, STANDARD_GRAVITY_M_S2, net_outward_flux, &
            layer_mass, divergence, coupling_ierr)
        call assert_true(coupling_ierr == PRESSURE_FLUX_OK, 'composition coupling status')

        tracer_mass = tracer_fraction * layer_mass
        face_tracer_fraction = tracer_fraction
        zero_source = 0.0_real64
        call advance_mass_and_tracer( &
            dt_s, layer_mass, tracer_mass, face_flux, face_tracer_fraction, &
            zero_source, zero_source, next_mass, next_tracer, budget, transport_ierr)
        call assert_true(transport_ierr == TRANSPORT_OK, 'composition transport status')

        mass_tendency_kg_s = (next_mass(1) - layer_mass(1)) / dt_s
        tracer_tendency_kg_s = (next_tracer(1) - tracer_mass(1)) / dt_s
        call assert_near(mass_tendency_kg_s, -net_outward_flux(1), &
            2.0e-11_real64, 'transport uses authoritative carrier flux')
        call assert_near(mass_tendency_kg_s, -layer_mass(1) * divergence(1), &
            2.0e-11_real64, 'transport continuity mass identity')
        call assert_near(tracer_tendency_kg_s, tracer_fraction * mass_tendency_kg_s, &
            2.0e-11_real64, 'tracer follows authoritative carrier flux')

        call integrate_pressure_velocity( &
            pressure, divergence, 0.0_real64, interface_omega, &
            column_divergence_pa_s, continuity_ierr)
        call assert_true(continuity_ierr == CONTINUITY_OK, 'composition continuity status')
        call assert_near(interface_omega(2), &
            STANDARD_GRAVITY_M_S2 * net_outward_flux(1) / area_m2, &
            2.0e-15_real64, 'same carrier flux determines omega increment')

        ! Evaluate the pressure-coordinate energy exchange at the upper
        ! interface using the omega produced from the same carrier flux.
        call evaluate_pressure_energy_exchange( &
            0.0_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
            interface_omega(2), 0.8_real64, 0.0_real64, exchange, energy_ierr)
        call assert_true(energy_ierr == PRESSURE_ENERGY_OK, 'composition energy status')
        call assert_near(exchange%enthalpy_pressure_work_w_kg, &
            0.8_real64 * interface_omega(2), 2.0e-15_real64, &
            'carrier-flux-derived omega enters pressure work')
        call assert_near(exchange%closure_residual_w_kg, 0.0_real64, &
            2.0e-15_real64, 'pressure energy exchange closes')
    end subroutine test_transport_continuity_energy_composition

    subroutine test_invalid_geometry_fails_closed()
        real(real64) :: pressure(2), flux(1), layer_mass(1), divergence(1), nan_value
        integer :: ierr

        pressure = [100000.0_real64, 90000.0_real64]
        flux = [1.0_real64]

        call map_net_face_mass_flux_to_pressure_divergence( &
            pressure, 0.0_real64, STANDARD_GRAVITY_M_S2, flux, &
            layer_mass, divergence, ierr)
        call assert_true(ierr == PRESSURE_FLUX_ERR_GEOMETRY, 'zero area rejected')
        call assert_near(layer_mass(1), 0.0_real64, 0.0_real64, 'zero area mass fail closed')
        call assert_near(divergence(1), 0.0_real64, 0.0_real64, 'zero area divergence fail closed')

        nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
        flux = [nan_value]
        call map_net_face_mass_flux_to_pressure_divergence( &
            pressure, 1.0_real64, STANDARD_GRAVITY_M_S2, flux, &
            layer_mass, divergence, ierr)
        call assert_true(ierr == PRESSURE_FLUX_ERR_NONFINITE, 'nonfinite flux rejected')
    end subroutine test_invalid_geometry_fails_closed

end program test_pressure_flux_coupling
