program test_conservative_transport
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_conservative_transport
    implicit none

    call test_zero_flux_identity()
    call test_closed_internal_flux_manual_state()
    call test_uniform_fraction_invariance()
    call test_boundary_and_source_budget()
    call test_invalid_inputs_fail_closed()
    call test_negative_result_rejected()

contains

    subroutine assert_close(actual, expected, tolerance, label)
        real(real64), intent(in) :: actual, expected, tolerance
        character(len=*), intent(in) :: label
        if (abs(actual - expected) > tolerance) then
            write(*,*) 'FAIL ', trim(label), actual, expected
            error stop 1
        end if
    end subroutine assert_close

    subroutine assert_array_close(actual, expected, tolerance, label)
        real(real64), intent(in) :: actual(:), expected(:), tolerance
        character(len=*), intent(in) :: label
        if (size(actual) /= size(expected) .or. any(abs(actual - expected) > tolerance)) then
            write(*,*) 'FAIL ', trim(label)
            write(*,*) 'actual  ', actual
            write(*,*) 'expected', expected
            error stop 1
        end if
    end subroutine assert_array_close

    subroutine assert_true(condition, label)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        if (.not. condition) then
            write(*,*) 'FAIL ', trim(label)
            error stop 1
        end if
    end subroutine assert_true

    subroutine test_zero_flux_identity()
        real(real64) :: mass(2), tracer(2), flux(3), qface(3), msource(2), tsource(2)
        real(real64) :: next_mass(2), next_tracer(2)
        type(transport_budget) :: budget
        integer :: ierr

        mass = [5.0_real64, 7.0_real64]
        tracer = [0.5_real64, 1.4_real64]
        flux = 0.0_real64
        qface = 0.0_real64
        msource = 0.0_real64
        tsource = 0.0_real64

        call advance_mass_and_tracer(10.0_real64, mass, tracer, flux, qface, msource, tsource, &
                                     next_mass, next_tracer, budget, ierr)
        call assert_true(ierr == TRANSPORT_OK, 'zero flux status')
        call assert_array_close(next_mass, mass, 1.0e-14_real64, 'zero flux mass identity')
        call assert_array_close(next_tracer, tracer, 1.0e-14_real64, 'zero flux tracer identity')
        call assert_close(budget%mass_balance_residual_kg, 0.0_real64, 1.0e-14_real64, 'zero flux mass residual')
        call assert_close(budget%tracer_balance_residual_kg, 0.0_real64, 1.0e-14_real64, 'zero flux tracer residual')
    end subroutine test_zero_flux_identity

    subroutine test_closed_internal_flux_manual_state()
        real(real64) :: mass(3), tracer(3), flux(4), qface(4), zeros(3)
        real(real64) :: next_mass(3), next_tracer(3)
        type(transport_budget) :: budget
        integer :: ierr

        mass = [10.0_real64, 20.0_real64, 30.0_real64]
        tracer = [1.0_real64, 4.0_real64, 9.0_real64]
        flux = [0.0_real64, 2.0_real64, -1.0_real64, 0.0_real64]
        qface = [0.0_real64, 0.1_real64, 0.3_real64, 0.0_real64]
        zeros = 0.0_real64

        call advance_mass_and_tracer(1.0_real64, mass, tracer, flux, qface, zeros, zeros, &
                                     next_mass, next_tracer, budget, ierr)
        call assert_true(ierr == TRANSPORT_OK, 'closed redistribution status')
        call assert_array_close(next_mass, [8.0_real64, 23.0_real64, 29.0_real64], &
                                1.0e-14_real64, 'closed redistribution mass')
        call assert_array_close(next_tracer, [0.8_real64, 4.5_real64, 8.7_real64], &
                                1.0e-14_real64, 'closed redistribution tracer')
        call assert_close(budget%mass_after_kg, budget%mass_before_kg, 1.0e-14_real64, 'closed mass conservation')
        call assert_close(budget%tracer_after_kg, budget%tracer_before_kg, 1.0e-14_real64, 'closed tracer conservation')
    end subroutine test_closed_internal_flux_manual_state

    subroutine test_uniform_fraction_invariance()
        real(real64) :: mass(3), tracer(3), flux(4), qface(4), zeros(3)
        real(real64) :: next_mass(3), next_tracer(3)
        type(transport_budget) :: budget
        integer :: ierr

        mass = [4.0_real64, 8.0_real64, 12.0_real64]
        tracer = 0.25_real64 * mass
        flux = [0.0_real64, 1.0_real64, -2.0_real64, 0.0_real64]
        qface = 0.25_real64
        zeros = 0.0_real64

        call advance_mass_and_tracer(1.0_real64, mass, tracer, flux, qface, zeros, zeros, &
                                     next_mass, next_tracer, budget, ierr)
        call assert_true(ierr == TRANSPORT_OK, 'uniform fraction status')
        call assert_array_close(next_mass, [3.0_real64, 11.0_real64, 10.0_real64], &
                                1.0e-14_real64, 'uniform fraction mass state')
        call assert_array_close(next_tracer, 0.25_real64 * next_mass, &
                                1.0e-14_real64, 'uniform fraction preserved')
    end subroutine test_uniform_fraction_invariance

    subroutine test_boundary_and_source_budget()
        real(real64) :: mass(2), tracer(2), flux(3), qface(3), msource(2), tsource(2)
        real(real64) :: next_mass(2), next_tracer(2)
        type(transport_budget) :: budget
        integer :: ierr

        mass = [10.0_real64, 10.0_real64]
        tracer = [1.0_real64, 2.0_real64]
        flux = [1.0_real64, 0.5_real64, 0.25_real64]
        qface = [0.2_real64, 0.4_real64, 0.8_real64]
        msource = [0.1_real64, -0.2_real64]
        tsource = [0.02_real64, -0.01_real64]

        call advance_mass_and_tracer(2.0_real64, mass, tracer, flux, qface, msource, tsource, &
                                     next_mass, next_tracer, budget, ierr)
        call assert_true(ierr == TRANSPORT_OK, 'boundary/source status')
        call assert_array_close(next_mass, [11.2_real64, 10.1_real64], 1.0e-14_real64, 'boundary/source mass')
        call assert_array_close(next_tracer, [1.04_real64, 1.98_real64], 1.0e-14_real64, 'boundary/source tracer')
        call assert_close(budget%lower_boundary_mass_change_kg, 2.0_real64, 1.0e-14_real64, &
                          'lower boundary mass contribution')
        call assert_close(budget%upper_boundary_mass_change_kg, -0.5_real64, 1.0e-14_real64, &
                          'upper boundary mass contribution')
        call assert_close(budget%resolved_mass_source_change_kg, 0.2_real64, 1.0e-14_real64, &
                          'resolved positive mass source')
        call assert_close(budget%resolved_mass_sink_change_kg, -0.4_real64, 1.0e-14_real64, &
                          'resolved negative mass sink')
        call assert_close(budget%expected_mass_change_kg, 1.3_real64, 1.0e-14_real64, 'expected mass budget')
        call assert_close(budget%lower_boundary_tracer_change_kg, 0.4_real64, 1.0e-14_real64, &
                          'lower boundary tracer contribution')
        call assert_close(budget%upper_boundary_tracer_change_kg, -0.4_real64, 1.0e-14_real64, &
                          'upper boundary tracer contribution')
        call assert_close(budget%resolved_tracer_source_change_kg, 0.04_real64, 1.0e-14_real64, &
                          'resolved positive tracer source')
        call assert_close(budget%resolved_tracer_sink_change_kg, -0.02_real64, 1.0e-14_real64, &
                          'resolved negative tracer sink')
        call assert_close(budget%expected_tracer_change_kg, 0.02_real64, 1.0e-14_real64, 'expected tracer budget')
        call assert_close(budget%mass_balance_residual_kg, 0.0_real64, 2.0e-14_real64, 'boundary mass residual')
        call assert_close(budget%tracer_balance_residual_kg, 0.0_real64, 2.0e-14_real64, 'boundary tracer residual')
    end subroutine test_boundary_and_source_budget

    subroutine test_invalid_inputs_fail_closed()
        real(real64) :: mass(1), tracer(1), flux(2), qface(2), source(1)
        real(real64) :: next_mass(1), next_tracer(1), nan_value
        type(transport_budget) :: budget
        integer :: ierr

        mass = [1.0_real64]
        tracer = [0.1_real64]
        flux = 0.0_real64
        qface = [0.0_real64, 1.2_real64]
        source = 0.0_real64

        call advance_mass_and_tracer(1.0_real64, mass, tracer, flux, qface, source, source, &
                                     next_mass, next_tracer, budget, ierr)
        call assert_true(ierr == TRANSPORT_ERR_FRACTION, 'invalid face fraction rejected')
        call assert_array_close(next_mass, [0.0_real64], 0.0_real64, 'invalid fraction mass fail closed')
        call assert_array_close(next_tracer, [0.0_real64], 0.0_real64, 'invalid fraction tracer fail closed')

        nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
        flux = [nan_value, 0.0_real64]
        qface = 0.0_real64
        call advance_mass_and_tracer(1.0_real64, mass, tracer, flux, qface, source, source, &
                                     next_mass, next_tracer, budget, ierr)
        call assert_true(ierr == TRANSPORT_ERR_NONFINITE, 'nonfinite flux rejected')

        flux = 0.0_real64
        call advance_mass_and_tracer(-1.0_real64, mass, tracer, flux, qface, source, source, &
                                     next_mass, next_tracer, budget, ierr)
        call assert_true(ierr == TRANSPORT_ERR_TIMESTEP, 'negative timestep rejected')
    end subroutine test_invalid_inputs_fail_closed

    subroutine test_negative_result_rejected()
        real(real64) :: mass(1), tracer(1), flux(2), qface(2), source(1)
        real(real64) :: next_mass(1), next_tracer(1)
        type(transport_budget) :: budget
        integer :: ierr

        mass = [1.0_real64]
        tracer = [0.1_real64]
        flux = [0.0_real64, 2.0_real64]
        qface = 0.1_real64
        source = 0.0_real64

        call advance_mass_and_tracer(1.0_real64, mass, tracer, flux, qface, source, source, &
                                     next_mass, next_tracer, budget, ierr)
        call assert_true(ierr == TRANSPORT_ERR_RESULT, 'negative mass result rejected')
        call assert_array_close(next_mass, [0.0_real64], 0.0_real64, 'negative result mass fail closed')
        call assert_array_close(next_tracer, [0.0_real64], 0.0_real64, 'negative result tracer fail closed')
    end subroutine test_negative_result_rejected

end program test_conservative_transport
