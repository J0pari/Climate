module climate_conservative_transport
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: TRANSPORT_OK = 0
    integer, parameter, public :: TRANSPORT_ERR_SIZE = 1
    integer, parameter, public :: TRANSPORT_ERR_NONFINITE = 2
    integer, parameter, public :: TRANSPORT_ERR_TIMESTEP = 3
    integer, parameter, public :: TRANSPORT_ERR_STATE = 4
    integer, parameter, public :: TRANSPORT_ERR_FRACTION = 5
    integer, parameter, public :: TRANSPORT_ERR_RESULT = 6

    type, public :: transport_budget
        real(dp) :: mass_before_kg = 0.0_dp
        real(dp) :: mass_after_kg = 0.0_dp
        real(dp) :: lower_boundary_mass_change_kg = 0.0_dp
        real(dp) :: upper_boundary_mass_change_kg = 0.0_dp
        real(dp) :: resolved_mass_source_change_kg = 0.0_dp
        real(dp) :: resolved_mass_sink_change_kg = 0.0_dp
        real(dp) :: expected_mass_change_kg = 0.0_dp
        real(dp) :: mass_balance_residual_kg = 0.0_dp
        real(dp) :: tracer_before_kg = 0.0_dp
        real(dp) :: tracer_after_kg = 0.0_dp
        real(dp) :: lower_boundary_tracer_change_kg = 0.0_dp
        real(dp) :: upper_boundary_tracer_change_kg = 0.0_dp
        real(dp) :: resolved_tracer_source_change_kg = 0.0_dp
        real(dp) :: resolved_tracer_sink_change_kg = 0.0_dp
        real(dp) :: expected_tracer_change_kg = 0.0_dp
        real(dp) :: tracer_balance_residual_kg = 0.0_dp
        real(dp) :: min_cell_mass_kg = 0.0_dp
        real(dp) :: min_tracer_mass_fraction = 0.0_dp
        real(dp) :: max_tracer_mass_fraction = 0.0_dp
    end type transport_budget

    public :: advance_mass_and_tracer

contains

    subroutine advance_mass_and_tracer(dt_s, cell_mass_kg, tracer_mass_kg, &
                                       face_mass_flux_kg_s, face_tracer_mass_fraction, &
                                       cell_mass_source_kg_s, tracer_mass_source_kg_s, &
                                       next_cell_mass_kg, next_tracer_mass_kg, budget, ierr)
        real(dp), intent(in) :: dt_s
        real(dp), intent(in) :: cell_mass_kg(:), tracer_mass_kg(:)
        real(dp), intent(in) :: face_mass_flux_kg_s(:), face_tracer_mass_fraction(:)
        real(dp), intent(in) :: cell_mass_source_kg_s(:), tracer_mass_source_kg_s(:)
        real(dp), intent(out) :: next_cell_mass_kg(:), next_tracer_mass_kg(:)
        type(transport_budget), intent(out) :: budget
        integer, intent(out) :: ierr

        integer :: i, n
        real(dp) :: tracer_flux_left, tracer_flux_right
        real(dp) :: expected_mass_change, expected_tracer_change
        real(dp), allocatable :: candidate_mass(:), candidate_tracer(:), fraction(:)

        next_cell_mass_kg = 0.0_dp
        next_tracer_mass_kg = 0.0_dp
        budget = transport_budget()
        ierr = TRANSPORT_OK

        n = size(cell_mass_kg)
        if (n < 1 .or. size(tracer_mass_kg) /= n .or. &
            size(face_mass_flux_kg_s) /= n + 1 .or. &
            size(face_tracer_mass_fraction) /= n + 1 .or. &
            size(cell_mass_source_kg_s) /= n .or. &
            size(tracer_mass_source_kg_s) /= n .or. &
            size(next_cell_mass_kg) /= n .or. size(next_tracer_mass_kg) /= n) then
            ierr = TRANSPORT_ERR_SIZE
            return
        end if

        if (.not. ieee_is_finite(dt_s)) then
            ierr = TRANSPORT_ERR_NONFINITE
            return
        end if
        if (dt_s < 0.0_dp) then
            ierr = TRANSPORT_ERR_TIMESTEP
            return
        end if

        if (.not. all(ieee_is_finite(cell_mass_kg)) .or. &
            .not. all(ieee_is_finite(tracer_mass_kg)) .or. &
            .not. all(ieee_is_finite(face_mass_flux_kg_s)) .or. &
            .not. all(ieee_is_finite(face_tracer_mass_fraction)) .or. &
            .not. all(ieee_is_finite(cell_mass_source_kg_s)) .or. &
            .not. all(ieee_is_finite(tracer_mass_source_kg_s))) then
            ierr = TRANSPORT_ERR_NONFINITE
            return
        end if

        if (any(cell_mass_kg <= 0.0_dp) .or. any(tracer_mass_kg < 0.0_dp) .or. &
            any(tracer_mass_kg > cell_mass_kg)) then
            ierr = TRANSPORT_ERR_STATE
            return
        end if
        if (any(face_tracer_mass_fraction < 0.0_dp) .or. &
            any(face_tracer_mass_fraction > 1.0_dp)) then
            ierr = TRANSPORT_ERR_FRACTION
            return
        end if

        allocate(candidate_mass(n), candidate_tracer(n), fraction(n))

        do i = 1, n
            tracer_flux_left = face_mass_flux_kg_s(i) * face_tracer_mass_fraction(i)
            tracer_flux_right = face_mass_flux_kg_s(i + 1) * face_tracer_mass_fraction(i + 1)

            candidate_mass(i) = cell_mass_kg(i) + dt_s * ( &
                face_mass_flux_kg_s(i) - face_mass_flux_kg_s(i + 1) + cell_mass_source_kg_s(i))
            candidate_tracer(i) = tracer_mass_kg(i) + dt_s * ( &
                tracer_flux_left - tracer_flux_right + tracer_mass_source_kg_s(i))
        end do

        if (.not. all(ieee_is_finite(candidate_mass)) .or. &
            .not. all(ieee_is_finite(candidate_tracer)) .or. &
            any(candidate_mass <= 0.0_dp) .or. any(candidate_tracer < 0.0_dp) .or. &
            any(candidate_tracer > candidate_mass)) then
            ierr = TRANSPORT_ERR_RESULT
            return
        end if

        fraction = candidate_tracer / candidate_mass
        if (.not. all(ieee_is_finite(fraction)) .or. any(fraction < 0.0_dp) .or. any(fraction > 1.0_dp)) then
            ierr = TRANSPORT_ERR_RESULT
            return
        end if

        budget%lower_boundary_mass_change_kg = dt_s * face_mass_flux_kg_s(1)
        budget%upper_boundary_mass_change_kg = -dt_s * face_mass_flux_kg_s(n + 1)
        budget%resolved_mass_source_change_kg = dt_s * sum(max(cell_mass_source_kg_s, 0.0_dp))
        budget%resolved_mass_sink_change_kg = dt_s * sum(min(cell_mass_source_kg_s, 0.0_dp))
        expected_mass_change = budget%lower_boundary_mass_change_kg + &
            budget%upper_boundary_mass_change_kg + &
            budget%resolved_mass_source_change_kg + budget%resolved_mass_sink_change_kg

        budget%lower_boundary_tracer_change_kg = &
            dt_s * face_mass_flux_kg_s(1) * face_tracer_mass_fraction(1)
        budget%upper_boundary_tracer_change_kg = &
            -dt_s * face_mass_flux_kg_s(n + 1) * face_tracer_mass_fraction(n + 1)
        budget%resolved_tracer_source_change_kg = dt_s * sum(max(tracer_mass_source_kg_s, 0.0_dp))
        budget%resolved_tracer_sink_change_kg = dt_s * sum(min(tracer_mass_source_kg_s, 0.0_dp))
        expected_tracer_change = budget%lower_boundary_tracer_change_kg + &
            budget%upper_boundary_tracer_change_kg + &
            budget%resolved_tracer_source_change_kg + budget%resolved_tracer_sink_change_kg

        budget%mass_before_kg = sum(cell_mass_kg)
        budget%mass_after_kg = sum(candidate_mass)
        budget%expected_mass_change_kg = expected_mass_change
        budget%mass_balance_residual_kg = budget%mass_after_kg - budget%mass_before_kg - expected_mass_change
        budget%tracer_before_kg = sum(tracer_mass_kg)
        budget%tracer_after_kg = sum(candidate_tracer)
        budget%expected_tracer_change_kg = expected_tracer_change
        budget%tracer_balance_residual_kg = &
            budget%tracer_after_kg - budget%tracer_before_kg - expected_tracer_change
        budget%min_cell_mass_kg = minval(candidate_mass)
        budget%min_tracer_mass_fraction = minval(fraction)
        budget%max_tracer_mass_fraction = maxval(fraction)

        next_cell_mass_kg = candidate_mass
        next_tracer_mass_kg = candidate_tracer
    end subroutine advance_mass_and_tracer

end module climate_conservative_transport
