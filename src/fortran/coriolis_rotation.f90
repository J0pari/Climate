module climate_coriolis_rotation
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64
    integer, parameter, public :: CORIOLIS_OK = 0
    integer, parameter, public :: CORIOLIS_ERR_NONFINITE = 1

    public :: rotate_coriolis_pair

contains

    subroutine rotate_coriolis_pair(u_ms, v_ms, coriolis_s_inv, dt_s, &
                                    u_next_ms, v_next_ms, ierr)
        real(dp), intent(in) :: u_ms, v_ms
        real(dp), intent(in) :: coriolis_s_inv, dt_s
        real(dp), intent(out) :: u_next_ms, v_next_ms
        integer, intent(out) :: ierr

        real(dp) :: angle, c, s

        ierr = CORIOLIS_OK
        u_next_ms = 0.0_dp
        v_next_ms = 0.0_dp

        if (.not. ieee_is_finite(u_ms) .or. .not. ieee_is_finite(v_ms) .or. &
            .not. ieee_is_finite(coriolis_s_inv) .or. .not. ieee_is_finite(dt_s)) then
            ierr = CORIOLIS_ERR_NONFINITE
            return
        end if

        ! For the primitive-equation sign convention
        !   du/dt = +f v,
        !   dv/dt = -f u,
        ! the Coriolis generator is skew-symmetric and its exact exponential is
        ! a rotation through angle f*dt. Negative dt is intentionally supported
        ! so reversibility is an executable property rather than a special case.
        angle = coriolis_s_inv * dt_s
        c = cos(angle)
        s = sin(angle)

        u_next_ms = c * u_ms + s * v_ms
        v_next_ms = -s * u_ms + c * v_ms

        if (.not. ieee_is_finite(u_next_ms) .or. .not. ieee_is_finite(v_next_ms)) then
            u_next_ms = 0.0_dp
            v_next_ms = 0.0_dp
            ierr = CORIOLIS_ERR_NONFINITE
        end if
    end subroutine rotate_coriolis_pair

end module climate_coriolis_rotation
