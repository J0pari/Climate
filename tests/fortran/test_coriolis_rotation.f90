program test_coriolis_rotation
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use climate_coriolis_rotation, only: dp, CORIOLIS_OK, CORIOLIS_ERR_NONFINITE, &
        rotate_coriolis_pair
    implicit none

    call test_quarter_turn()
    call test_energy_preservation()
    call test_reversibility()
    call test_group_composition()
    call test_infinitesimal_limit()
    call test_nonfinite_fails_closed()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_quarter_turn()
        real(dp), parameter :: pi = acos(-1.0_dp)
        real(dp) :: u_next, v_next
        integer :: ierr

        call rotate_coriolis_pair(1.0_dp, 0.0_dp, 1.0_dp, 0.5_dp * pi, u_next, v_next, ierr)
        call require(ierr == CORIOLIS_OK, 'quarter-turn exact Coriolis solve must succeed')
        call require(abs(u_next) < 1.0e-14_dp, 'quarter turn must rotate u to zero')
        call require(abs(v_next + 1.0_dp) < 1.0e-14_dp, &
                     'primitive-equation sign convention must rotate positive u toward negative v')
    end subroutine test_quarter_turn


    subroutine test_energy_preservation()
        real(dp) :: u, v, u_next, v_next, speed2_before, speed2_after
        integer :: ierr

        u = 12.5_dp
        v = -7.25_dp
        call rotate_coriolis_pair(u, v, 1.1e-4_dp, 1800.0_dp, u_next, v_next, ierr)
        call require(ierr == CORIOLIS_OK, 'finite Coriolis rotation must succeed')
        speed2_before = u*u + v*v
        speed2_after = u_next*u_next + v_next*v_next
        call require(abs(speed2_after - speed2_before) < 5.0e-13_dp * speed2_before, &
                     'isolated exact Coriolis rotation must preserve horizontal kinetic-energy norm')
    end subroutine test_energy_preservation


    subroutine test_reversibility()
        real(dp) :: u1, v1, u2, v2
        integer :: ierr

        call rotate_coriolis_pair(5.0_dp, 9.0_dp, 8.0e-5_dp, 2700.0_dp, u1, v1, ierr)
        call require(ierr == CORIOLIS_OK, 'forward Coriolis rotation must succeed')
        call rotate_coriolis_pair(u1, v1, 8.0e-5_dp, -2700.0_dp, u2, v2, ierr)
        call require(ierr == CORIOLIS_OK, 'negative-time Coriolis rotation must succeed')
        call require(abs(u2 - 5.0_dp) < 1.0e-13_dp .and. abs(v2 - 9.0_dp) < 1.0e-13_dp, &
                     'exact Coriolis rotation must be reversible under dt -> -dt')
    end subroutine test_reversibility


    subroutine test_group_composition()
        real(dp) :: ua, va, ub, vb, uc, vc
        real(dp), parameter :: f = 9.0e-5_dp, dt1 = 700.0_dp, dt2 = 1300.0_dp
        integer :: ierr

        call rotate_coriolis_pair(4.0_dp, -3.0_dp, f, dt1, ua, va, ierr)
        call require(ierr == CORIOLIS_OK, 'first composed Coriolis step must succeed')
        call rotate_coriolis_pair(ua, va, f, dt2, ub, vb, ierr)
        call require(ierr == CORIOLIS_OK, 'second composed Coriolis step must succeed')
        call rotate_coriolis_pair(4.0_dp, -3.0_dp, f, dt1 + dt2, uc, vc, ierr)
        call require(ierr == CORIOLIS_OK, 'single combined Coriolis step must succeed')
        call require(abs(ub - uc) < 1.0e-13_dp .and. abs(vb - vc) < 1.0e-13_dp, &
                     'constant-f exact Coriolis flow must satisfy the one-parameter group law')
    end subroutine test_group_composition


    subroutine test_infinitesimal_limit()
        real(dp), parameter :: u = 2.0_dp, v = -5.0_dp, f = 1.0e-4_dp, dt = 1.0e-3_dp
        real(dp) :: u_next, v_next, du_dt, dv_dt
        integer :: ierr

        call rotate_coriolis_pair(u, v, f, dt, u_next, v_next, ierr)
        call require(ierr == CORIOLIS_OK, 'infinitesimal Coriolis witness must succeed')
        du_dt = (u_next - u) / dt
        dv_dt = (v_next - v) / dt
        call require(abs(du_dt - f*v) < 1.0e-8_dp, &
                     'exact rotation must recover du/dt = f v in the small-step limit')
        call require(abs(dv_dt + f*u) < 1.0e-8_dp, &
                     'exact rotation must recover dv/dt = -f u in the small-step limit')
    end subroutine test_infinitesimal_limit


    subroutine test_nonfinite_fails_closed()
        real(dp) :: nan_value, u_next, v_next
        integer :: ierr

        nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
        call rotate_coriolis_pair(1.0_dp, 2.0_dp, nan_value, 1.0_dp, u_next, v_next, ierr)
        call require(ierr == CORIOLIS_ERR_NONFINITE, 'non-finite Coriolis parameter must fail closed')
        call require(u_next == 0.0_dp .and. v_next == 0.0_dp, &
                     'failed Coriolis rotation must not emit partially updated velocity')
    end subroutine test_nonfinite_fails_closed

end program test_coriolis_rotation
