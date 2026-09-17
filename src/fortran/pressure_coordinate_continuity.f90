module climate_pressure_coordinate_continuity
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use climate_pressure_coordinate_grid, only: &
        validate_pressure_interfaces, PRESSURE_GRID_OK, PRESSURE_GRID_ERR_SIZE, &
        PRESSURE_GRID_ERR_NONFINITE, PRESSURE_GRID_ERR_PRESSURE, &
        PRESSURE_GRID_ERR_ORDER
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: CONTINUITY_OK = 0
    integer, parameter, public :: CONTINUITY_ERR_SIZE = 1
    integer, parameter, public :: CONTINUITY_ERR_NONFINITE = 2
    integer, parameter, public :: CONTINUITY_ERR_PRESSURE = 3
    integer, parameter, public :: CONTINUITY_ERR_PRESSURE_ORDER = 4
    integer, parameter, public :: CONTINUITY_ERR_RESULT = 5

    public :: integrate_pressure_velocity
    public :: evaluate_pressure_continuity

contains

    integer function map_pressure_grid_error(grid_ierr) result(ierr)
        integer, intent(in) :: grid_ierr

        select case (grid_ierr)
        case (PRESSURE_GRID_OK)
            ierr = CONTINUITY_OK
        case (PRESSURE_GRID_ERR_SIZE)
            ierr = CONTINUITY_ERR_SIZE
        case (PRESSURE_GRID_ERR_NONFINITE)
            ierr = CONTINUITY_ERR_NONFINITE
        case (PRESSURE_GRID_ERR_PRESSURE)
            ierr = CONTINUITY_ERR_PRESSURE
        case (PRESSURE_GRID_ERR_ORDER)
            ierr = CONTINUITY_ERR_PRESSURE_ORDER
        case default
            ierr = CONTINUITY_ERR_RESULT
        end select
    end function map_pressure_grid_error


    subroutine validate_pressure_inputs(interface_pressure_pa, &
                                        layer_pressure_mean_horizontal_divergence_s1, ierr)
        real(dp), intent(in) :: interface_pressure_pa(:)
        real(dp), intent(in) :: layer_pressure_mean_horizontal_divergence_s1(:)
        integer, intent(out) :: ierr

        integer :: n_layers, grid_ierr

        ierr = CONTINUITY_OK
        n_layers = size(layer_pressure_mean_horizontal_divergence_s1)

        if (n_layers < 1 .or. size(interface_pressure_pa) /= n_layers + 1) then
            ierr = CONTINUITY_ERR_SIZE
            return
        end if
        if (.not. all(ieee_is_finite(layer_pressure_mean_horizontal_divergence_s1))) then
            ierr = CONTINUITY_ERR_NONFINITE
            return
        end if

        call validate_pressure_interfaces(interface_pressure_pa, grid_ierr)
        ierr = map_pressure_grid_error(grid_ierr)
    end subroutine validate_pressure_inputs


    subroutine integrate_pressure_velocity(interface_pressure_pa, &
                                           layer_pressure_mean_horizontal_divergence_s1, &
                                           lower_boundary_omega_pa_s, interface_omega_pa_s, &
                                           column_pressure_divergence_pa_s, ierr)
        real(dp), intent(in) :: interface_pressure_pa(:)
        real(dp), intent(in) :: layer_pressure_mean_horizontal_divergence_s1(:)
        real(dp), intent(in) :: lower_boundary_omega_pa_s
        real(dp), intent(out) :: interface_omega_pa_s(:)
        real(dp), intent(out) :: column_pressure_divergence_pa_s
        integer, intent(out) :: ierr

        integer :: n_layers, k
        real(dp) :: delta_pressure_pa, increment_pa_s

        interface_omega_pa_s = 0.0_dp
        column_pressure_divergence_pa_s = 0.0_dp
        ierr = CONTINUITY_OK

        n_layers = size(layer_pressure_mean_horizontal_divergence_s1)
        if (size(interface_omega_pa_s) /= n_layers + 1) then
            ierr = CONTINUITY_ERR_SIZE
            return
        end if

        call validate_pressure_inputs(interface_pressure_pa, &
            layer_pressure_mean_horizontal_divergence_s1, ierr)
        if (ierr /= CONTINUITY_OK) return

        if (.not. ieee_is_finite(lower_boundary_omega_pa_s)) then
            ierr = CONTINUITY_ERR_NONFINITE
            return
        end if

        interface_omega_pa_s(1) = lower_boundary_omega_pa_s
        do k = 1, n_layers
            delta_pressure_pa = interface_pressure_pa(k) - interface_pressure_pa(k + 1)
            increment_pa_s = delta_pressure_pa * &
                layer_pressure_mean_horizontal_divergence_s1(k)
            column_pressure_divergence_pa_s = column_pressure_divergence_pa_s + increment_pa_s
            interface_omega_pa_s(k + 1) = interface_omega_pa_s(k) + increment_pa_s
        end do

        if (.not. all(ieee_is_finite(interface_omega_pa_s)) .or. &
            .not. ieee_is_finite(column_pressure_divergence_pa_s)) then
            interface_omega_pa_s = 0.0_dp
            column_pressure_divergence_pa_s = 0.0_dp
            ierr = CONTINUITY_ERR_RESULT
        end if
    end subroutine integrate_pressure_velocity


    subroutine evaluate_pressure_continuity(interface_pressure_pa, &
                                            layer_pressure_mean_horizontal_divergence_s1, &
                                            interface_omega_pa_s, layer_residual_pa_s, &
                                            column_residual_pa_s, ierr)
        real(dp), intent(in) :: interface_pressure_pa(:)
        real(dp), intent(in) :: layer_pressure_mean_horizontal_divergence_s1(:)
        real(dp), intent(in) :: interface_omega_pa_s(:)
        real(dp), intent(out) :: layer_residual_pa_s(:)
        real(dp), intent(out) :: column_residual_pa_s
        integer, intent(out) :: ierr

        integer :: n_layers, k
        real(dp) :: delta_pressure_pa, integrated_divergence_pa_s

        layer_residual_pa_s = 0.0_dp
        column_residual_pa_s = 0.0_dp
        ierr = CONTINUITY_OK

        n_layers = size(layer_pressure_mean_horizontal_divergence_s1)
        if (size(interface_omega_pa_s) /= n_layers + 1 .or. &
            size(layer_residual_pa_s) /= n_layers) then
            ierr = CONTINUITY_ERR_SIZE
            return
        end if

        call validate_pressure_inputs(interface_pressure_pa, &
            layer_pressure_mean_horizontal_divergence_s1, ierr)
        if (ierr /= CONTINUITY_OK) return

        if (.not. all(ieee_is_finite(interface_omega_pa_s))) then
            ierr = CONTINUITY_ERR_NONFINITE
            return
        end if

        integrated_divergence_pa_s = 0.0_dp
        do k = 1, n_layers
            delta_pressure_pa = interface_pressure_pa(k) - interface_pressure_pa(k + 1)
            integrated_divergence_pa_s = integrated_divergence_pa_s + &
                delta_pressure_pa * layer_pressure_mean_horizontal_divergence_s1(k)
            layer_residual_pa_s(k) = interface_omega_pa_s(k + 1) - &
                interface_omega_pa_s(k) - &
                delta_pressure_pa * layer_pressure_mean_horizontal_divergence_s1(k)
        end do

        column_residual_pa_s = interface_omega_pa_s(n_layers + 1) - &
            interface_omega_pa_s(1) - integrated_divergence_pa_s

        if (.not. all(ieee_is_finite(layer_residual_pa_s)) .or. &
            .not. ieee_is_finite(column_residual_pa_s)) then
            layer_residual_pa_s = 0.0_dp
            column_residual_pa_s = 0.0_dp
            ierr = CONTINUITY_ERR_RESULT
        end if
    end subroutine evaluate_pressure_continuity

end module climate_pressure_coordinate_continuity
