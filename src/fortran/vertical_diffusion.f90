module climate_vertical_diffusion
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter :: VBC_UNSPECIFIED = -1
    integer, parameter, public :: VBC_ZERO_FLUX = 0
    integer, parameter, public :: VBC_PRESCRIBED_FLUX = 1
    integer, parameter, public :: VBC_PRESCRIBED_VALUE = 2

    integer, parameter, public :: VDIFF_OK = 0
    integer, parameter, public :: VDIFF_ERR_EMPTY = 1
    integer, parameter, public :: VDIFF_ERR_DIMENSION = 2
    integer, parameter, public :: VDIFF_ERR_GEOMETRY = 3
    integer, parameter, public :: VDIFF_ERR_DIFFUSIVITY = 4
    integer, parameter, public :: VDIFF_ERR_BOUNDARY = 5
    integer, parameter, public :: VDIFF_ERR_NONFINITE = 6
    integer, parameter, public :: VDIFF_ERR_OPERATOR = 7

    type, public :: height_diffusion_boundary
        ! Both components are required in a structure constructor. A boundary
        ! condition is a physical choice and therefore has no implicit default.
        integer :: kind
        ! For PRESCRIBED_VALUE, value has the same units as the transported
        ! scalar. For PRESCRIBED_FLUX, value is the physical upward flux
        ! q=-K d(phi)/dz and therefore has scalar_unit*m/s.
        real(dp) :: value
    end type height_diffusion_boundary

    type, public :: height_diffusion_diagnostics
        real(dp) :: total_height_m = 0.0_dp
        real(dp) :: min_layer_thickness_m = 0.0_dp
        real(dp) :: max_layer_thickness_m = 0.0_dp
        real(dp) :: min_face_diffusivity_m2_s = 0.0_dp
        real(dp) :: max_face_diffusivity_m2_s = 0.0_dp
        real(dp) :: operator_inf_norm_s_inv = 0.0_dp
        real(dp) :: constant_nullspace_defect_s_inv = 0.0_dp
        real(dp) :: weighted_conservation_defect_m_s = 0.0_dp
        logical :: constant_nullspace_expected = .false.
        logical :: conservative_homogeneous_operator_expected = .false.
    end type height_diffusion_diagnostics

    type, public :: height_diffusion_operator
        integer :: n = 0
        ! Geometric-height finite volumes, per unit horizontal area.
        real(dp), allocatable :: layer_thickness_m(:)
        real(dp), allocatable :: face_diffusivity_m2_s(:)
        ! Bands multiply the transported scalar and therefore have s^-1.
        real(dp), allocatable :: lower_s_inv(:)
        real(dp), allocatable :: diagonal_s_inv(:)
        real(dp), allocatable :: upper_s_inv(:)
        ! Additive source has transported_scalar_unit/s.
        real(dp), allocatable :: source_per_s(:)
        type(height_diffusion_boundary) :: bottom_boundary
        type(height_diffusion_boundary) :: top_boundary
        type(height_diffusion_diagnostics) :: diagnostics
    end type height_diffusion_operator

    public :: build_height_diffusion_operator
    public :: evaluate_height_diffusion

contains

    subroutine reset_operator(operator)
        type(height_diffusion_operator), intent(inout) :: operator

        operator%n = 0
        if (allocated(operator%layer_thickness_m)) deallocate(operator%layer_thickness_m)
        if (allocated(operator%face_diffusivity_m2_s)) deallocate(operator%face_diffusivity_m2_s)
        if (allocated(operator%lower_s_inv)) deallocate(operator%lower_s_inv)
        if (allocated(operator%diagonal_s_inv)) deallocate(operator%diagonal_s_inv)
        if (allocated(operator%upper_s_inv)) deallocate(operator%upper_s_inv)
        if (allocated(operator%source_per_s)) deallocate(operator%source_per_s)
        operator%bottom_boundary%kind = VBC_UNSPECIFIED
        operator%bottom_boundary%value = 0.0_dp
        operator%top_boundary%kind = VBC_UNSPECIFIED
        operator%top_boundary%value = 0.0_dp
        operator%diagnostics = height_diffusion_diagnostics()
    end subroutine reset_operator


    logical function valid_boundary(boundary)
        type(height_diffusion_boundary), intent(in) :: boundary

        valid_boundary = ieee_is_finite(boundary%value) .and. &
            (boundary%kind == VBC_ZERO_FLUX .or. &
             boundary%kind == VBC_PRESCRIBED_FLUX .or. &
             boundary%kind == VBC_PRESCRIBED_VALUE)
        if (valid_boundary .and. boundary%kind == VBC_ZERO_FLUX) then
            valid_boundary = boundary%value == 0.0_dp
        end if
    end function valid_boundary


    subroutine build_height_diffusion_operator(layer_thickness_m, face_diffusivity_m2_s, &
                                               bottom_boundary, top_boundary, operator, ierr)
        real(dp), intent(in) :: layer_thickness_m(:)
        real(dp), intent(in) :: face_diffusivity_m2_s(:)
        type(height_diffusion_boundary), intent(in) :: bottom_boundary, top_boundary
        type(height_diffusion_operator), intent(inout) :: operator
        integer, intent(out) :: ierr

        real(dp) :: center_distance_m, conductance_m_s, rate_s_inv, column_balance_m_s
        integer :: i, n

        call reset_operator(operator)
        ierr = VDIFF_OK
        n = size(layer_thickness_m)

        if (n == 0) then
            ierr = VDIFF_ERR_EMPTY
            return
        end if
        if (size(face_diffusivity_m2_s) /= n + 1) then
            ierr = VDIFF_ERR_DIMENSION
            return
        end if
        if (.not. all(ieee_is_finite(layer_thickness_m)) .or. &
            .not. all(layer_thickness_m > 0.0_dp)) then
            ierr = VDIFF_ERR_GEOMETRY
            return
        end if
        if (.not. all(ieee_is_finite(face_diffusivity_m2_s)) .or. &
            .not. all(face_diffusivity_m2_s >= 0.0_dp)) then
            ierr = VDIFF_ERR_DIFFUSIVITY
            return
        end if
        if (.not. valid_boundary(bottom_boundary) .or. .not. valid_boundary(top_boundary)) then
            ierr = VDIFF_ERR_BOUNDARY
            return
        end if

        operator%n = n
        allocate(operator%layer_thickness_m(n), operator%face_diffusivity_m2_s(n + 1))
        allocate(operator%lower_s_inv(n), operator%diagonal_s_inv(n), &
                 operator%upper_s_inv(n), operator%source_per_s(n))
        operator%layer_thickness_m = layer_thickness_m
        operator%face_diffusivity_m2_s = face_diffusivity_m2_s
        operator%lower_s_inv = 0.0_dp
        operator%diagonal_s_inv = 0.0_dp
        operator%upper_s_inv = 0.0_dp
        operator%source_per_s = 0.0_dp
        operator%bottom_boundary = bottom_boundary
        operator%top_boundary = top_boundary

        operator%diagnostics%total_height_m = sum(layer_thickness_m)
        operator%diagnostics%min_layer_thickness_m = minval(layer_thickness_m)
        operator%diagnostics%max_layer_thickness_m = maxval(layer_thickness_m)
        operator%diagnostics%min_face_diffusivity_m2_s = minval(face_diffusivity_m2_s)
        operator%diagnostics%max_face_diffusivity_m2_s = maxval(face_diffusivity_m2_s)

        ! Interior diffusive flux q=-K d(phi)/dz. Each face contribution is
        ! shared by the adjacent cells with opposite inventory changes, which
        ! makes the per-unit-area finite-volume column exactly conservative.
        do i = 1, n - 1
            center_distance_m = 0.5_dp * (layer_thickness_m(i) + layer_thickness_m(i + 1))
            conductance_m_s = face_diffusivity_m2_s(i + 1) / center_distance_m

            rate_s_inv = conductance_m_s / layer_thickness_m(i)
            operator%diagonal_s_inv(i) = operator%diagonal_s_inv(i) - rate_s_inv
            operator%upper_s_inv(i) = operator%upper_s_inv(i) + rate_s_inv

            rate_s_inv = conductance_m_s / layer_thickness_m(i + 1)
            operator%lower_s_inv(i + 1) = operator%lower_s_inv(i + 1) + rate_s_inv
            operator%diagonal_s_inv(i + 1) = operator%diagonal_s_inv(i + 1) - rate_s_inv
        end do

        ! q is positive in the +z direction. Cell inventory tendency is
        ! q_bottom-q_top. Boundary values use a half-cell distance from the
        ! cell center to the physical boundary face.
        select case (bottom_boundary%kind)
        case (VBC_ZERO_FLUX)
            continue
        case (VBC_PRESCRIBED_FLUX)
            operator%source_per_s(1) = operator%source_per_s(1) + &
                bottom_boundary%value / layer_thickness_m(1)
        case (VBC_PRESCRIBED_VALUE)
            center_distance_m = 0.5_dp * layer_thickness_m(1)
            conductance_m_s = face_diffusivity_m2_s(1) / center_distance_m
            rate_s_inv = conductance_m_s / layer_thickness_m(1)
            operator%diagonal_s_inv(1) = operator%diagonal_s_inv(1) - rate_s_inv
            operator%source_per_s(1) = operator%source_per_s(1) + &
                rate_s_inv * bottom_boundary%value
        end select

        select case (top_boundary%kind)
        case (VBC_ZERO_FLUX)
            continue
        case (VBC_PRESCRIBED_FLUX)
            operator%source_per_s(n) = operator%source_per_s(n) - &
                top_boundary%value / layer_thickness_m(n)
        case (VBC_PRESCRIBED_VALUE)
            center_distance_m = 0.5_dp * layer_thickness_m(n)
            conductance_m_s = face_diffusivity_m2_s(n + 1) / center_distance_m
            rate_s_inv = conductance_m_s / layer_thickness_m(n)
            operator%diagonal_s_inv(n) = operator%diagonal_s_inv(n) - rate_s_inv
            operator%source_per_s(n) = operator%source_per_s(n) + &
                rate_s_inv * top_boundary%value
        end select

        do i = 1, n
            operator%diagnostics%operator_inf_norm_s_inv = max( &
                operator%diagnostics%operator_inf_norm_s_inv, &
                abs(operator%lower_s_inv(i)) + abs(operator%diagonal_s_inv(i)) + &
                abs(operator%upper_s_inv(i)))
        end do

        operator%diagnostics%constant_nullspace_expected = &
            bottom_boundary%kind /= VBC_PRESCRIBED_VALUE .and. &
            top_boundary%kind /= VBC_PRESCRIBED_VALUE
        operator%diagnostics%conservative_homogeneous_operator_expected = &
            operator%diagnostics%constant_nullspace_expected

        if (operator%diagnostics%constant_nullspace_expected) then
            operator%diagnostics%constant_nullspace_defect_s_inv = maxval(abs( &
                operator%lower_s_inv + operator%diagonal_s_inv + operator%upper_s_inv))

            do i = 1, n
                column_balance_m_s = layer_thickness_m(i) * operator%diagonal_s_inv(i)
                if (i > 1) column_balance_m_s = column_balance_m_s + &
                    layer_thickness_m(i - 1) * operator%upper_s_inv(i - 1)
                if (i < n) column_balance_m_s = column_balance_m_s + &
                    layer_thickness_m(i + 1) * operator%lower_s_inv(i + 1)
                operator%diagnostics%weighted_conservation_defect_m_s = max( &
                    operator%diagnostics%weighted_conservation_defect_m_s, abs(column_balance_m_s))
            end do
        end if
    end subroutine build_height_diffusion_operator


    subroutine evaluate_height_diffusion(operator, state, tendency, ierr)
        type(height_diffusion_operator), intent(in) :: operator
        real(dp), intent(in) :: state(:)
        real(dp), allocatable, intent(out) :: tendency(:)
        integer, intent(out) :: ierr

        integer :: i, n

        ierr = VDIFF_OK
        n = operator%n
        if (n <= 0 .or. .not. allocated(operator%diagonal_s_inv)) then
            ierr = VDIFF_ERR_OPERATOR
            allocate(tendency(0))
            return
        end if
        if (size(state) /= n) then
            ierr = VDIFF_ERR_DIMENSION
            allocate(tendency(0))
            return
        end if
        if (.not. all(ieee_is_finite(state))) then
            ierr = VDIFF_ERR_NONFINITE
            allocate(tendency(0))
            return
        end if

        allocate(tendency(n))
        tendency = operator%diagonal_s_inv * state + operator%source_per_s
        do i = 2, n
            tendency(i) = tendency(i) + operator%lower_s_inv(i) * state(i - 1)
        end do
        do i = 1, n - 1
            tendency(i) = tendency(i) + operator%upper_s_inv(i) * state(i + 1)
        end do

        if (.not. all(ieee_is_finite(tendency))) then
            deallocate(tendency)
            allocate(tendency(0))
            ierr = VDIFF_ERR_NONFINITE
        end if
    end subroutine evaluate_height_diffusion

end module climate_vertical_diffusion
