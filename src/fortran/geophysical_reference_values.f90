module climate_geophysical_reference_values
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private

    integer, parameter, public :: dp = real64

    ! Standard gravity is a shared geophysical reference value, not a hidden
    ! per-kernel literal. Kernels that need spatially varying or model-specific
    ! gravity remain free to override their own narrow parameter records.
    character(len=*), parameter, public :: GEOPHYSICAL_REFERENCE_SET_ID = &
        'si.standard_gravity.v1'

    real(dp), parameter, public :: STANDARD_GRAVITY_M_S2 = 9.80665_dp

end module climate_geophysical_reference_values
