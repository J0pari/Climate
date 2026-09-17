module climate_thermodynamic_reference_values
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private

    integer, parameter, public :: dp = real64

    ! Versioned conventional reference values for the repository's simple
    ! ideal-gas atmosphere kernels. These are defaults, not universal physical
    ! truths or a runtime configuration object. Domain kernels copy only the
    ! values they need into their own narrow, caller-overridable parameter types.
    character(len=*), parameter, public :: THERMODYNAMIC_REFERENCE_SET_ID = &
        'climate.thermodynamic_reference.v1'

    real(dp), parameter, public :: REFERENCE_PRESSURE_PA = 100000.0_dp
    real(dp), parameter, public :: DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05_dp
    real(dp), parameter, public :: DRY_AIR_HEAT_CAPACITY_CP_J_KG_K = 1004.0_dp
    real(dp), parameter, public :: WATER_VAPOR_GAS_CONSTANT_J_KG_K = 461.5_dp

end module climate_thermodynamic_reference_values
