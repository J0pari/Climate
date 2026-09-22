module climate_integration_restart
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: dp = real64

    integer, parameter, public :: RESTART_OK = 0
    integer, parameter, public :: RESTART_ERR_EMPTY_STATE = 1
    integer, parameter, public :: RESTART_ERR_NONFINITE = 2
    integer, parameter, public :: RESTART_ERR_IDENTITY = 3
    integer, parameter, public :: RESTART_ERR_UNINITIALIZED = 4

    type, public :: restart_record
        real(dp) :: time = 0.0_dp
        real(dp), allocatable :: physical_state(:)
        character(len=:), allocatable :: solver_id
        character(len=:), allocatable :: configuration_id
        character(len=:), allocatable :: stochastic_state_id
        character(len=:), allocatable :: provenance_id
    end type restart_record

    public :: capture_restart
    public :: validate_restart
    public :: restore_restart

contains

    subroutine capture_restart( &
        time, physical_state, solver_id, configuration_id, stochastic_state_id, &
        provenance_id, record, ierr)
        real(dp), intent(in) :: time
        real(dp), intent(in) :: physical_state(:)
        character(len=*), intent(in) :: solver_id
        character(len=*), intent(in) :: configuration_id
        character(len=*), intent(in) :: stochastic_state_id
        character(len=*), intent(in) :: provenance_id
        type(restart_record), intent(out) :: record
        integer, intent(out) :: ierr

        ierr = RESTART_OK
        if (size(physical_state) == 0) then
            ierr = RESTART_ERR_EMPTY_STATE
            return
        end if
        if (.not. ieee_is_finite(time) .or. &
            .not. all(ieee_is_finite(physical_state))) then
            ierr = RESTART_ERR_NONFINITE
            return
        end if
        if (len_trim(solver_id) == 0 .or. len_trim(configuration_id) == 0 .or. &
            len_trim(stochastic_state_id) == 0 .or. len_trim(provenance_id) == 0) then
            ierr = RESTART_ERR_IDENTITY
            return
        end if

        record%time = time
        allocate(record%physical_state(size(physical_state)))
        record%physical_state = physical_state
        record%solver_id = trim(solver_id)
        record%configuration_id = trim(configuration_id)
        record%stochastic_state_id = trim(stochastic_state_id)
        record%provenance_id = trim(provenance_id)
    end subroutine capture_restart


    subroutine validate_restart(record, ierr)
        type(restart_record), intent(in) :: record
        integer, intent(out) :: ierr

        ierr = RESTART_OK
        if (.not. allocated(record%physical_state) .or. &
            .not. allocated(record%solver_id) .or. &
            .not. allocated(record%configuration_id) .or. &
            .not. allocated(record%stochastic_state_id) .or. &
            .not. allocated(record%provenance_id)) then
            ierr = RESTART_ERR_UNINITIALIZED
            return
        end if
        if (size(record%physical_state) == 0) then
            ierr = RESTART_ERR_EMPTY_STATE
            return
        end if
        if (.not. ieee_is_finite(record%time) .or. &
            .not. all(ieee_is_finite(record%physical_state))) then
            ierr = RESTART_ERR_NONFINITE
            return
        end if
        if (len_trim(record%solver_id) == 0 .or. &
            len_trim(record%configuration_id) == 0 .or. &
            len_trim(record%stochastic_state_id) == 0 .or. &
            len_trim(record%provenance_id) == 0) then
            ierr = RESTART_ERR_IDENTITY
        end if
    end subroutine validate_restart


    subroutine restore_restart(record, time, physical_state, ierr)
        type(restart_record), intent(in) :: record
        real(dp), intent(out) :: time
        real(dp), allocatable, intent(out) :: physical_state(:)
        integer, intent(out) :: ierr

        call validate_restart(record, ierr)
        if (ierr /= RESTART_OK) then
            time = 0.0_dp
            allocate(physical_state(0))
            return
        end if

        time = record%time
        allocate(physical_state(size(record%physical_state)))
        physical_state = record%physical_state
    end subroutine restore_restart

end module climate_integration_restart
