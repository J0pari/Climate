! Climate oscillation capability stub
!
! The previous executable prototype was removed under the semantic sanitation
! policy because it mixed arbitrary/fake forecast skill, incomplete transforms,
! uninitialized-data execution, and broad scientific labels. Returning plausible
! values from that implementation was more dangerous than leaving the capability
! unavailable.
!
! Design and reintroduction requirements:
!   blueprints/oscillation-monitor.md
!   docs/SEMANTIC-SANITATION.md
!
! This file intentionally provides no scientific result-producing API.
! Consumers must treat the capability as unavailable until a replacement is
! introduced with narrow contracts and verification.

module oscillations
    implicit none
    private

    public :: oscillation_monitor_unavailable

contains

    subroutine oscillation_monitor_unavailable()
        error stop "Climate oscillation monitoring is unavailable: legacy placeholder implementation removed; see blueprints/oscillation-monitor.md"
    end subroutine oscillation_monitor_unavailable

end module oscillations
