! Climate spectral/time-frequency capability stub
!
! The previous omnibus prototype was removed under the semantic sanitation
! policy because several advertised algorithms were empty, returned zero-filled
! scientific outputs, or had unresolved implementation/interface errors.
!
! Reintroduction blueprint:
!   blueprints/spectral-analysis.md
! Policy:
!   docs/SEMANTIC-SANITATION.md
!
! This module intentionally exposes only an unavailable entrypoint. Individual
! spectral methods should return as narrow, independently verified modules.

module climate_spectral_analysis
    implicit none
    private

    public :: spectral_analysis_unavailable

contains

    subroutine spectral_analysis_unavailable()
        error stop "Climate spectral analysis is unavailable: misleading omnibus prototype removed; see blueprints/spectral-analysis.md"
    end subroutine spectral_analysis_unavailable

end module climate_spectral_analysis
