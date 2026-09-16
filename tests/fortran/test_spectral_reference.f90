program test_spectral_reference
    use climate_spectral_reference, only: dp, PI, SPECTRAL_OK, &
        SPECTRAL_ERR_INVALID_INTERVAL, dft, instantaneous_frequency
    implicit none

    call test_dft_round_trip()
    call test_sine_frequency()
    call test_invalid_interval()

contains

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'FAIL: ' // trim(message)
            error stop 1
        end if
    end subroutine require


    subroutine test_dft_round_trip()
        complex(dp) :: input(4)
        complex(dp), allocatable :: spectrum(:), reconstructed(:)
        real(dp) :: error_norm

        input = [ &
            cmplx(1.0_dp, 0.5_dp, kind=dp), &
            cmplx(-2.0_dp, 1.0_dp, kind=dp), &
            cmplx(0.25_dp, -0.75_dp, kind=dp), &
            cmplx(3.0_dp, 0.0_dp, kind=dp) &
        ]

        call dft(input, .false., spectrum)
        call dft(spectrum, .true., reconstructed)
        error_norm = maxval(abs(reconstructed - input))

        call require(error_norm < 1.0e-12_dp, 'DFT inverse must recover input')
    end subroutine test_dft_round_trip


    subroutine test_sine_frequency()
        integer, parameter :: n = 128
        real(dp), parameter :: dt = 0.25_dp
        real(dp), parameter :: expected_frequency = 0.5_dp
        real(dp) :: signal(n)
        real(dp), allocatable :: frequency(:), amplitude(:), phase(:)
        real(dp) :: mean_frequency, mean_amplitude
        integer :: k, ierr

        do k = 1, n
            signal(k) = sin(2.0_dp * PI * expected_frequency * &
                            real(k - 1, dp) * dt)
        end do

        call instantaneous_frequency(signal, dt, frequency, amplitude, phase, ierr)
        call require(ierr == SPECTRAL_OK, 'sine frequency analysis must succeed')

        mean_frequency = sum(frequency(2:n - 1)) / real(n - 2, dp)
        mean_amplitude = sum(amplitude) / real(n, dp)

        call require(abs(mean_frequency - expected_frequency) < 1.0e-10_dp, &
                     'instantaneous frequency must recover known sinusoid')
        call require(abs(mean_amplitude - 1.0_dp) < 1.0e-10_dp, &
                     'analytic-signal amplitude must recover unit sinusoid')
    end subroutine test_sine_frequency


    subroutine test_invalid_interval()
        real(dp) :: signal(3)
        real(dp), allocatable :: frequency(:), amplitude(:), phase(:)
        integer :: ierr

        signal = [0.0_dp, 1.0_dp, 0.0_dp]
        call instantaneous_frequency(signal, 0.0_dp, frequency, amplitude, phase, ierr)

        call require(ierr == SPECTRAL_ERR_INVALID_INTERVAL, &
                     'non-positive sample interval must fail closed')
        call require(size(frequency) == 0, 'failed analysis must not emit frequency data')
        call require(size(amplitude) == 0, 'failed analysis must not emit amplitude data')
        call require(size(phase) == 0, 'failed analysis must not emit phase data')
    end subroutine test_invalid_interval

end program test_spectral_reference
