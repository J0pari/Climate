module climate_spectral_reference
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private

    integer, parameter, public :: dp = real64
    real(dp), parameter, public :: PI = acos(-1.0_dp)

    integer, parameter, public :: SPECTRAL_OK = 0
    integer, parameter, public :: SPECTRAL_ERR_INVALID_INTERVAL = 1
    integer, parameter, public :: SPECTRAL_ERR_TOO_SHORT = 2

    public :: dft
    public :: analytic_signal
    public :: instantaneous_frequency

contains

    subroutine dft(input, inverse, output)
        complex(dp), intent(in) :: input(:)
        logical, intent(in) :: inverse
        complex(dp), allocatable, intent(out) :: output(:)

        integer :: n, j, k
        real(dp) :: angle, sign
        complex(dp) :: phase

        n = size(input)
        allocate(output(n))
        if (n == 0) return

        if (inverse) then
            sign = 1.0_dp
        else
            sign = -1.0_dp
        end if

        do k = 1, n
            output(k) = cmplx(0.0_dp, 0.0_dp, kind=dp)
            do j = 1, n
                angle = sign * 2.0_dp * PI * real((j - 1) * (k - 1), dp) / real(n, dp)
                phase = cmplx(cos(angle), sin(angle), kind=dp)
                output(k) = output(k) + input(j) * phase
            end do
        end do

        if (inverse) output = output / real(n, dp)
    end subroutine dft


    subroutine analytic_signal(signal, analytic)
        real(dp), intent(in) :: signal(:)
        complex(dp), allocatable, intent(out) :: analytic(:)

        integer :: n
        complex(dp), allocatable :: samples(:), spectrum(:), filtered(:)

        n = size(signal)
        allocate(samples(n))
        samples = cmplx(signal, 0.0_dp, kind=dp)

        call dft(samples, .false., spectrum)
        filtered = spectrum

        if (n > 1) then
            if (mod(n, 2) == 0) then
                ! Even length: preserve DC and Nyquist, double strictly positive
                ! frequencies, and remove strictly negative frequencies.
                if (n / 2 >= 2) filtered(2:n/2) = 2.0_dp * filtered(2:n/2)
                if (n / 2 + 2 <= n) then
                    filtered(n/2 + 2:n) = cmplx(0.0_dp, 0.0_dp, kind=dp)
                end if
            else
                ! Odd length: there is no Nyquist bin.
                filtered(2:(n + 1)/2) = 2.0_dp * filtered(2:(n + 1)/2)
                if ((n + 3)/2 <= n) then
                    filtered((n + 3)/2:n) = cmplx(0.0_dp, 0.0_dp, kind=dp)
                end if
            end if
        end if

        call dft(filtered, .true., analytic)
    end subroutine analytic_signal


    subroutine instantaneous_frequency(signal, sample_interval, frequency, amplitude, phase, ierr)
        real(dp), intent(in) :: signal(:)
        real(dp), intent(in) :: sample_interval
        real(dp), allocatable, intent(out) :: frequency(:)
        real(dp), allocatable, intent(out) :: amplitude(:)
        real(dp), allocatable, intent(out) :: phase(:)
        integer, intent(out) :: ierr

        integer :: n, k
        real(dp) :: raw_phase, previous_raw, delta, offset
        complex(dp), allocatable :: analytic(:)

        ierr = SPECTRAL_OK
        n = size(signal)

        if (.not. (sample_interval > 0.0_dp)) then
            ierr = SPECTRAL_ERR_INVALID_INTERVAL
            allocate(frequency(0), amplitude(0), phase(0))
            return
        end if

        if (n < 3) then
            ierr = SPECTRAL_ERR_TOO_SHORT
            allocate(frequency(0), amplitude(0), phase(0))
            return
        end if

        call analytic_signal(signal, analytic)
        allocate(frequency(n), amplitude(n), phase(n))

        amplitude = abs(analytic)

        previous_raw = atan2(aimag(analytic(1)), real(analytic(1), dp))
        phase(1) = previous_raw
        offset = 0.0_dp

        do k = 2, n
            raw_phase = atan2(aimag(analytic(k)), real(analytic(k), dp))
            delta = raw_phase - previous_raw
            if (delta > PI) then
                offset = offset - 2.0_dp * PI
            else if (delta < -PI) then
                offset = offset + 2.0_dp * PI
            end if
            phase(k) = raw_phase + offset
            previous_raw = raw_phase
        end do

        ! Return cycles per unit time, not angular frequency.
        frequency(1) = (phase(2) - phase(1)) / (2.0_dp * PI * sample_interval)
        do k = 2, n - 1
            frequency(k) = (phase(k + 1) - phase(k - 1)) / &
                           (4.0_dp * PI * sample_interval)
        end do
        frequency(n) = (phase(n) - phase(n - 1)) / (2.0_dp * PI * sample_interval)
    end subroutine instantaneous_frequency

end module climate_spectral_reference
