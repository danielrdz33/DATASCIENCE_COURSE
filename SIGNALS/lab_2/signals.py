"""Generate discrete time signals.

Python versions of the signal generators from lab 1. Every
generator has the same form as the function crcbgenqcsig.m:

    sig_vec = gen_xxx_sig(data_x, snr, params)

data_x is the vector of time stamps at which the samples of the signal are
to be computed, snr is the matched filtering signal-to-noise ratio of the
signal (it sets the amplitude A so that the norm of the signal equals snr),
and params is the list of signal parameters described in each function.

Daniel Rodriguez, September 2026
"""

import numpy as np


def _scale_to_snr(sig_vec, snr):
    """Scale a signal so that its norm equals snr (sets the amplitude A)."""
    return snr * sig_vec / np.linalg.norm(sig_vec)


def gen_qc_sig(data_x, snr, qc_coefs):
    """Generate a quadratic chirp signal.

    S = GEN_QC_SIG(X, SNR, C)
    Generates a quadratic chirp signal S. X is the vector of time stamps at
    which the samples of the signal are to be computed. SNR is the matched
    filtering signal-to-noise ratio of S and C is the vector of three
    coefficients [a1, a2, a3] that parametrize the phase of the signal:
    a1*t + a2*t^2 + a3*t^3.
    """
    phase_vec = (qc_coefs[0] * data_x + qc_coefs[1] * data_x**2
                 + qc_coefs[2] * data_x**3)
    sig_vec = np.sin(2 * np.pi * phase_vec)
    return _scale_to_snr(sig_vec, snr)


def gen_sin_sig(data_x, snr, sin_params):
    """Generate a sinusoidal signal.

    S = GEN_SIN_SIG(X, SNR, P)
    Generates a sinusoidal signal S. X is the vector of time stamps at which
    the samples of the signal are to be computed. SNR is the matched
    filtering signal-to-noise ratio of S and P is the vector of two
    parameters [f0, phi0] of the signal: sin(2*pi*f0*t + phi0), where f0 is
    the frequency in Hz and phi0 is the initial phase in radians.
    """
    sig_vec = np.sin(2 * np.pi * sin_params[0] * data_x + sin_params[1])
    return _scale_to_snr(sig_vec, snr)


def gen_lc_sig(data_x, snr, lc_params):
    """Generate a linear chirp signal.

    S = GEN_LC_SIG(X, SNR, P)
    Generates a linear chirp signal S. X is the vector of time stamps at
    which the samples of the signal are to be computed. SNR is the matched
    filtering signal-to-noise ratio of S and P is the vector of three
    parameters [f0, f1, phi0] of the signal:
    sin(2*pi*(f0*t + 0.5*f1*t^2) + phi0), where f0 is the starting frequency
    in Hz, f1 is the rate of increase of frequency in Hz/sec and phi0 is the
    initial phase in radians.
    """
    phase_vec = (2 * np.pi * (lc_params[0] * data_x
                              + 0.5 * lc_params[1] * data_x**2)
                 + lc_params[2])
    sig_vec = np.sin(phase_vec)
    return _scale_to_snr(sig_vec, snr)


def gen_sg_sig(data_x, snr, sg_params):
    """Generate a sine-Gaussian signal.

    S = GEN_SG_SIG(X, SNR, P)
    Generates a sine-Gaussian signal S. X is the vector of time stamps at
    which the samples of the signal are to be computed. SNR is the matched
    filtering signal-to-noise ratio of S and P is the vector of four
    parameters [t0, sigma, f0, phi0] of the signal:
    exp(-(t-t0)^2/(2*sigma^2))*sin(2*pi*f0*t + phi0), where t0 is the time
    of the envelope peak in sec, sigma is the envelope width in sec, f0 is
    the frequency in Hz and phi0 is the initial phase in radians.
    """
    env_vec = np.exp(-(data_x - sg_params[0])**2 / (2 * sg_params[1]**2))
    sig_vec = env_vec * np.sin(2 * np.pi * sg_params[2] * data_x
                               + sg_params[3])
    return _scale_to_snr(sig_vec, snr)


def gen_fm_sig(data_x, snr, fm_params):
    """Generate a frequency modulated (FM) sinusoid.

    S = GEN_FM_SIG(X, SNR, P)
    Generates an FM sinusoid S. X is the vector of time stamps at which the
    samples of the signal are to be computed. SNR is the matched filtering
    signal-to-noise ratio of S and P is the vector of three parameters
    [b, f0, f1] of the signal: sin(2*pi*f0*t + b*cos(2*pi*f1*t)), where b is
    the modulation index, f0 is the carrier frequency in Hz and f1 is the
    modulation frequency in Hz.
    """
    phase_vec = (2 * np.pi * fm_params[1] * data_x
                 + fm_params[0] * np.cos(2 * np.pi * fm_params[2] * data_x))
    sig_vec = np.sin(phase_vec)
    return _scale_to_snr(sig_vec, snr)


def gen_am_sig(data_x, snr, am_params):
    """Generate an amplitude modulated (AM) sinusoid.

    S = GEN_AM_SIG(X, SNR, P)
    Generates an AM sinusoid S. X is the vector of time stamps at which the
    samples of the signal are to be computed. SNR is the matched filtering
    signal-to-noise ratio of S and P is the vector of three parameters
    [f0, f1, phi0] of the signal: cos(2*pi*f1*t)*sin(2*pi*f0*t + phi0),
    where f0 is the carrier frequency in Hz, f1 is the modulation frequency
    in Hz and phi0 is the initial phase in radians.
    """
    env_vec = np.cos(2 * np.pi * am_params[1] * data_x)
    sig_vec = env_vec * np.sin(2 * np.pi * am_params[0] * data_x
                               + am_params[2])
    return _scale_to_snr(sig_vec, snr)


def gen_amfm_sig(data_x, snr, amfm_params):
    """Generate an AM-FM sinusoid.

    S = GEN_AMFM_SIG(X, SNR, P)
    Generates an AM-FM sinusoid S. X is the vector of time stamps at which
    the samples of the signal are to be computed. SNR is the matched
    filtering signal-to-noise ratio of S and P is the vector of three
    parameters [b, f0, f1] of the signal:
    cos(2*pi*f1*t)*sin(2*pi*f0*t + b*cos(2*pi*f1*t)), where b is the
    modulation index, f0 is the carrier frequency in Hz and f1 is the
    modulation frequency in Hz.
    """
    mod_vec = np.cos(2 * np.pi * amfm_params[2] * data_x)
    phase_vec = 2 * np.pi * amfm_params[1] * data_x + amfm_params[0] * mod_vec
    sig_vec = mod_vec * np.sin(phase_vec)
    return _scale_to_snr(sig_vec, snr)


def gen_ltc_sig(data_x, snr, ltc_params):
    """Generate a linear transient chirp signal.

    S = GEN_LTC_SIG(X, SNR, P)
    Generates a linear transient chirp signal S. X is the vector of time
    stamps at which the samples of the signal are to be computed. SNR is the
    matched filtering signal-to-noise ratio of S and P is the vector of five
    parameters [ta, f0, f1, phi0, L] of the signal:
    sin(2*pi*(f0*(t-ta) + f1*(t-ta)^2) + phi0) for ta <= t <= ta+L and 0
    otherwise, where ta is the start time in sec, f0 is the starting
    frequency in Hz, f1 is the chirp coefficient in Hz/sec, phi0 is the
    initial phase in radians and L is the length of the chirp in sec.
    """
    start_time, start_freq, freq_coef, init_phase, chirp_len = ltc_params
    sig_vec = np.zeros_like(data_x, dtype=float)
    in_chirp = (data_x >= start_time) & (data_x <= start_time + chirp_len)
    shifted_time = data_x[in_chirp] - start_time           # t - ta
    phase_vec = (2 * np.pi * (start_freq * shifted_time
                              + freq_coef * shifted_time**2)
                 + init_phase)
    sig_vec[in_chirp] = np.sin(phase_vec)
    return _scale_to_snr(sig_vec, snr)


def periodogram(sig_vec, samp_freq):
    """Compute the periodogram (magnitude of the FFT) of a signal.

    F, P = PERIODOGRAM(S, FS)
    Returns the positive Fourier frequencies F in Hz and the magnitude of
    the FFT P of the signal S sampled at FS Hz. The positive frequencies are
    FFT indices 0 to floor(N/2) (1 to floor(N/2)+1 in Matlab) and the
    frequency spacing is 1/(N*Delta) = FS/N.
    """
    n_samples = len(sig_vec)
    k_nyq = n_samples // 2 + 1                    # Number of positive freqs
    pos_freq = np.arange(k_nyq) * samp_freq / n_samples
    fft_sig = np.fft.fft(sig_vec)[:k_nyq]
    return pos_freq, np.abs(fft_sig)


# Signals used in lab 2: name, generator, parameters and the highest
# instantaneous frequency (Hz), all for a signal of duration SIG_DURATION.
SIG_DURATION = 1.0      # Length of each signal in seconds

SIGNAL_LIST = [
    # Quadratic chirp: f(t) = a1 + 2*a2*t + 3*a3*t^2, largest at t = 1
    dict(name='Quadratic chirp', func=gen_qc_sig,
         params=[10, 3, 3], max_freq=10 + 2*3*1 + 3*3*1**2),
    # Sinusoid: constant frequency f0
    dict(name='Sinusoid', func=gen_sin_sig,
         params=[10, np.pi/4], max_freq=10),
    # Linear chirp: f(t) = f0 + f1*t, largest at t = 1
    dict(name='Linear chirp', func=gen_lc_sig,
         params=[5, 20, 0], max_freq=5 + 20*1),
    # Sine-Gaussian: frequency f0 inside the envelope
    dict(name='Sine-Gaussian', func=gen_sg_sig,
         params=[0.5, 0.1, 10, 0], max_freq=10),
    # FM: f(t) = f0 - b*f1*sin(2*pi*f1*t), largest value f0 + b*f1
    dict(name='FM sinusoid', func=gen_fm_sig,
         params=[3, 10, 2], max_freq=10 + 3*2),
    # AM: sum of sinusoids at f0 - f1 and f0 + f1
    dict(name='AM sinusoid', func=gen_am_sig,
         params=[10, 1, 0], max_freq=10 + 1),
    # AM-FM: FM swing up to f0 + b*f1, plus f1 from the AM part
    dict(name='AM-FM sinusoid', func=gen_amfm_sig,
         params=[3, 10, 2], max_freq=10 + 3*2 + 2),
    # Transient chirp: f(t) = f0 + 2*f1*(t-ta), largest at t = ta + L
    dict(name='Linear transient chirp', func=gen_ltc_sig,
         params=[0.3, 5, 20, 0, 0.4], max_freq=5 + 2*20*0.4),
]
