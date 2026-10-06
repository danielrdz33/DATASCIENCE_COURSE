"""Application of the Nyquist theorem.

For each oscillatory signal in signals.SIGNAL_LIST, sample the signal at
the Nyquist rate (2 x highest instantaneous frequency), twice the Nyquist
rate and 10 x the Nyquist rate, and plot it in the time domain and the
Fourier domain (periodogram).

Daniel Rodriguez, September 2026
"""

import numpy as np
import matplotlib.pyplot as plt

from signals import SIGNAL_LIST, SIG_DURATION, periodogram

snr = 10                                # Matched filtering SNR (sets A)
nyq_multiples = [1, 2, 10]              # Multiples of the Nyquist rate

for sig_info in SIGNAL_LIST:
    max_freq = sig_info['max_freq']
    nyq_rate = 2 * max_freq             # Nyquist rate for this signal

    fig, axes = plt.subplots(len(nyq_multiples), 2, figsize=(11, 8))
    fig.suptitle(f"{sig_info['name']} (highest frequency = "
                 f"{max_freq:g} Hz)")

    for row, nyq_mult in enumerate(nyq_multiples):
        samp_freq = nyq_mult * nyq_rate             # 1/Delta
        samp_intrvl = 1 / samp_freq                 # Delta
        n_samples = int(np.floor(SIG_DURATION / samp_intrvl))   # N
        time_vec = np.arange(n_samples) * samp_intrvl   # t = n*Delta

        sig_vec = sig_info['func'](time_vec, snr, sig_info['params'])
        pos_freq, pgram = periodogram(sig_vec, samp_freq)

        # Time domain
        ax_time = axes[row, 0]
        ax_time.plot(time_vec, sig_vec, '.-')
        ax_time.set_xlabel('Time (sec)')
        ax_time.set_title(f'{nyq_mult} x Nyquist rate: '
                          f'fs = {samp_freq:g} Hz')

        # Fourier domain; zoom to 2 x max_freq
        ax_freq = axes[row, 1]
        ax_freq.plot(pos_freq, pgram, '.-')
        ax_freq.set_xlim(0, min(samp_freq / 2, 2 * max_freq))
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.set_ylabel('|FFT|')
        ax_freq.set_title(f'Periodogram, {nyq_mult} x Nyquist rate')

    fig.tight_layout()

plt.show()
