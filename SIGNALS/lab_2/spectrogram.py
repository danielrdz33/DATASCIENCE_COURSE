"""Time-frequency analysis of the signals with spectrograms.

For each signal in signals.SIGNAL_LIST, sample the signal at 10 x its
Nyquist rate and plot its time series and its spectrogram, made with
scipy.signal.spectrogram.

Daniel Rodriguez, September 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from signals import SIGNAL_LIST, SIG_DURATION

snr = 10                                # Matched filtering SNR (sets A)
nyq_mult = 10                           # Sample at 10 x Nyquist rate

# Spectrogram settings
# A longer window gives finer frequency resolution (about 1/win_len Hz)
# but coarser time resolution
win_len = 0.2                           # Window length in sec
ovrlp = 0.18                            # Overlap between windows in sec

for sig_info in SIGNAL_LIST:
    max_freq = sig_info['max_freq']
    samp_freq = nyq_mult * 2 * max_freq         # 1/Delta
    samp_intrvl = 1 / samp_freq                 # Delta
    n_samples = int(np.floor(SIG_DURATION / samp_intrvl))   # N
    time_vec = np.arange(n_samples) * samp_intrvl
    sig_vec = sig_info['func'](time_vec, snr, sig_info['params'])

    # Convert window length and overlap to integer numbers of samples
    win_len_smpls = int(np.floor(win_len * samp_freq))
    ovrlp_smpls = int(np.floor(ovrlp * samp_freq))

    # Hamming window as in Matlab's default; nfft > window zero-pads each
    # segment for a smoother frequency axis (no extra resolution)
    freq_vec, seg_times, spec_mag = spectrogram(
        sig_vec, fs=samp_freq, window='hamming',
        nperseg=win_len_smpls, noverlap=ovrlp_smpls,
        nfft=8 * win_len_smpls, mode='magnitude')

    fig, (ax_time, ax_spec) = plt.subplots(2, 1, figsize=(9, 7),
                                           sharex=True)
    ax_time.plot(time_vec, sig_vec)
    ax_time.set_title(f"{sig_info['name']}: time series "
                      f"(fs = {samp_freq:g} Hz)")

    mesh = ax_spec.pcolormesh(seg_times, freq_vec, spec_mag,
                              shading='gouraud')
    ax_spec.set_ylim(0, 2 * max_freq)
    ax_spec.set_xlim(0, SIG_DURATION)
    ax_spec.set_xlabel('Time (sec)')
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title(f'Spectrogram (window = {win_len:g} s)')
    fig.colorbar(mesh, ax=[ax_time, ax_spec], label='|S|')

plt.show()
