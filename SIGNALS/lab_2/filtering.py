"""Separate a sum of three sinusoids with FIR filters.

Generates the sum of three sinusoids (100, 200 and 300 Hz) and designs
three FIR filters with scipy.signal.firwin (the Python version of Matlab's
fir1) such that filter #i passes only sinusoid #i: a low pass, a band pass
and a high pass filter. Each filter is applied to the signal and the
periodograms of the input and the outputs are plotted.

With a sampling frequency of 1024 Hz, the highest frequency a discrete time
sinusoid can have is the Nyquist frequency, 1024/2 = 512 Hz.

Daniel Rodriguez, September 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz

from signals import gen_sin_sig, periodogram

# Data parameters
n_samples = 2048                        # N
samp_freq = 1024                        # Sampling frequency in Hz
samp_intrvl = 1 / samp_freq             # Delta
time_vec = np.arange(n_samples) * samp_intrvl
print(f'Maximum frequency of a discrete time sinusoid = '
      f'{samp_freq / 2:g} Hz')

# Sinusoid parameters: amplitude A, frequency f0 (Hz), phase phi0 (rad)
sin_ampls = [10, 5, 2.5]
sin_freqs = [100, 200, 300]
sin_phases = [0, np.pi/6, np.pi/4]

# Generate the sum of the three sinusoids
# gen_sin_sig sets the amplitude through the SNR (the norm of the signal).
# A sinusoid with a whole number of cycles in the data has
# norm = A*sqrt(N/2), so SNR = A*sqrt(N/2) gives amplitude A.
sin_sigs = []
for ampl, freq, phase in zip(sin_ampls, sin_freqs, sin_phases):
    snr = ampl * np.sqrt(n_samples / 2)
    sin_sigs.append(gen_sin_sig(time_vec, snr, [freq, phase]))
input_sig = sum(sin_sigs)

# Design the filters
# firwin takes the number of coefficients (filter order + 1); high pass and
# band pass designs need an odd number. With fs given, cutoffs are in Hz.
n_taps = 101                            # Filter order 100
low_cut = 150                           # Between signals 1 and 2 (Hz)
high_cut = 250                          # Between signals 2 and 3 (Hz)

filt_coefs = [
    firwin(n_taps, low_cut, fs=samp_freq),                      # Low pass
    firwin(n_taps, [low_cut, high_cut], pass_zero=False,
           fs=samp_freq),                                       # Band pass
    firwin(n_taps, high_cut, pass_zero=False, fs=samp_freq),    # High pass
]
filt_names = [f'Low pass (< {low_cut} Hz)',
              f'Band pass ({low_cut}-{high_cut} Hz)',
              f'High pass (> {high_cut} Hz)']

# Apply the filters (lfilter is the Python version of Matlab's filter)
output_sigs = [lfilter(coefs, 1, input_sig) for coefs in filt_coefs]

# Plot the filter frequency responses
fig, ax = plt.subplots(figsize=(9, 4))
for coefs, name in zip(filt_coefs, filt_names):
    resp_freq, resp = freqz(coefs, worN=2048, fs=samp_freq)
    ax.plot(resp_freq, np.abs(resp), label=name)
for freq in sin_freqs:
    ax.axvline(freq, color='gray', linestyle=':')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain')
ax.set_title('Filter frequency responses (dotted: sinusoid frequencies)')
ax.legend()
fig.tight_layout()

# Plot the periodograms of the input and outputs
fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
pos_freq, pgram = periodogram(input_sig, samp_freq)
axes[0].plot(pos_freq, pgram)
axes[0].set_title('Input: sum of three sinusoids')
for ax, out_sig, name in zip(axes[1:], output_sigs, filt_names):
    pos_freq, pgram = periodogram(out_sig, samp_freq)
    ax.plot(pos_freq, pgram)
    ax.set_title(f'Output of filter: {name}')
for ax in axes:
    ax.set_ylabel('|FFT|')
axes[-1].set_xlabel('Frequency (Hz)')
fig.tight_layout()

# Plot a short stretch of the input and outputs in time
# The first n_taps - 1 samples of each output are the filter start-up
# transient, so the plot starts after them.
show_idx = slice(n_taps, n_taps + 60)
fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
axes[0].plot(time_vec[show_idx], input_sig[show_idx], '.-')
axes[0].set_title('Input: sum of three sinusoids')
for ax, out_sig, name in zip(axes[1:], output_sigs, filt_names):
    ax.plot(time_vec[show_idx], out_sig[show_idx], '.-')
    ax.set_title(f'Output of filter: {name}')
axes[-1].set_xlabel('Time (sec)')
fig.tight_layout()

plt.show()
