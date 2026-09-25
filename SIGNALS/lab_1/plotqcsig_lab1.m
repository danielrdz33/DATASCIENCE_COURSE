% Plot a quadratic chirp signal for decreasing sampling intervals
% Generates a quadratic chirp signal with CRCBGENQCSIG and plots it for
% several sampling intervals, from coarse to fine. Reduce the sampling
% interval until the plot stops changing significantly.
% Requires crcbgenqcsig.m to be in the same folder or on the Matlab path.

% Daniel Rodriguez, September 2026

%% Signal parameters
snr = 10;               % Matched filtering SNR (sets the amplitude A)
qcCoefs = [10, 3, 3];   % Phase coefficients [a1, a2, a3]
sigDuration = 1;        % Length of the signal in seconds

%% Maximum instantaneous frequency
% f(t) = a1 + 2*a2*t + 3*a3*t^2 is largest at the end of the signal
maxFreq = qcCoefs(1) + 2*qcCoefs(2)*sigDuration + 3*qcCoefs(3)*sigDuration^2;
disp(['Maximum instantaneous frequency = ', num2str(maxFreq), ' Hz']);

%% Sampling frequencies to try, as multiples of maxFreq
sampFreqList = [2, 5, 10, 50]*maxFreq;
nPlots = length(sampFreqList);

%% Generate and plot the signal for each sampling frequency
figure;
for k = 1:nPlots
    sampFreq = sampFreqList(k);                 % 1/Delta
    sampIntrvl = 1/sampFreq;                    % Delta
    nSamples = floor(sigDuration/sampIntrvl);   % N
    timeVec = (0:(nSamples-1))*sampIntrvl;      % t = n*Delta, n = 0,...,N-1

    sigVec = crcbgenqcsig(timeVec, snr, qcCoefs);

    subplot(nPlots, 1, k);
    plot(timeVec, sigVec, '.-');
    xlabel('Time (sec)');
    ylabel('Quad. Chirp');
    title(['Sampling frequency = ', num2str(sampFreq), ' Hz (', ...
           num2str(sampFreq/maxFreq), ' x max. frequency)']);
end
