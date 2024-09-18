import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
num_elements = 4  # Number of antenna elements
f = 1e9           # Frequency of signal (1 GHz)
c = 3e8           # Speed of light (m/s)
wavelength = c / f  # Wavelength
d = wavelength / 2  # Spacing between elements (half wavelength)
fs = 100*f         # Sampling frequency (100 times the frequency)
Ts = 1/fs           # Sampling period
t_total = 4*(1/f)  # Total time for simulation (4 signal cycles)

# Arrival angle of the signal (in degrees)
theta = 10  # Signal arrival angle

# Time vector for signal (assume 4 microsecond of data)
t = np.linspace(start=0, stop=t_total, num=int(t_total/Ts))

# Original signal (sinusoidal)
signal = np.sin(2 * np.pi * f * t)

# noise signal
# AWGN parameters
SNR_dB = 20  # Signal-to-noise ratio in dB
SNR = 10**(SNR_dB / 10)  # Convert SNR from dB to linear scale

# Compute noise variance
signal_power = np.mean(signal**2)  # Signal power
noise_power = signal_power / SNR  # Noise power

# Generate AWGN
#noise = np.sqrt(noise_power) * np.random.randn(len(t))
noise_signals = np.zeros((num_elements, len(t)), dtype=complex)
for i in range(num_elements):
    noise_signals[i, :] = np.sqrt(noise_power) * np.random.randn(len(t))


# Phase shifts for each antenna element based on arrival angle
# Steering vector calculation
#steering_vector = (1/np.sqrt(num_elements))*np.exp(1j * 2 * np.pi * d * np.arange(num_elements) * np.sin(np.radians(theta)) / wavelength)
steering_vector = np.zeros(num_elements, dtype=complex)
for i in range(num_elements):
    steering_vector[i] = (1/np.sqrt(num_elements))*np.exp(-1*1j*(i-1)*2*np.pi*d*np.sin(np.radians(theta))/wavelength)

# Create received signals for each element by applying phase shifts
received_signals = np.zeros((num_elements, len(t)), dtype=complex)
plt.figure(figsize=(10, 6))
for i in range(num_elements):
    #received_signals[i, :] = signal * np.exp(1j * 2 * np.pi * d * i * np.sin(np.radians(theta)) / wavelength)
    received_signals[i, :] = signal * steering_vector[i] + noise_signals[i, :]
    plt.plot(t, received_signals[i, :], label="Element "+str(i+1)+" rx signal")

plt.xlabel('Time (microseconds)')
plt.ylabel('Amplitude')
plt.title('Rx Signals for Each Antenna Element')
plt.legend()
plt.grid(True)

# Beamforming: sum signals after phase shifts
beamformed_signal = np.sum(received_signals, axis=0)

# Plotting the original and beamformed signals
plt.figure(figsize=(10, 6))
plt.plot(t * 1e6, signal, label='Original Signal')
plt.plot(t * 1e6, np.real(beamformed_signal), label='Beamformed Signal', linestyle='--')
plt.xlabel('Time (microseconds)')
plt.ylabel('Amplitude')
plt.title('Original vs Beamformed Signal')
plt.legend()
plt.grid(True)
plt.show()