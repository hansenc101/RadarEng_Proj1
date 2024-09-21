import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
num_elements = 4  # Number of antenna elements
f = 9.55*1e9           # Frequency of signal (1 GHz)
c = 3e8           # Speed of light (m/s)
wavelength = c / f  # Wavelength
d = wavelength/2  # Spacing between elements (half wavelength)
fs = 100*f         # Sampling frequency (100 times the frequency)
Ts = 1/fs           # Sampling period
t_total = 4*(1/f)  # Total time for simulation (4 signal cycles)

# Arrival angle of the signal (in degrees)
theta = 20  # Signal arrival angle

# Time vector for signal (assume 4 microsecond of data)
t = np.linspace(start=0, stop=t_total, num=int(t_total/Ts))
print("N samples: "+str(np.shape(t)))

# Original signal (sinusoidal)
signal = np.sin(2 * np.pi * f * t)

# noise signal
# AWGN parameters
SNR_dB = 7  # Signal-to-noise ratio in dB
SNR = 10**(SNR_dB / 10)  # Convert SNR from dB to linear scale

# Compute noise variance
signal_power = np.mean(signal**2)  # Signal power
noise_power = signal_power/SNR #signal_power / SNR  # Noise power

# Generate AWGN
noise_signals = np.zeros((num_elements, len(t)), dtype=complex)
for i in range(num_elements):
    noise_signals[i, :] = np.sqrt(noise_power) * np.random.randn(len(t))


# Phase shifts for each antenna element based on arrival angle
delta_theta=np.zeros(num_elements)

# Phase center of the array (in wavelengths)
phase_center = (num_elements-1) * d / 2  # For 4 elements, this will be 3 * (lambda/2) / 2 = 3 * lambda / 4
# Steering vector calculation
steering_vector = np.zeros(num_elements, dtype=complex)

for i in range(num_elements):
    element_position = i * d  # Position of the i-th element
    relative_position = element_position - phase_center  # Position relative to the phase center
    print("Relative position: ", relative_position)
    steering_vector[i] = (1/np.sqrt(num_elements)) * np.exp(-1j * 2 * np.pi * relative_position * np.sin(np.radians(theta)) / wavelength)
    #steering_vector[i] = np.exp(-1j * 2 * np.pi * relative_position * np.sin(np.radians(theta)) / wavelength)

    delta_theta[i] = 2*np.pi*np.sin(np.radians(theta))*d*i/wavelength


# Calculate the center point for delta_theta
delta_theta = delta_theta - np.mean(delta_theta)
print("delta_theta (degrees):", np.degrees(delta_theta))

print("Steering Vector Values:", steering_vector)
steering_phases = np.angle(steering_vector)  # Phase in radians
print("Steering phases (degrees):", np.degrees(steering_phases))


print("Steering vector N samples: "+str(np.size(steering_vector)))

# Create received signals for each element by applying phase shifts

raw_rx = np.zeros((num_elements, len(t)), dtype=complex)
w_rx = np.zeros((num_elements, len(t)), dtype=complex)

# Plot I (in-phase) and Q (quadrature) components for each antenna element
plt.figure(figsize=(12, 8))

for i in range(num_elements):
    attenuation = 1
    # The initial rx signal at each element (same signal w/varying time/phase delay)
    raw_rx[i,:] = attenuation*np.sin(2*np.pi*f*t + delta_theta[i]) + noise_signals[i, :]

    # Weighted ("steered") rx signals
    w_rx[i, :] = raw_rx[i,:]*steering_vector[i]
    #w_rx[i, :] = raw_rx[i, :] * np.exp(1j * steering_phases[i])


    # Plot the in-phase component (real part)
    plt.subplot(num_elements, 2, 2*i+1)
    plt.plot(t * 1e6, np.real(raw_rx[i, :]), label=f'Element {i+1} - In-Phase (I)')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Element {i+1} - In-Phase (I) - Raw Signal')
    plt.grid(True)

    # Plot the quadrature component (imaginary part)
    plt.subplot(num_elements, 2, 2*i+2)
    plt.plot(t * 1e6, np.imag(raw_rx[i, :]), label=f'Element {i+1} - Quadrature (Q)')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Element {i+1} - Quadrature (Q) - Raw Signal')
    plt.grid(True)
# Adjust layout for better viewing
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

for i in range(num_elements):
  # Plot the in-phase component (real part)
    plt.subplot(num_elements, 2, 2*i+1)
    plt.plot(t * 1e6, np.real(w_rx[i, :]), label=f'Element {i+1} - In-Phase (I)')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Element {i+1} - In-Phase (I) - Rx*W')
    plt.grid(True)

    # Plot the quadrature component (imaginary part)
    plt.subplot(num_elements, 2, 2*i+2)
    plt.plot(t * 1e6, np.imag(w_rx[i, :]), label=f'Element {i+1} - Quadrature (Q)')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Element {i+1} - Quadrature (Q) Rx*W')
    plt.grid(True)

# Adjust layout for better viewing
plt.tight_layout()
plt.show()

# Beamforming: sum signals after phase shifts
beamformed_signal = np.sum(w_rx, axis=0)

# Plotting the original and beamformed signals
plt.figure(figsize=(10, 6))

# Plot the original signal
plt.plot(t * 1e6, signal, label='Original Signal')

# Plot the real part of the beamformed signal (in-phase component)
plt.plot(t * 1e6, np.real(beamformed_signal), label='Beamformed Signal (In-Phase)')

# Plot the imaginary part of the beamformed signal (quadrature component)
#plt.plot(t * 1e6, np.imag(beamformed_signal), label='Beamformed Signal (Quadrature)')

# Labels and legend
plt.xlabel('Time (microseconds)')
plt.ylabel('Amplitude')
plt.title('Real vs Imaginary Parts of Beamformed Signal')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
