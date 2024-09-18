%% Static Beam
% Parameters
N = 8;                 % Number of array elements
d = 0.5;               % Element spacing (in wavelengths)
theta_steering = 30;   % Steering angle (in degrees)
lambda = 1;            % Wavelength
k = 2*pi/lambda;       % Wavenumber

% Define the angle range
theta = -90:0.1:90;    % Angles to plot (degrees)
theta_rad = deg2rad(theta);  % Convert to radians

% Phase shift for beam steering
steering_vector = exp(1j * k * d * (0:N-1)' * sind(theta_steering));

% Array factor calculation
array_factor = zeros(size(theta));
for n = 1:N
    % Phase shift for each element
    phase_shift = exp(1j * (n-1) * k * d * sind(theta));
    array_factor = array_factor + steering_vector(n) * phase_shift;
end

% Normalization
array_factor = abs(array_factor) / max(abs(array_factor));

% Plot the beam pattern
figure;
polarplot(deg2rad(theta), array_factor, 'LineWidth', 2);
title('Beamforming Pattern');
rlim([0 1]);
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';
ax.RColor = 'r';
ax.ThetaColor = 'b';

% Optional plot in cartesian coordinates
figure;
plot(theta, 20*log10(array_factor), 'LineWidth', 2);
xlabel('Angle (degrees)');
ylabel('Array Factor (dB)');
title('Beamforming Pattern (in dB)');
grid on;

%% Dynamic Beam
clear; clc;
% Parameters
N = 4;                 % Number of array elements
d = 0.5;               % Element spacing (in wavelengths)
lambda = 1;            % Wavelength
k = 2*pi/lambda;       % Wavenumber

% Define the angle range (top half of the circle: 0 to 180 degrees)
theta = -90:0.1:90;     % Angles to plot (degrees)
theta_rad = deg2rad(theta);  % Convert to radians

% Prepare the figure for animation
figure;
figure('Position', [100, 100, 1200, 900]);  % Larger figure: 800x600 pixels
h = polarplot(theta_rad, zeros(size(theta)), 'LineWidth', 1);  % Beam pattern plot
hold on;
h_main_lobe = polarplot([0 0], [0 1], 'r', 'LineWidth', 2);    % Red line for main lobe
hold off;
rlim([0 1]);
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';
ax.RColor = 'r';
ax.ThetaColor = 'b';
title('Beamforming Pattern with Steering Angle Indicator');

% Animation loop (sweeping steering angle from -30 to 30 degrees)
for theta_steering = -40:0.5:40
    % Phase shift for beam steering
    steering_vector = exp(1j * k * d * (0:N-1)' * sind(theta_steering));
    
    % Array factor calculation
    array_factor = zeros(size(theta));
    for n = 1:N
        % Phase shift for each element
        phase_shift = exp(1j * (n-1) * k * d * sind(theta));
        array_factor = array_factor + steering_vector(n) * phase_shift;
    end

    % Normalization
    array_factor = abs(array_factor) / max(abs(array_factor));

    % Update the polar plot data
    set(h, 'YData', array_factor);
    
    % Update the main lobe direction (red line)
    theta_lobe = deg2rad(-1*theta_steering);  % Convert steering angle to radians
    set(h_main_lobe, 'ThetaData', [theta_lobe theta_lobe], 'RData', [0 1]);  % Update red line

    % Pause to create animation effect
    pause(0.05);  % Adjust the pause time to control speed of the animation
end
