%% Backscatter Signal Demodulation and Image Reconstruction

clear; clc; close all;

%% User Parameters

iq_binary_file    = 'rx_data.bin';
ground_truth_file = 'tx_bits.txt';
output_bits_file  = 'demodulated_bits.txt';

sample_rate       = 1e6;                                        % Sampling rate [Hz]
smoothing_window  = 10;                                         % Adjust based on noise level
peak_prominence   = 0.01;                                       % Adjust based on signal amplitude
num_clusters      = 2;                                          % OOK: 2 clusters
start_bit_offset  = 1;                                          % Bit alignment offset
image_width       = 256;                                        % Image width [pixels]
image_height      = 256;                                        % Image height [pixels]

%% IQ Data Loading

fid = fopen(iq_binary_file, 'rb');
raw_data = fread(fid, [2 inf], 'float32');
fclose(fid);

complex_signal = raw_data(1,:) + 1j * raw_data(2,:);
num_samples = length(complex_signal);
time_ms = (0:num_samples-1).' / sample_rate * 1e3;

signal_real = real(complex_signal).';
signal_imag = imag(complex_signal).';
signal_mag  = abs(complex_signal).';

%% Smoothing and Peak Detection

smoothed = smoothdata(signal_real, 'movmean', smoothing_window);

TF_peaks     = islocalmax(smoothed, 'MinProminence', peak_prominence);
peak_indices = find(TF_peaks);

peak_real = signal_real(peak_indices);
peak_imag = signal_imag(peak_indices);
peak_amp  = smoothed(peak_indices);

%% K-Means Demodulation (OOK)

[cluster_idx, centroids] = kmeans(peak_amp, num_clusters);

[~, zero_cluster] = max(centroids);
demodulated_bits  = double(cluster_idx ~= zero_cluster);

writematrix(demodulated_bits, output_bits_file);

%% BER Calculation

original_bits  = readmatrix(ground_truth_file);
recovered_bits = readmatrix(output_bits_file);

original_bits  = original_bits(start_bit_offset:end, 1);
min_length     = min(length(original_bits), length(recovered_bits));
original_bits  = original_bits(1:min_length);
recovered_bits = recovered_bits(1:min_length);

num_errors = sum(original_bits ~= recovered_bits);
BER = num_errors / min_length;

if BER > 0
    BER_dB = 10 * log10(BER);
else
    BER_dB = -inf;
end

fprintf('Total Bits: %d | Errors: %d | BER: %.6f (%.2f dB)\n', ...
    min_length, num_errors, BER, BER_dB);

%% Visualization

figure;
plot(time_ms, signal_real, 'r'); hold on;
plot(time_ms, signal_imag, 'b');
xlabel('Time (ms)'); ylabel('Amplitude');
legend('I', 'Q'); grid on;

figure;
plot(time_ms, signal_mag); hold on;
plot(time_ms, smoothed, 'g');
xlabel('Time (ms)'); ylabel('Amplitude');
legend('Original', 'Smoothed'); grid on;

figure;
plot(time_ms, smoothed); hold on;
plot(time_ms(peak_indices), smoothed(peak_indices), 'g*');
xlabel('Time (ms)'); ylabel('Amplitude');
legend('Smoothed', 'Peaks'); grid on;

figure;
plot(peak_real, peak_imag, 'o');
xlabel('I'); ylabel('Q');
axis equal; grid on;

%% Image Reconstruction

reconstruct_color_image(output_bits_file, image_width, image_height);

function reconstruct_color_image(bits_file, width, height)
    fid = fopen(bits_file, 'r');
    bit_stream = fscanf(fid, '%s');
    fclose(fid);

    num_pixels    = width * height * 3;
    required_bits = num_pixels * 8;
    total_bits    = length(bit_stream);

    if total_bits < required_bits
        noise = char(randi([0, 1], 1, required_bits - total_bits) + '0');
        bit_stream = [bit_stream, noise];
    elseif total_bits > required_bits
        bit_stream = bit_stream(1:required_bits);
    end

    byte_array = zeros(num_pixels, 1, 'uint8');
    for i = 1:num_pixels
        byte_array(i) = bin2dec(bit_stream((i-1)*8 + 1 : i*8));
    end

    image_data = permute(reshape(byte_array, [3, width, height]), [3, 2, 1]);

    figure;
    imshow(uint8(image_data));
    title('Reconstructed Image');
end
