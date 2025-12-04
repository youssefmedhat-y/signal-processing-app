import numpy as np


class Signal:
    def __init__(self):
        self.N1 = 0
        self.samples = {}
        self.is_periodic = False  # Flag to indicate if signal is periodic


def write_signal_file(signal_object, file_name, is_periodic=False):
    """
    Write signal to file
    For time domain signals: writes index and amplitude
    For frequency domain signals: writes amplitude and phase
    """
    with open(file_name, 'w') as f:
        # Determine if this is a frequency domain signal
        is_frequency_domain = hasattr(signal_object, 'frequencies')
        
        # 1. Write SignalType
        f.write(f"{1 if is_frequency_domain else 0}\n")  # 1 for Frequency, 0 for Time
        
        # 2. Write IsPeriodic
        f.write(f"{1 if is_periodic else 0}\n")  # 1 for periodic, 0 for non-periodic
        
        # 3. Write Number of Samples
        f.write(f"{len(signal_object.samples)}\n")

        # 4. Write Samples
        if is_frequency_domain:
            # For frequency domain: write amplitude and phase (without index)
            for key in sorted(signal_object.samples.keys()):
                amplitude = signal_object.amplitudes[key]
                phase = signal_object.phase_shifts[key]
                f.write(f"{amplitude} {phase}\n")
        else:
            # For time domain: write index and amplitude
            for key in sorted(signal_object.samples.keys()):
                data = signal_object.samples[key]
                f.write(f"{key} {data[0]}\n")

    print(f"Save successful to {file_name}")
    return True


def load_signal_file(file_name):
    """
    Reads a signal from a file with the format:
    [SignalType] // Time-->0/Freq-->1
    [IsPeriodic] // takes 0 or 1
    [N1] // number of samples to follow
    
    For time domain (SignalType=0):
        Index SampleAmp
        i0 amp0
        i1 amp1
        ...
    
    For frequency domain (SignalType=1):
        Amplitude Phase
        amp0 phase0
        amp1 phase1
        ...
    """
    signal_object = Signal()
    signal_object.samples = {}  # Initialize samples dictionary

    try:
        with open(file_name, 'r') as f:
            # 1. Read Signal Type (0=Time, 1=Frequency)
            try:
                signal_type_line = f.readline().strip()
                signal_type = int(signal_type_line)  # 0 for Time, 1 for Frequency
                is_frequency_domain = (signal_type == 1)
            except ValueError:
                print(f"Error reading signal type in file: {file_name}")
                return None
                
            # 2. Read IsPeriodic flag (0=Non-periodic, 1=Periodic)
            try:
                is_periodic_line = f.readline().strip()
                is_periodic = int(is_periodic_line) == 1
                signal_object.is_periodic = is_periodic
            except ValueError:
                print(f"Error reading periodic flag in file: {file_name}, assuming non-periodic")
                signal_object.is_periodic = False

            # 3. Read Number of Samples (N1)
            try:
                num_samples_line = f.readline().strip()
                signal_object.N1 = int(num_samples_line)
            except ValueError:
                print(f"Error reading number of samples in file: {file_name}")
                return None

            # 4. Read Samples
            if is_frequency_domain:
                # For frequency domain: read amplitude and phase pairs
                amplitudes = []
                phase_shifts = []
                index = 0
                
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            amplitude = float(parts[0])
                            phase = float(parts[1])
                            amplitudes.append(amplitude)
                            phase_shifts.append(phase)
                            # Store in samples dict as [amplitude, phase]
                            signal_object.samples[index] = [amplitude, phase]
                            index += 1
                        except ValueError:
                            print(f"Skipping malformed line in {file_name}: {line.strip()}")
                            continue
                    elif parts:
                        print(f"Skipping unexpected line format in {file_name}: {line.strip()}")
                        continue
                
                # Store frequency domain metadata
                signal_object.amplitudes = np.array(amplitudes)
                signal_object.phase_shifts = np.array(phase_shifts)
                
                # Calculate normalized amplitudes
                max_amp = np.max(signal_object.amplitudes) if len(amplitudes) > 0 else 1.0
                if max_amp > 0:
                    signal_object.normalized_amplitudes = signal_object.amplitudes / max_amp
                else:
                    signal_object.normalized_amplitudes = signal_object.amplitudes.copy()
                
                # We don't know sampling frequency from file, set to 1.0 as default
                signal_object.sampling_frequency = 1.0
                signal_object.frequencies = np.array([k * signal_object.sampling_frequency / len(amplitudes) 
                                                     for k in range(len(amplitudes))])
                
            else:
                # For time domain: read index and amplitude pairs
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            index = int(parts[0])
                            amplitude = float(parts[1])
                            # The samples dict stores {index: [amplitude]}
                            signal_object.samples[index] = [amplitude]
                        except ValueError:
                            print(f"Skipping malformed line in {file_name}: {line.strip()}")
                            continue
                    elif parts:  # Log non-empty lines that don't split into 2 parts
                        print(f"Skipping unexpected line format in {file_name}: {line.strip()}")
                        continue

            # Recalculate N1 based on actual loaded samples in case file N1 was incorrect
            if signal_object.N1 != len(signal_object.samples):
                print(
                    f"Warning: File header N1 ({signal_object.N1}) does not match actual samples loaded ({len(signal_object.samples)}). Using loaded count.")
                signal_object.N1 = len(signal_object.samples)

        print(f"Load successful from {file_name}")
        if is_frequency_domain:
            print(f"  Loaded as frequency domain signal with {signal_object.N1} components")
        return signal_object

    except FileNotFoundError:
        print(f"Error: File not found at {file_name}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_name}: {e}")
        return None


def _perform_two_signal_op(signal1, signal2, operation):
    result_signal = Signal()

    all_keys = set(signal1.samples.keys()) | set(signal2.samples.keys())

    for key in all_keys:
        amp1 = signal1.samples.get(key, [0.0])[0]
        amp2 = signal2.samples.get(key, [0.0])[0]

        if operation == 'add':
            result_amp = amp1 + amp2
        elif operation == 'subtract':
            result_amp = amp1 - amp2

        result_signal.samples[key] = [result_amp]

    result_signal.N1 = len(result_signal.samples)
    return result_signal


def add_signals(signal1, signal2):
    return _perform_two_signal_op(signal1, signal2, 'add')


def subtract_signals(signal1, signal2):
    return _perform_two_signal_op(signal1, signal2, 'subtract')


def multiply_signal(signal, constant):
    result_signal = Signal()

    for key, data in signal.samples.items():
        original_amp = data[0]
        result_amp = original_amp * constant
        result_signal.samples[key] = [result_amp]

    result_signal.N1 = len(result_signal.samples)
    return result_signal


def square_signal(signal):
    result_signal = Signal()

    for key, data in signal.samples.items():
        original_amp = data[0]
        result_amp = original_amp ** 2
        result_signal.samples[key] = [result_amp]

    result_signal.N1 = len(result_signal.samples)
    return result_signal


def accumulate_signal(signal):
    result_signal = Signal()

    sorted_indices = sorted(signal.samples.keys())
    running_sum = 0.0

    for index in sorted_indices:
        current_amp = signal.samples[index][0]
        running_sum += current_amp
        result_signal.samples[index] = [running_sum]

    result_signal.N1 = len(result_signal.samples)
    return result_signal


def normalize_signal(signal, target_range):
    samples = np.array([data[0] for data in signal.samples.values()])
    if len(samples) == 0:
        return signal

    min_val = np.min(samples)
    max_val = np.max(samples)

    range_val = max_val - min_val
    # Avoid division by zero for constant signals
    if range_val == 0:
        if target_range == '-1_to_1':
            normalized_samples = np.ones_like(samples) * (1 if min_val >= 0 else -1)
        elif target_range == '0_to_1':
            normalized_samples = np.ones_like(samples) * (1 if min_val >= 0 else 0)
    else:
        scaled_samples = (samples - min_val) / range_val

        if target_range == '-1_to_1':
            normalized_samples = 2 * scaled_samples - 1
        elif target_range == '0_to_1':
            normalized_samples = scaled_samples

    result_signal = Signal()
    result_signal.N1 = signal.N1

    sorted_keys = sorted(signal.samples.keys())
    # Ensure correct mapping back to indices
    # We must use the sorted order of keys to match the sorted order of 'samples' used to create 'normalized_samples'
    for i, key in enumerate(sorted_keys):
        result_signal.samples[key] = [normalized_samples[i]]

    return result_signal


def compare_signal(signal1, signal2):
    # This implementation is likely a placeholder from the original code.
    # A proper comparison would check all samples.
    if (signal1 == signal2):
        print(1);
    else:
        print(2);


def generate_sin_cos(w_type, A, theta, F, Fs, duration=1.0, is_periodic=True):
    if Fs < 2 * F:
        print(f"WARNING: Nyquist criterion violated (Fs < 2F). Aliasing will occur. Fs={Fs}, 2F={2 * F}")
        return (0)

    N = int(Fs * duration)
    n_indices = np.arange(N)

    # y[n] = A * sin/cos(2*pi * (F/Fs) * n + theta)
    discrete_frequency = (2 * np.pi * F) / Fs

    if w_type == 'sine':
        samples_array = A * np.sin(discrete_frequency * n_indices + theta)
    elif w_type == 'cosine':
        samples_array = A * np.cos(discrete_frequency * n_indices + theta)

    result_signal = Signal()
    result_signal.N1 = N
    result_signal.samples = {n: [samples_array[n]] for n in n_indices}
    result_signal.is_periodic = is_periodic  # Set the periodic flag based on the parameter

    return result_signal


def quantize_signal_bits(signal, num_bits):
    """
    Quantize a signal based on number of bits
    
    Args:
        signal: Signal object
        num_bits: Number of bits for quantization
    
    Returns:
        tuple: (encoded_values, quantized_values)
    """
    # Calculate number of levels from bits
    num_levels = 2 ** num_bits
    
    # Extract signal values and find min/max
    samples = np.array([data[0] for data in signal.samples.values()])
    min_val = np.min(samples)
    max_val = np.max(samples)
    
    # Calculate step size (Delta)
    delta = (max_val - min_val) / num_levels
    
    # Initialize result arrays
    encoded_values = []
    quantized_values = []
    
    # Sort indices to ensure proper ordering
    sorted_indices = sorted(signal.samples.keys())
    
    for index in sorted_indices:
        sample_val = signal.samples[index][0]
        
        # Calculate the level for this sample
        level = int((sample_val - min_val) / delta)
        
        # Handle edge case for max value
        if level == num_levels:
            level = num_levels - 1
            
        # Calculate quantized value (midpoint of the level)
        quantized_val = min_val + delta * (level + 0.5)
        
        # Generate binary code for the level
        # Convert to binary and remove '0b' prefix, then pad with zeros
        binary_code = bin(level)[2:].zfill(num_bits)
        
        # Store results
        encoded_values.append(binary_code)
        quantized_values.append(quantized_val)
    
    return encoded_values, quantized_values


def quantize_signal_levels(signal, num_levels):
    """
    Quantize a signal based on number of levels
    
    Args:
        signal: Signal object
        num_levels: Number of levels for quantization
    
    Returns:
        tuple: (interval_indices, encoded_values, quantized_values, sampled_errors)
    """
    # Calculate number of bits from levels
    num_bits = int(np.ceil(np.log2(num_levels)))
    
    # Extract signal values and find min/max
    samples = np.array([data[0] for data in signal.samples.values()])
    min_val = np.min(samples)
    max_val = np.max(samples)
    
    # Calculate step size (Delta)
    delta = (max_val - min_val) / num_levels
    
    # Initialize result arrays
    interval_indices = []
    encoded_values = []
    quantized_values = []
    sampled_errors = []
    
    # Sort indices to ensure proper ordering
    sorted_indices = sorted(signal.samples.keys())
    
    for index in sorted_indices:
        sample_val = signal.samples[index][0]
        
        # Calculate the interval index for this sample
        interval_idx = int((sample_val - min_val) / delta)
        
        # Handle edge case for max value
        if interval_idx == num_levels:
            interval_idx = num_levels - 1
            
        # Calculate quantized value (midpoint of the interval)
        quantized_val = min_val + delta * (interval_idx + 0.5)
        
        # Calculate the error
        error = quantized_val - sample_val
        
        # Generate binary code for the interval index
        # Convert to binary and remove '0b' prefix, then pad with zeros
        binary_code = bin(interval_idx)[2:].zfill(num_bits)
        
        # Store results
        interval_indices.append(interval_idx+1)
        encoded_values.append(binary_code)
        quantized_values.append(quantized_val)
        sampled_errors.append(error)
    
    return interval_indices, encoded_values, quantized_values, sampled_errors


def _fft_recursive(x, is_inverse=False):
    """
    Recursive FFT implementation using Decimation-in-Time (Cooley-Tukey algorithm)
    Follows the butterfly operation structure
    Handles both FFT and IFFT with conditional logic
    
    Args:
        x: Input array (complex numbers)
        is_inverse: If True, compute IFFT; if False, compute FFT
    
    Returns:
        Complex array of FFT/IFFT coefficients
    """
    N = len(x)
    
    # Base case: if N = 1, return the input
    if N == 1:
        return x
    
    # Base case: if N = 2, direct computation
    if N == 2:
        # Butterfly operation for N=2
        # return List {x[0]+x[1], x[0]-x[1]}
        return np.array([x[0] + x[1], x[0] - x[1]])
    
    # Check if N is a power of 2
    if N & (N - 1) != 0:
        # Pad with zeros to next power of 2
        next_pow2 = 2 ** int(np.ceil(np.log2(N)))
        x = np.pad(x, (0, next_pow2 - N), 'constant')
        N = next_pow2
    
    # Divide: L1 = samples with even indices, L2 = samples with odd indices
    L1 = x[0::2]  # Even indices
    L2 = x[1::2]  # Odd indices
    
    # Recursive calls
    fft_1 = _fft_recursive(L1, is_inverse)  # FFT of even samples
    fft_2 = _fft_recursive(L2, is_inverse)  # FFT of odd samples
    
    # Initialize result array
    result = np.zeros(N, dtype=complex)
    
    # Twiddle factor sign: -1 for FFT, +1 for IFFT
    sign = 1 if is_inverse else -1
    
    # Loop from k = 0 to N/2 - 1
    for k in range(N // 2):
        # W = exp(−j2πk/N) for FFT, exp(+j2πk/N) for IFFT
        W = np.exp(sign * 2j * np.pi * k / N)
        
        # Butterfly operations
        # butterflyTop: fft_1[k] + W * fft_2[k]
        # butterflyDown: fft_1[k] - W * fft_2[k]
        result[k] = fft_1[k] + W * fft_2[k]           # Top butterfly
        result[k + N // 2] = fft_1[k] - W * fft_2[k]  # Bottom butterfly
    
    return result


def fft_ifft(x, sampling_frequency=1.0, is_inverse=False):
    """
    Unified FFT/IFFT function using Decimation-in-Time algorithm
    
    Args:
        x: Input signal (numpy array or Signal object for FFT, 
           or tuple of (amplitudes, phase_shifts) for IFFT)
        sampling_frequency: Sampling frequency in Hz (for FFT)
        is_inverse: If True, compute IFFT; if False, compute FFT
    
    Returns:
        For FFT: tuple (frequencies, amplitudes, phase_shifts)
        For IFFT: Signal object in time domain
    """
    if not is_inverse:
        # FFT Mode
        # Handle Signal object input
        if isinstance(x, Signal):
            sorted_indices = sorted(x.samples.keys())
            x_array = np.array([x.samples[i][0] for i in sorted_indices])
        else:
            x_array = np.array(x)
        
        N = len(x_array)
        
        # Apply FFT
        X = _fft_recursive(x_array.astype(complex), is_inverse=False)
        
        # Trim to original size if padding was applied
        X = X[:N]
        
        # Extract amplitude and phase
        amplitudes = np.abs(X)
        phase_shifts = np.angle(X)
        
        # Generate frequency values
        fundamental_frequency = sampling_frequency / N
        frequencies = np.array([k * fundamental_frequency for k in range(N)])
        
        return frequencies, amplitudes, phase_shifts
    
    else:
        # IFFT Mode
        # Handle tuple input (amplitudes, phase_shifts)
        if isinstance(x, tuple):
            amplitudes, phase_shifts = x
        else:
            # Assume x is already complex array
            amplitudes = np.abs(x)
            phase_shifts = np.angle(x)
        
        N = len(amplitudes)
        
        # Convert polar form to complex form
        X = amplitudes * np.exp(1j * phase_shifts)
        
        # Apply IFFT
        x_reconstructed = _fft_recursive(X, is_inverse=True)
        
        # Normalize by N for IFFT
        x_reconstructed = x_reconstructed / N
        
        # Take real part (imaginary part should be negligible)
        x_real = np.real(x_reconstructed)
        
        # Create Signal object
        result_signal = Signal()
        result_signal.N1 = N
        result_signal.samples = {i: [x_real[i]] for i in range(N)}
        result_signal.is_periodic = False
        
        return result_signal


# Legacy function names for backward compatibility
def dft(signal, sampling_frequency):
    """
    Compute the Discrete Fourier Transform using FFT
    (Wrapper for backward compatibility)
    
    Args:
        signal: Signal object (time domain)
        sampling_frequency: Sampling frequency in Hz
    
    Returns:
        tuple: (frequencies, amplitudes, phase_shifts)
    """
    return fft_ifft(signal, sampling_frequency, is_inverse=False)


def idft(frequencies, amplitudes, phase_shifts):
    """
    Compute the Inverse Discrete Fourier Transform using IFFT
    (Wrapper for backward compatibility)
    
    Args:
        frequencies: array of frequency values
        amplitudes: array of amplitude values
        phase_shifts: array of phase shift values in radians
    
    Returns:
        Signal object in time domain
    """
    return fft_ifft((amplitudes, phase_shifts), sampling_frequency=1.0, is_inverse=True)


def remove_dc_component(amplitudes, phase_shifts):
    """
    Remove the DC component (zero frequency component) from frequency domain signal
    
    Args:
        amplitudes: array of amplitude values
        phase_shifts: array of phase shift values
    
    Returns:
        tuple: (modified_amplitudes, modified_phase_shifts) with DC removed
    """
    # Copy arrays to avoid modifying originals
    modified_amplitudes = amplitudes.copy()
    modified_phase_shifts = phase_shifts.copy()
    
    # Set the DC component (first element) to zero
    modified_amplitudes[0] = 0.0
    modified_phase_shifts[0] = 0.0
    
    return modified_amplitudes, modified_phase_shifts


# ============================================================================
# TIME DOMAIN OPERATIONS (Task 6)
# ============================================================================

def moving_average(signal, window_size):
    """
    Compute moving average (smoothing) using convolution
    
    Args:
        signal: Signal object
        window_size: Number of points to average
    
    Returns:
        Signal object with smoothed values
    """
    sorted_indices = sorted(signal.samples.keys())
    samples = [signal.samples[i][0] for i in sorted_indices]
    
    # Create moving average filter (all ones divided by window size)
    filter_values = [1.0 / window_size] * window_size
    
    # Apply convolution for smoothing
    smoothed = []
    n = len(samples)
    
    for i in range(n):
        avg_sum = 0.0
        count = 0
        for j in range(window_size):
            idx = i - j
            if 0 <= idx < n:
                avg_sum += samples[idx]
                count += 1
        smoothed.append(avg_sum / count if count > 0 else samples[i])
    
    # Create result signal
    result_signal = Signal()
    result_signal.N1 = len(smoothed)
    result_signal.samples = {sorted_indices[i]: [smoothed[i]] for i in range(len(smoothed))}
    result_signal.is_periodic = signal.is_periodic if hasattr(signal, 'is_periodic') else False
    
    return result_signal


def first_derivative(signal):
    """
    Compute first derivative: Y(n) = x(n) - x(n-1)
    
    Args:
        signal: Signal object
    
    Returns:
        Signal object with first derivative
    """
    sorted_indices = sorted(signal.samples.keys())
    samples = [signal.samples[i][0] for i in sorted_indices]
    
    # Compute first derivative
    derivative = []
    for i in range(len(samples)):
        if i == 0:
            # For first sample, use forward difference
            derivative.append(samples[i])
        else:
            derivative.append(samples[i] - samples[i-1])
    
    # Create result signal
    result_signal = Signal()
    result_signal.N1 = len(derivative)
    result_signal.samples = {sorted_indices[i]: [derivative[i]] for i in range(len(derivative))}
    result_signal.is_periodic = False
    
    return result_signal


def second_derivative(signal):
    """
    Compute second derivative: Y(n) = x(n+1) - 2*x(n) + x(n-1)
    
    Args:
        signal: Signal object
    
    Returns:
        Signal object with second derivative
    """
    sorted_indices = sorted(signal.samples.keys())
    samples = [signal.samples[i][0] for i in sorted_indices]
    
    # Compute second derivative
    derivative = []
    for i in range(len(samples)):
        if i == 0:
            # For first sample
            if len(samples) > 1:
                derivative.append(samples[1] - samples[0])
            else:
                derivative.append(0.0)
        elif i == len(samples) - 1:
            # For last sample
            derivative.append(samples[i] - samples[i-1])
        else:
            derivative.append(samples[i+1] - 2*samples[i] + samples[i-1])
    
    # Create result signal
    result_signal = Signal()
    result_signal.N1 = len(derivative)
    result_signal.samples = {sorted_indices[i]: [derivative[i]] for i in range(len(derivative))}
    result_signal.is_periodic = False
    
    return result_signal


def shift_signal(signal, k):
    """
    Shift signal by k steps (positive k = delay, negative k = advance)
    
    Args:
        signal: Signal object
        k: Number of steps to shift
    
    Returns:
        Signal object with shifted indices
    """
    result_signal = Signal()
    result_signal.N1 = signal.N1
    result_signal.samples = {}
    
    # Shift all indices by k
    for index, value in signal.samples.items():
        result_signal.samples[index + k] = value
    
    result_signal.is_periodic = signal.is_periodic if hasattr(signal, 'is_periodic') else False
    
    return result_signal


def fold_signal(signal):
    """
    Fold signal (time reversal): x(-n)
    
    Args:
        signal: Signal object
    
    Returns:
        Signal object with folded indices
    """
    result_signal = Signal()
    result_signal.N1 = signal.N1
    result_signal.samples = {}
    
    # Negate all indices
    for index, value in signal.samples.items():
        result_signal.samples[-index] = value
    
    result_signal.is_periodic = signal.is_periodic if hasattr(signal, 'is_periodic') else False
    
    return result_signal


def fold_and_shift_signal(signal, k):
    """
    Fold then shift signal by k steps
    
    Args:
        signal: Signal object
        k: Number of steps to shift after folding
    
    Returns:
        Signal object with folded and shifted indices
    """
    # First fold
    folded = fold_signal(signal)
    # Then shift
    result = shift_signal(folded, k)
    
    return result


def remove_dc_time_domain(signal):
    """
    Remove DC component in time domain by subtracting mean
    
    Args:
        signal: Signal object
    
    Returns:
        Signal object with DC removed
    """
    sorted_indices = sorted(signal.samples.keys())
    samples = [signal.samples[i][0] for i in sorted_indices]
    
    # Calculate mean
    mean_value = sum(samples) / len(samples)
    
    # Subtract mean from all samples
    result_signal = Signal()
    result_signal.N1 = signal.N1
    result_signal.samples = {idx: [signal.samples[idx][0] - mean_value] for idx in sorted_indices}
    result_signal.is_periodic = signal.is_periodic if hasattr(signal, 'is_periodic') else False
    
    return result_signal


def convolve_signals(signal1, signal2):
    """
    Convolve two signals using the standard mathematical formula:
    y[n] = sum(x[k] * h[n-k])
    """
    # 1. Get sorted indices and samples just like before
    indices1 = sorted(signal1.samples.keys())
    samples1 = [signal1.samples[i][0] for i in indices1]
    
    indices2 = sorted(signal2.samples.keys())
    samples2 = [signal2.samples[i][0] for i in indices2]
    
    len1 = len(samples1)
    len2 = len(samples2)
    
    # 2. Determine Output Length
    output_length = len1 + len2 - 1
    
    # Initialize output list
    conv_result = []
    
    # 3. THE NORMAL METHOD (Output-Side Algorithm)
    # We iterate through every point 'n' in the result signal first
    for n in range(output_length):
        
        # Accumulator for this specific point y[n]
        current_sum = 0.0
        
        # Inner loop: Iterate through 'k' (Variable for Signal 1)
        # We try to multiply samples1[k] * samples2[n-k]
        for k in range(len1):
            
            # 4. Check Bounds for Signal 2
            # We need the index (n-k) to be valid for Signal 2
            if 0 <= n - k < len2:
                current_sum += samples1[k] * samples2[n - k]
        
        conv_result.append(current_sum)
    
    # 5. Calculate output indices (Same as before)
    min_index = indices1[0] + indices2[0]
    output_indices = [min_index + i for i in range(output_length)]
    
    # Create result signal
    result_signal = Signal()
    result_signal.N1 = output_length
    result_signal.samples = {output_indices[i]: [conv_result[i]] for i in range(output_length)}
    result_signal.is_periodic = False
    
    return result_signal


def cross_correlation(signal1, signal2, normalize=True):
    """
    Compute cross-correlation of two signals using circular/periodic correlation
    
    Args:
        signal1: First Signal object
        signal2: Second Signal object
        normalize: If True, compute normalized cross-correlation
    
    Returns:
        Signal object containing correlation result
    """
    # Get samples
    indices1 = sorted(signal1.samples.keys())
    samples1 = np.array([signal1.samples[i][0] for i in indices1])
    
    indices2 = sorted(signal2.samples.keys())
    samples2 = np.array([signal2.samples[i][0] for i in indices2])
    
    N = len(samples1)
    
    # Compute correlation manually using circular/periodic approach
    # R(j) = sum(x1[n] * x2[(n+j) mod N]) for all n
    correlation = []
    
    for lag in range(N):
        sum_val = 0.0
        for n in range(N):
            # Use modulo for circular wrapping
            sum_val += samples1[n] * samples2[(n + lag) % N]
        correlation.append(sum_val)
    
    # Normalize if requested
    if normalize:
        # Calculate normalization factor: sqrt(sum(x1^2) * sum(x2^2))
        sum_sq1 = np.sum(samples1 ** 2)
        sum_sq2 = np.sum(samples2 ** 2)
        norm_factor = np.sqrt(sum_sq1 * sum_sq2)
        
        if norm_factor > 0:
            correlation = [c / norm_factor for c in correlation]
    
    # Create result signal
    result_signal = Signal()
    result_signal.N1 = len(correlation)
    result_signal.samples = {i: [correlation[i]] for i in range(len(correlation))}
    result_signal.is_periodic = False
    
    return result_signal


def auto_correlation(signal, normalize=True):
    """
    Compute auto-correlation of a signal
    
    Args:
        signal: Signal object
        normalize: If True, compute normalized auto-correlation
    
    Returns:
        Signal object containing auto-correlation result
    """
    return cross_correlation(signal, signal, normalize)


def periodic_cross_correlation(signal1, signal2, normalize=True):
    """
    Compute normalized cross-correlation of periodic signals (can be different lengths)
    
    Args:
        signal1: First Signal object (periodic)
        signal2: Second Signal object (periodic)
        normalize: If True, compute normalized correlation
    
    Returns:
        Signal object containing correlation result
    """
    # Get samples
    indices1 = sorted(signal1.samples.keys())
    samples1 = np.array([signal1.samples[i][0] for i in indices1])
    
    indices2 = sorted(signal2.samples.keys())
    samples2 = np.array([signal2.samples[i][0] for i in indices2])
    
    N1 = len(samples1)
    N2 = len(samples2)
    N = max(N1, N2)
    
    # For periodic signals, compute correlation over one period
    correlation = []
    
    for lag in range(N):
        sum_val = 0.0
        for n in range(N):
            idx1 = n % N1
            idx2 = (n + lag) % N2
            sum_val += samples1[idx1] * samples2[idx2]
        correlation.append(sum_val / N)
    
    # Normalize if requested
    if normalize:
        sum_sq1 = np.sum(samples1 ** 2) / N1
        sum_sq2 = np.sum(samples2 ** 2) / N2
        norm_factor = np.sqrt(sum_sq1 * sum_sq2)
        
        if norm_factor > 0:
            correlation = [c / norm_factor for c in correlation]
    
    # Create result signal
    result_signal = Signal()
    result_signal.N1 = len(correlation)
    result_signal.samples = {i: [correlation[i]] for i in range(len(correlation))}
    result_signal.is_periodic = True
    
    return result_signal


def time_delay_analysis(signal1, signal2, sampling_period):
    """
    Perform time delay analysis between two periodic signals
    Find the delay between them using cross-correlation
    
    Args:
        signal1: First Signal object (periodic)
        signal2: Second Signal object (periodic)
        sampling_period: Sampling period (Ts) in seconds
    
    Returns:
        tuple: (delay_samples, delay_time)
            delay_samples: delay in number of samples
            delay_time: delay in seconds
    """
    # Compute cross-correlation
    corr_signal = periodic_cross_correlation(signal1, signal2, normalize=True)
    
    # Find the index of maximum correlation
    indices = sorted(corr_signal.samples.keys())
    correlations = [corr_signal.samples[i][0] for i in indices]
    
    max_corr_idx = correlations.index(max(correlations))
    delay_samples = indices[max_corr_idx]
    
    # Convert to time
    delay_time = delay_samples * sampling_period
    
    return delay_samples, delay_time
