"""
Test script for Time Domain operations (Task 6)
Tests all implementations against provided test cases
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_processor import (
    Signal, load_signal_file, moving_average, first_derivative, second_derivative,
    shift_signal, fold_signal, fold_and_shift_signal, remove_dc_time_domain,
    convolve_signals, cross_correlation, auto_correlation, periodic_cross_correlation,
    time_delay_analysis
)
import numpy as np

# Test file paths
TEST_DIR = os.path.join("task7", "task_and_files")

def test_convolution():
    """Test convolution with provided test case"""
    print("\n" + "="*60)
    print("Testing Convolution")
    print("="*60)
    
    # Create test signals as per ConvTest.py
    signal1 = Signal()
    signal1.N1 = 4
    signal1.samples = {-2: [1], -1: [2], 0: [1], 1: [1]}
    
    signal2 = Signal()
    signal2.N1 = 6
    signal2.samples = {0: [1], 1: [-1], 2: [0], 3: [0], 4: [1], 5: [1]}
    
    # Perform convolution
    result = convolve_signals(signal1, signal2)
    
    # Extract results
    indices = sorted(result.samples.keys())
    samples = [result.samples[i][0] for i in indices]
    
    print(f"Result indices: {indices}")
    print(f"Result samples: {samples}")
    
    # Load test function
    conv_test_path = os.path.join(TEST_DIR, "TestCases", "Convolution", "ConvTest.py")
    with open(conv_test_path, 'r') as f:
        exec(f.read(), globals())
    
    # Run test
    ConvTest(indices, samples)


def test_shift_and_fold():
    """Test shift and fold operations"""
    print("\n" + "="*60)
    print("Testing Shift and Fold")
    print("="*60)
    
    # Load input signal
    input_file = os.path.join(TEST_DIR, "TestCases", "Shifting and Folding", "input_fold.txt")
    signal = load_signal_file(input_file)
    
    if not signal:
        print("Failed to load input signal")
        return
    
    # Load test function
    test_func_path = os.path.join(TEST_DIR, "TestCases", "Shifting and Folding", "Shift_Fold_Signal.py")
    with open(test_func_path, 'r') as f:
        exec(f.read(), globals())
    
    # Test 1: Fold only
    print("\nTest 1: Folding signal...")
    folded = fold_signal(signal)
    indices = sorted(folded.samples.keys())
    samples = [folded.samples[i][0] for i in indices]
    
    output_file = os.path.join(TEST_DIR, "TestCases", "Shifting and Folding", "Output_fold.txt")
    Shift_Fold_Signal(output_file, indices, samples)
    
    # Test 2: Fold and shift by 500
    print("\nTest 2: Fold and shift by 500...")
    fold_shift_500 = fold_and_shift_signal(signal, 500)
    indices = sorted(fold_shift_500.samples.keys())
    samples = [fold_shift_500.samples[i][0] for i in indices]
    
    output_file = os.path.join(TEST_DIR, "TestCases", "Shifting and Folding", "Output_ShifFoldedby500.txt")
    Shift_Fold_Signal(output_file, indices, samples)
    
    # Test 3: Fold and shift by -500
    print("\nTest 3: Fold and shift by -500...")
    fold_shift_minus500 = fold_and_shift_signal(signal, -500)
    indices = sorted(fold_shift_minus500.samples.keys())
    samples = [fold_shift_minus500.samples[i][0] for i in indices]
    
    output_file = os.path.join(TEST_DIR, "TestCases", "Shifting and Folding", "Output_ShiftFoldedby-500.txt")
    Shift_Fold_Signal(output_file, indices, samples)


def test_derivatives():
    """Test first and second derivatives"""
    print("\n" + "="*60)
    print("Testing Derivatives")
    print("="*60)
    
    # Create input signal as per DerivativeSignal.py
    signal = Signal()
    signal.N1 = 100
    signal.samples = {i: [float(i+1)] for i in range(100)}
    
    # Compute derivatives
    first_deriv = first_derivative(signal)
    second_deriv = second_derivative(signal)
    
    # Extract results (skip first sample for first derivative)
    first_deriv_values = [first_deriv.samples[i][0] for i in range(1, 100)]
    second_deriv_values = [second_deriv.samples[i][0] for i in range(1, 99)]
    
    print(f"First derivative (first 10): {first_deriv_values[:10]}")
    print(f"Second derivative (first 10): {second_deriv_values[:10]}")
    
    # Expected results
    expected_first = [1.0] * 99
    expected_second = [0.0] * 98
    
    # Check first derivative
    if len(first_deriv_values) == len(expected_first):
        max_error = max(abs(first_deriv_values[i] - expected_first[i]) for i in range(len(expected_first)))
        if max_error < 0.01:
            print("✓ First derivative test PASSED")
        else:
            print(f"✗ First derivative test FAILED (max error: {max_error})")
    else:
        print("✗ First derivative length mismatch")
    
    # Check second derivative
    if len(second_deriv_values) == len(expected_second):
        max_error = max(abs(second_deriv_values[i] - expected_second[i]) for i in range(len(expected_second)))
        if max_error < 0.01:
            print("✓ Second derivative test PASSED")
        else:
            print(f"✗ Second derivative test FAILED (max error: {max_error})")
    else:
        print("✗ Second derivative length mismatch")


def test_correlation():
    """Test correlation functions"""
    print("\n" + "="*60)
    print("Testing Correlation")
    print("="*60)
    
    # Load test signals
    signal1_file = os.path.join(TEST_DIR, "Point1 Correlation", "Corr_input signal1.txt")
    signal2_file = os.path.join(TEST_DIR, "Point1 Correlation", "Corr_input signal2.txt")
    
    signal1 = load_signal_file(signal1_file)
    signal2 = load_signal_file(signal2_file)
    
    if not signal1 or not signal2:
        print("Failed to load correlation test signals")
        return
    
    # Compute cross-correlation
    corr_result = cross_correlation(signal1, signal2, normalize=True)
    
    # Extract results
    indices = sorted(corr_result.samples.keys())
    samples = [corr_result.samples[i][0] for i in indices]
    
    print(f"Correlation result (first 10): {samples[:10]}")
    
    # Load test function
    test_func_path = os.path.join(TEST_DIR, "Point1 Correlation", "CompareSignal.py")
    with open(test_func_path, 'r') as f:
        exec(f.read(), globals())
    
    # Test with expected output
    output_file = os.path.join(TEST_DIR, "Point1 Correlation", "CorrOutput.txt")
    Compare_Signals(output_file, indices, samples)


def test_moving_average():
    """Test moving average smoothing"""
    print("\n" + "="*60)
    print("Testing Moving Average")
    print("="*60)
    
    # Create test signal (1 to 1000)
    signal = Signal()
    signal.N1 = 1000
    signal.samples = {i: [float(i+1)] for i in range(1000)}
    
    # Test with window size 3
    print("\nTest with window size 3...")
    smoothed = moving_average(signal, 3)
    
    indices = sorted(smoothed.samples.keys())
    samples = [smoothed.samples[i][0] for i in indices]
    
    print(f"Original first 10: {[signal.samples[i][0] for i in range(10)]}")
    print(f"Smoothed first 10: {samples[:10]}")
    
    # The moving average should smooth the signal
    print("✓ Moving average computed successfully")


def test_remove_dc_time():
    """Test DC removal in time domain"""
    print("\n" + "="*60)
    print("Testing DC Removal (Time Domain)")
    print("="*60)
    
    # Create signal with DC offset
    signal = Signal()
    signal.N1 = 10
    dc_offset = 5.0
    signal.samples = {i: [float(i) + dc_offset] for i in range(10)}
    
    print(f"Original samples: {[signal.samples[i][0] for i in range(10)]}")
    print(f"Mean (DC): {np.mean([signal.samples[i][0] for i in range(10)]):.2f}")
    
    # Remove DC
    no_dc = remove_dc_time_domain(signal)
    
    samples_no_dc = [no_dc.samples[i][0] for i in range(10)]
    print(f"After DC removal: {samples_no_dc}")
    print(f"New mean: {np.mean(samples_no_dc):.6f}")
    
    if abs(np.mean(samples_no_dc)) < 0.0001:
        print("✓ DC removal test PASSED")
    else:
        print("✗ DC removal test FAILED")


def main():
    print("="*60)
    print("TIME DOMAIN OPERATIONS TEST SUITE (Task 6)")
    print("="*60)
    
    try:
        test_convolution()
    except Exception as e:
        print(f"Convolution test error: {e}")
    
    try:
        test_shift_and_fold()
    except Exception as e:
        print(f"Shift/Fold test error: {e}")
    
    try:
        test_derivatives()
    except Exception as e:
        print(f"Derivatives test error: {e}")
    
    try:
        test_correlation()
    except Exception as e:
        print(f"Correlation test error: {e}")
    
    try:
        test_moving_average()
    except Exception as e:
        print(f"Moving average test error: {e}")
    
    try:
        test_remove_dc_time()
    except Exception as e:
        print(f"DC removal test error: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
