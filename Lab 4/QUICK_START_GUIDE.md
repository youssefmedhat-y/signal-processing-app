# Quick Start Guide - Frequency Domain Analysis

## Accessing the Features

Open the application by running:
```
python f1.py
```

Look for the **"Frequency Domain"** section in the left control panel (purple buttons).

## Feature Guide

### 1. 🌊 Apply DFT (Discrete Fourier Transform)

**What it does**: Converts a time domain signal to frequency domain

**Steps**:
1. Click "🌊 Apply DFT"
2. Type the name of your time domain signal (e.g., "Signal1")
3. Enter the sampling frequency in Hz (e.g., 8000)
4. View the results:
   - Two plots appear: Amplitude vs Frequency and Phase vs Frequency
   - Console shows a table with numerical values
   - A new signal is created with name `YourSignal_DFT`

**Example**:
```
Input: Signal1 (time domain with 8 samples)
Sampling Freq: 1 Hz
Output: Signal1_DFT (frequency domain with 8 components)
```

---

### 2. 📊 Show Dominant Frequencies

**What it does**: Finds frequency components with large amplitudes (> 0.5 normalized)

**Steps**:
1. First, apply DFT to a signal
2. Click "📊 Show Dominant Frequencies"
3. Type the name of your DFT signal (e.g., "Signal1_DFT")
4. View results:
   - Console lists all dominant frequencies
   - A message box shows a summary

**Use Case**: Identify the main frequencies in a signal (like finding musical notes)

---

### 3. ✏️ Modify Component

**What it does**: Change the amplitude or phase of any frequency component

**Steps**:
1. Click "✏️ Modify Component"
2. Select your DFT signal
3. Enter the index you want to modify (e.g., 0 for DC, 1 for first harmonic)
4. Enter new amplitude value
5. Enter new phase value (in radians)
6. A new modified signal is created with name `YourSignal_modified`

**Use Cases**:
- Filter out specific frequencies (set amplitude to 0)
- Boost certain frequencies (increase amplitude)
- Phase shifting for signal processing

---

### 4. 🚫 Remove DC Component

**What it does**: Removes the constant offset (DC component) from your signal

**Steps**:
1. Click "🚫 Remove DC Component"
2. Select your DFT signal
3. The DC component (F[0]) is automatically removed
4. A new signal is created with name `YourSignal_no_DC`

**Use Case**: Remove constant bias from sensors or normalize signals

---

### 5. ↩️ Apply IDFT (Inverse DFT)

**What it does**: Converts frequency domain back to time domain

**Steps**:
1. Click "↩️ Apply IDFT"
2. Select your DFT signal (original, modified, or DC-removed)
3. The time domain signal is reconstructed
4. A new signal is created with name `YourSignal_IDFT`
5. View the reconstructed waveform

**Use Cases**:
- Verify your DFT worked correctly (should match original)
- See the effect of modifications in time domain
- Reconstruct signal after filtering

---

## Complete Workflow Examples

### Example 1: Basic DFT Analysis
```
1. Import signal "MySine.txt" → creates "MySine" signal
2. Apply DFT with Fs=100 Hz → creates "MySine_DFT"
3. Show dominant frequencies → see which frequencies are strong
4. Apply IDFT → verify you get the original signal back
```

### Example 2: Remove DC Offset
```
1. Load signal with DC offset → "Sensor_Data"
2. Apply DFT → "Sensor_Data_DFT"
3. Remove DC Component → "Sensor_Data_DFT_no_DC"
4. Apply IDFT → "Sensor_Data_DFT_no_DC_IDFT"
5. Compare original and reconstructed to see DC removed
```

### Example 3: Frequency Filtering
```
1. Load noisy signal → "NoisySignal"
2. Apply DFT → "NoisySignal_DFT"
3. Show dominant frequencies → identify noise frequencies
4. Modify Component → set noise frequency amplitudes to 0
5. Apply IDFT → get clean signal
```

---

## Understanding the Output

### Frequency Plot (Top)
- **X-axis**: Frequency in Hz
- **Y-axis**: Normalized Amplitude (0 to 1)
- **Interpretation**: Peaks show the dominant frequencies

### Phase Plot (Bottom)
- **X-axis**: Frequency in Hz
- **Y-axis**: Phase in radians (-π to π)
- **Interpretation**: Shows phase shift of each frequency component

### Console Table
```
Index    Freq (Hz)    Amplitude    Norm. Amp    Phase (rad)
0        0.000        10.500       1.000        0.000        <- DC component
1        1.000        5.250        0.500        1.571        <- First harmonic
2        2.000        2.100        0.200       -1.571        <- Second harmonic
...
```

---

## Tips and Tricks

1. **Sampling Frequency**: 
   - Use the actual sampling rate of your signal
   - If unknown, use 1 Hz for relative frequency analysis

2. **Dominant Frequencies**:
   - Threshold is 0.5 (50% of maximum)
   - Adjust in code if you want different threshold

3. **Phase Values**:
   - Range: -π to π radians
   - 0 rad = no phase shift
   - π/2 rad = 90° shift
   - π rad = 180° shift (inverted)

4. **DC Component**:
   - Always at index 0
   - Represents average value of signal
   - Remove if you want zero-mean signal

5. **Signal Naming**:
   - Descriptive names help track processing steps
   - Use underscore for derived signals (e.g., "Signal1_DFT_no_DC")

---

## Troubleshooting

**"Please select a signal that has been transformed with DFT"**
- You're trying to use a time domain signal
- Apply DFT first

**"No signals in storage"**
- Import or generate a signal first
- Check the console for loaded signals

**Wrong frequency values**
- Check your sampling frequency input
- Formula: Frequency[k] = k * (Fs / N)

**IDFT doesn't match original**
- Small numerical errors are normal (< 0.001)
- Large errors indicate a problem - check DFT signal

---

## Testing Your Implementation

Use the provided test files in `Lab 4/Test Cases/`:

1. **DFT Test**:
   - Load `Lab 4/Test Cases/DFT/input_Signal_DFT.txt`
   - Apply DFT with Fs=1 Hz
   - Compare with expected output

2. **IDFT Test**:
   - Load `Lab 4/Test Cases/IDFT/Output_Signal_IDFT.txt`
   - Apply DFT then IDFT
   - Should get the same signal back

3. **Remove DC Test**:
   - Load `Lab 4/Remove DC component/DC_component_input.txt`
   - Apply DFT → Remove DC → Apply IDFT
   - Compare with `DC_component_output.txt`

---

## Summary

The Frequency Domain features allow you to:
- ✓ Analyze signal frequency content
- ✓ Identify dominant frequencies
- ✓ Filter and modify signals
- ✓ Remove DC offsets
- ✓ Reconstruct signals

All operations maintain signal integrity and provide visual feedback!
