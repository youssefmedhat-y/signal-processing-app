# Lab 4 - Frequency Domain Analysis Implementation

## Overview
This implementation adds comprehensive frequency domain analysis capabilities to the Digital Signal Analysis Tool, following the requirements in `task instructions.txt`.

## Implemented Features

### 1. Discrete Fourier Transform (DFT)
- **Location**: `signal_processor.py` - `dft()` function
- **Implementation**: Manual DFT computation without using `numpy.fft`
- **Formula**: 
  ```
  For each frequency k:
    Real[k] = Σ(n=0 to N-1) x[n] * cos(-2πkn/N)
    Imag[k] = Σ(n=0 to N-1) x[n] * sin(-2πkn/N)
    Amplitude[k] = sqrt(Real[k]² + Imag[k]²)
    Phase[k] = arctan2(Imag[k], Real[k])
  ```
- **Features**:
  - Prompts user for sampling frequency (Hz)
  - Computes amplitude and phase for all frequency components
  - Normalizes amplitudes to [0, 1] range
  - Displays frequency vs amplitude and frequency vs phase plots
  - Stores frequency domain signal for further processing

### 2. Display Dominant Frequencies
- **Location**: `f1.py` - `show_dominant_frequencies()` method
- **Functionality**: 
  - Identifies frequency components with normalized amplitude > 0.5
  - Lists dominant frequencies in console
  - Shows summary in message box
  - Helps identify main frequency content

### 3. Modify Signal Components
- **Location**: `f1.py` - `modify_frequency_component()` method
- **Functionality**:
  - Allows selection of any frequency index
  - Modify amplitude value
  - Modify phase value (in radians)
  - Creates new modified frequency domain signal
  - Shows visualization of modified spectrum

### 4. Remove DC Component
- **Location**: `signal_processor.py` - `remove_dc_component()` function
- **Functionality**:
  - Removes F[0] (zero frequency/DC component)
  - Sets amplitude[0] and phase[0] to zero
  - Useful for removing constant offset from signals
  - Creates new signal without DC

### 5. Inverse Discrete Fourier Transform (IDFT)
- **Location**: `signal_processor.py` - `idft()` function
- **Implementation**: Manual IDFT computation without using `numpy.ifft`
- **Formula**:
  ```
  For each time sample n:
    Real_freq[k] = Amplitude[k] * cos(Phase[k])
    Imag_freq[k] = Amplitude[k] * sin(Phase[k])
    
    x[n] = (1/N) * Σ(k=0 to N-1) [
      Real_freq[k] * cos(2πkn/N) - Imag_freq[k] * sin(2πkn/N)
    ]
  ```
- **Features**:
  - Reconstructs time domain signal from frequency components
  - Works with any frequency domain signal (original, modified, or DC-removed)
  - Displays reconstructed waveform
  - Verifies signal reconstruction accuracy

## GUI Integration

### New Menu: "Frequency Domain"
Located in the left control panel with 5 buttons:

1. **🌊 Apply DFT**
   - Select signal → Enter sampling frequency → View results

2. **📊 Show Dominant Frequencies**
   - Select DFT signal → View dominant frequencies

3. **✏️ Modify Component**
   - Select DFT signal → Choose index → Enter new amplitude/phase

4. **🚫 Remove DC Component**
   - Select DFT signal → DC removed automatically

5. **↩️ Apply IDFT**
   - Select DFT signal → View reconstructed time signal

## File Format Handling

The implementation correctly handles signal file format:
```
Line 1: IsPeriodic (0 or 1)
Line 2: SignalType (0=Time domain, 1=Frequency domain)
Line 3: Number of samples
Lines 4+: Index Value [Phase]
```

## Code Quality

### Modular Design
- Separate functions for DFT, IDFT, DC removal
- Reusable across different parts of the application
- Easy to test and maintain

### Well-Commented
- Clear explanation of formulas
- Step-by-step computation comments
- Function docstrings with parameters and returns

### Error Handling
- Validates user input
- Checks signal type before operations
- Provides clear error messages

### User-Friendly
- Intuitive button labels with emojis
- Console output with formatted tables
- Visual plots for easy interpretation
- Message boxes for important information

## Testing

Test files are provided in `Lab 4/Test Cases/`:
- `DFT/input_Signal_DFT.txt` - Test input for DFT
- `IDFT/Output_Signal_IDFT.txt` - Expected output for IDFT
- `signalcompare.py` - Comparison functions for validation
- `Remove DC component/` - DC removal test cases

## Usage Example

### Example 1: Apply DFT
1. Import a time domain signal
2. Click "🌊 Apply DFT"
3. Select the signal
4. Enter sampling frequency (e.g., 8000 Hz)
5. View amplitude and phase plots
6. Check console for numerical values

### Example 2: Remove DC and Reconstruct
1. Apply DFT to a signal (creates `signal_DFT`)
2. Click "🚫 Remove DC Component"
3. Select `signal_DFT`
4. New signal `signal_DFT_no_DC` is created
5. Click "↩️ Apply IDFT"
6. Select `signal_DFT_no_DC`
7. View reconstructed signal without DC offset

### Example 3: Find Dominant Frequencies
1. Apply DFT to a signal
2. Click "📊 Show Dominant Frequencies"
3. Select the DFT signal
4. View list of frequencies with amplitude > 0.5

## Implementation Notes

### Smart Code Reuse (as suggested in hint)
The DFT and IDFT implementations share the same mathematical structure:
- Both use summation over samples
- Both use sin/cos calculations
- IDFT is essentially DFT with sign reversal and normalization
- This makes the code elegant and easy to understand

### Amplitude Normalization
- Uses existing `normalize_signal()` function concept
- Normalizes by dividing by maximum amplitude
- Range: [0, 1] as required

### Frequency Calculation
- Fundamental frequency = Fs / N
- Frequency[k] = k * (Fs / N)
- Handles any sampling frequency

## Files Modified

1. **signal_processor.py**
   - Added `dft()` function
   - Added `idft()` function
   - Added `remove_dc_component()` function

2. **f1.py**
   - Added imports for new functions
   - Added "Frequency Domain" section to GUI
   - Added 5 new methods for frequency domain operations
   - Added visualization function for frequency plots

## Dependencies

- numpy (for mathematical operations)
- matplotlib (for plotting)
- tkinter (for GUI - already present)

## Compliance with Requirements

✓ Apply DFT without using numpy.fft  
✓ Display frequency vs amplitude (normalized to [0,1])  
✓ Display frequency vs phase  
✓ Ask user for sampling frequency  
✓ Display dominant frequencies (amplitude > 0.5)  
✓ Modify amplitude and phase of components  
✓ Remove DC component (F[0])  
✓ Signal reconstruction using IDFT  
✓ Smart code design (DFT and IDFT similarity)  
✓ Maintain good GUI design  
✓ Well-commented, easy to understand code  
✓ Signal format handling (periodic and domain flags)  

## Summary

This implementation provides a complete frequency domain analysis suite integrated seamlessly into the existing Digital Signal Analysis Tool. All features from the task instructions have been implemented with clean, well-documented code that maintains the application's quality standards.
