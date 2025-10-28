"""
TESTING GUIDE - Lab 4 Frequency Domain Implementation
======================================================

This guide explains how to test all implemented features using the provided test cases.

SETUP
=====

1. Ensure you have the application running:
   > python f1.py

2. Test files are located in:
   - Lab 4/Test Cases/DFT/input_Signal_DFT.txt
   - Lab 4/Test Cases/IDFT/Output_Signal_IDFT.txt  
   - Lab 4/Remove DC component/DC_component_input.txt
   - Lab 4/Remove DC component/DC_component_output.txt

TEST 1: DFT FUNCTIONALITY
==========================

Objective: Verify DFT computes correct frequency domain representation

Steps:
1. Click "📁 Import Signal"
2. Load: Lab 4/Test Cases/DFT/input_Signal_DFT.txt
3. Signal "input_Signal_DFT" is now loaded
4. Click "🌊 Apply DFT"
5. Type: input_Signal_DFT
6. Enter sampling frequency: 1 (Hz)
7. View the plots and console output

Expected Results:
- Amplitude vs Frequency plot appears
- Phase vs Frequency plot appears
- Console shows table with indices, frequencies, amplitudes, phases
- Signal "input_Signal_DFT_DFT" is registered

Verification:
- Check console values match expected frequency components
- Amplitudes should be normalized to [0, 1]
- All 8 frequency components should be displayed

TEST 2: IDFT ROUND-TRIP
========================

Objective: Verify IDFT correctly reconstructs the original signal

Steps:
1. Using the DFT signal from Test 1 ("input_Signal_DFT_DFT")
2. Click "↩️ Apply IDFT"
3. Type: input_Signal_DFT_DFT
4. View the reconstructed signal plot

Expected Results:
- Time domain signal is reconstructed
- Signal "input_Signal_DFT_DFT_IDFT" is registered
- Waveform plot appears

Verification:
- Compare original "input_Signal_DFT" with "input_Signal_DFT_DFT_IDFT"
- Values should match within 0.001 tolerance
- Use "📊 View Signal Plot" to compare both signals visually

Manual Verification:
- Click "📊 View Signal Plot" → input_Signal_DFT (original)
- Click "📊 View Signal Plot" → input_Signal_DFT_DFT_IDFT (reconstructed)
- Compare the waveforms - they should be identical

TEST 3: DOMINANT FREQUENCIES
==============================

Objective: Identify frequency components with amplitude > 0.5

Steps:
1. Using the DFT signal from Test 1
2. Click "📊 Show Dominant Frequencies"
3. Type: input_Signal_DFT_DFT
4. View the console and message box

Expected Results:
- Console lists dominant frequencies
- Message box shows summary
- Only frequencies with normalized amplitude > 0.5 are listed

Verification:
- Check the amplitudes in the DFT console output
- Manually count components with amplitude > 0.5
- Compare with the dominant frequencies list

TEST 4: MODIFY FREQUENCY COMPONENT
===================================

Objective: Change amplitude/phase of a specific frequency

Steps:
1. Using the DFT signal from Test 1
2. Click "✏️ Modify Component"
3. Type: input_Signal_DFT_DFT
4. Enter index to modify: 0 (DC component)
5. Enter new amplitude: 0 (to remove DC)
6. Enter new phase: 0
7. View the updated plots

Expected Results:
- Signal "input_Signal_DFT_DFT_modified" is created
- Plots show F[0] = 0
- Console confirms modification

Verification:
- Check that index 0 now has amplitude 0
- Compare with original DFT plot
- Other frequencies should remain unchanged

TEST 5: REMOVE DC COMPONENT
============================

Objective: Remove the DC (zero frequency) component from signal

Test Data:
- Input: Lab 4/Remove DC component/DC_component_input.txt
- Expected Output: Lab 4/Remove DC component/DC_component_output.txt

Steps:
1. Click "📁 Import Signal"
2. Load: Lab 4/Remove DC component/DC_component_input.txt
3. Signal "DC_component_input" is loaded
4. Click "🌊 Apply DFT"
5. Type: DC_component_input
6. Enter sampling frequency: 1 (Hz)
7. Note the DC component value (F[0] amplitude)
8. Click "🚫 Remove DC Component"
9. Type: DC_component_input_DFT
10. Signal "DC_component_input_DFT_no_DC" is created
11. Click "↩️ Apply IDFT"
12. Type: DC_component_input_DFT_no_DC
13. Signal "DC_component_input_DFT_no_DC_IDFT" is reconstructed
14. Save this signal and compare with expected output

Expected Results:
- DC component is removed (F[0] = 0)
- Time domain signal has no DC offset
- Reconstructed signal matches DC_component_output.txt

Verification Using CompareSignals:
```python
# In Python console or script
from Lab_4.Remove_DC_component.CompareSignals import SignalsAreEqual

# After saving your reconstructed signal
your_output = "path/to/your/saved/file.txt"
expected = "Lab 4/Remove DC component/DC_component_output.txt"

# Load your signal
signal = load_signal_file(your_output)
indices = sorted(signal.samples.keys())
samples = [signal.samples[i][0] for i in indices]

# Compare
SignalsAreEqual("Remove DC Test", expected, indices, samples)
```

TEST 6: SAVE/LOAD FREQUENCY DOMAIN
===================================

Objective: Verify frequency domain signals can be saved and loaded

Steps:
1. Complete Test 1 to get a DFT signal
2. After computing DFT, click "Yes" to save
3. Save to: Lab 4/my_dft_output.txt
4. Close and restart the application
5. Click "📁 Import Signal"
6. Load: Lab 4/my_dft_output.txt
7. Check console - should say "Loaded as frequency domain signal"
8. Use this signal for IDFT, modify, or remove DC

Expected Results:
- File has format: 1 0 N (frequency domain indicator)
- File contains amplitude and phase pairs
- Signal loads correctly with all metadata
- Can be used for IDFT and other operations

Verification:
- Open saved file in text editor
- Check format matches frequency domain specification
- Reload and apply IDFT - should reconstruct original

USING TEST COMPARISON FUNCTIONS
================================

The provided signalcompare.py has two functions:

1. SignalComapreAmplitude(input, output)
   - Compares amplitude arrays
   - Returns True if match within 0.001
   - Use for DFT amplitude verification

2. SignalComaprePhaseShift(input, output)
   - Compares phase arrays (rounded)
   - Returns True if match
   - Use for DFT phase verification

Example Usage:
```python
# After computing DFT
your_amplitudes = [...]  # Your computed amplitudes
your_phases = [...]      # Your computed phases

# Load expected from reference file
expected_amplitudes = [...]
expected_phases = [...]

# Test
if SignalComapreAmplitude(your_amplitudes, expected_amplitudes):
    print("✓ Amplitude test passed")
    
if SignalComaprePhaseShift(your_phases, expected_phases):
    print("✓ Phase test passed")
```

CONSOLE VERIFICATION
====================

After each operation, check the console for:

✓ Operation confirmation messages
✓ Numerical tables with results
✓ Signal registration confirmations
✓ Warning/error messages if any

Example Console Output After DFT:
```
Computing DFT for 'input_Signal_DFT' with Fs = 1.0 Hz...
✓ Registered 'input_Signal_DFT_DFT' | Samples: 8

DFT Results for 'input_Signal_DFT':
Index    Freq (Hz)       Amplitude       Norm. Amp       Phase (rad)
----------------------------------------------------------------------
0        0.000           64.000          1.000           0.000
1        0.125           0.000           0.000           0.000
2        0.250           0.000           0.000           0.000
...
```

VISUAL VERIFICATION
===================

All operations provide visual plots:

DFT Plots:
- Top: Amplitude vs Frequency (stem plot)
- Bottom: Phase vs Frequency (stem plot)

IDFT Plots:
- Top: Discrete representation (scatter)
- Bottom: Continuous representation (line)

Check that:
✓ Plots display correctly
✓ Axes are labeled
✓ Values match console output
✓ Toolbar works (zoom, pan, save)

TROUBLESHOOTING
===============

Issue: "Please select a signal that has been transformed with DFT"
Solution: Make sure you selected a frequency domain signal (one created by DFT)

Issue: Plots don't appear
Solution: Check if matplotlib is installed and working

Issue: Values don't match expected
Solution: 
- Verify sampling frequency is correct
- Check signal was loaded properly
- Ensure no prior modifications

Issue: Can't load frequency domain file
Solution:
- Check file format (line 1 should be "1" for frequency domain)
- Verify amplitude and phase pairs format
- Check file has correct number of samples

SUCCESS CRITERIA
================

Your implementation is correct if:

✓ DFT produces correct amplitudes and phases
✓ IDFT reconstructs original signal (within 0.001 tolerance)
✓ Dominant frequencies are correctly identified
✓ Component modification works as expected
✓ DC removal reduces mean to near-zero
✓ Signals can be saved and loaded in correct format
✓ All plots display correctly
✓ Console output is clear and accurate

FINAL VERIFICATION
==================

Run all 6 tests in sequence:
1. DFT ✓
2. IDFT Round-trip ✓
3. Dominant Frequencies ✓
4. Modify Component ✓
5. Remove DC Component ✓
6. Save/Load ✓

If all tests pass, your implementation is complete and correct!
"""

print(__doc__)
