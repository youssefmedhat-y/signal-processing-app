# 🎵 Digital Signal Analysis Tool

A comprehensive digital signal processing application with GUI built using Python and Tkinter. This tool provides various signal analysis capabilities including time domain operations, frequency domain analysis (DFT/IDFT), quantization, and more.

## ✨ Features

### 📁 File Operations
- Import signals from text files
- Export processed signals
- Support for both time domain and frequency domain signals

### ⚡ Signal Generation
- Generate sine and cosine waves
- Configurable amplitude, frequency, phase, and sampling rate
- Support for periodic signals

### 🧮 Mathematical Operations
- **Add/Subtract Signals**: Combine or difference two signals
- **Multiply by Constant**: Scale signal amplitudes
- **Square Signal**: Square each sample value
- **Cumulative Sum**: Accumulate signal values
- **Normalize Signal**: Normalize to [-1, 1] or [0, 1] range

### 🔢 Quantization
- **Quantize by Bits**: Specify number of bits for quantization
- **Quantize by Levels**: Specify number of quantization levels
- Generate encoded values and quantization errors

### 🌊 Frequency Domain Analysis
- **DFT (Discrete Fourier Transform)**: Manual implementation without numpy.fft
- **IDFT (Inverse DFT)**: Signal reconstruction from frequency domain
- **Dominant Frequencies**: Identify frequencies with amplitude > 0.5
- **Modify Components**: Change amplitude/phase of specific frequency components
- **Remove DC Component**: Remove zero-frequency component
- Visual plots for amplitude and phase vs frequency

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib tkinter
```

### Running the Application

```bash
python f1.py
```

## 📖 Usage Guide

### Basic Workflow

1. **Load a Signal**
   - Click "📁 Import Signal"
   - Select a signal file (.txt format)
   - Signal appears in the signal list

2. **Apply DFT**
   - Click "🌊 Apply DFT"
   - Select signal from list
   - Enter sampling frequency
   - View frequency domain plots

3. **Analyze Frequencies**
   - Click "📊 Show Dominant Frequencies"
   - See frequencies with high amplitudes

4. **Remove DC Offset**
   - Click "🚫 Remove DC Component"
   - Works with both time and frequency domain signals

5. **Reconstruct Signal**
   - Click "↩️ Apply IDFT"
   - Get back time domain signal

### File Format

#### Time Domain Signals
```
0              # SignalType: 0 for time domain
0              # IsPeriodic: 0 for non-periodic, 1 for periodic
8              # Number of samples
0 1.5          # Index Amplitude
1 3.2          # Index Amplitude
...
```

#### Frequency Domain Signals
```
1              # SignalType: 1 for frequency domain
0              # IsPeriodic: 0 for non-periodic
8              # Number of components
10.5 0.0       # Amplitude Phase
5.25 1.571     # Amplitude Phase
...
```

## 🧪 Testing

Run the test suite:

```bash
python "Lab 4/test_dft.py"
```

Tests include:
- DFT computation accuracy
- IDFT round-trip verification
- DC component removal validation

## 📊 Signal Processing Labs

The project includes implementations for multiple lab assignments:

### Lab 1: Basic Signal Operations
- Signal addition and subtraction
- Multiplication by constant
- Signal normalization

### Lab 2: Advanced Operations
- Signal accumulation
- Squaring operations
- Difference calculations

### Lab 3: Quantization
- Bit-based quantization
- Level-based quantization
- Error analysis

### Lab 4: Frequency Domain
- DFT implementation
- IDFT implementation
- Dominant frequency detection
- Component modification
- DC removal

## 🎨 GUI Features

- **Scrollable Control Panel**: Easy access to all functions
- **List-Based Signal Selection**: No typing required
- **Interactive Plots**: Zoom, pan, and save capabilities
- **Console Output**: Detailed numerical results
- **Status Bar**: Real-time operation feedback

## 📝 Code Structure

```
f1/
├── f1.py                      # Main GUI application
├── signal_processor.py        # Core signal processing functions
├── Lab 1/                     # Lab 1 tasks and test cases
├── Lab 2/                     # Lab 2 tasks and test cases
├── Lab 3/                     # Lab 3 quantization tasks
│   └── Task files/
├── Lab 4/                     # Lab 4 frequency domain tasks
│   ├── Test Cases/
│   ├── Remove DC component/
│   ├── test_dft.py
│   └── QUICK_START_GUIDE.md
└── README.md
```

## 🔬 Technical Details

### DFT Implementation
- Manual computation without numpy.fft
- Formula: `X[k] = Σ x[n] * e^(-j2πkn/N)`
- Computes amplitude: `|X[k]| = sqrt(Real² + Imag²)`
- Computes phase: `φ[k] = arctan2(Imag, Real)`

### IDFT Implementation
- Manual computation without numpy.ifft
- Formula: `x[n] = (1/N) * Σ X[k] * e^(j2πkn/N)`
- Perfect reconstruction guarantee

### Quantization Methods
1. **Bits Method**: Levels = 2^bits
2. **Levels Method**: Direct level specification
3. Midpoint quantization for minimizing error

## 🐛 Known Issues

- The provided `SignalComapreAmplitude` function in test files has a logic bug with perfect matches
- Workaround implemented in test scripts

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is part of academic coursework for Digital Signal Processing.

## 👥 Authors

- Developed as part of DSP course labs

## 🙏 Acknowledgments

- Course instructors and TAs
- Test cases provided for validation
- Python scientific computing community

## 📸 Screenshots

### Main Interface
![Main Window](docs/screenshots/main_window.png)

### DFT Analysis
![DFT Plot](docs/screenshots/dft_plot.png)

### Signal Selection
![Signal Selection](docs/screenshots/signal_selection.png)

---

**Note**: This is an educational project demonstrating digital signal processing concepts.
