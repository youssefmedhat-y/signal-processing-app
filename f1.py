import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import os
from signal_processor import Signal, write_signal_file, add_signals, subtract_signals, multiply_signal, square_signal, \
    accumulate_signal, normalize_signal, generate_sin_cos, load_signal_file, quantize_signal_bits, quantize_signal_levels, \
    dft, idft, remove_dc_component
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Storage for signals in memory
signal_storage = {}


class SignalProcessorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digital Signal Analysis Tool")
        self.geometry("1100x750")
        self.configure(bg='#2c3e50')

        self.setup_interface()

    def setup_interface(self):
        # Header section with title
        header = tk.Frame(self, bg='#000000', height=60)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        title_label = tk.Label(
            header,
            text="Digital Signal Analysis Tool",
            font=('Helvetica', 18, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        title_label.pack(pady=15)

        # Main container with two columns
        main_container = tk.Frame(self, bg='#000000')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls with scrollbar
        left_panel_container = tk.Frame(main_container, bg='#34495e', width=300)
        left_panel_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel_container.pack_propagate(False)
        
        # Create canvas and scrollbar for left panel
        canvas = tk.Canvas(left_panel_container, bg='#34495e', highlightthickness=0)
        scrollbar = tk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#34495e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.build_control_panel(scrollable_frame)

        # Right panel for output
        right_panel = tk.Frame(main_container, bg='#34495e')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.build_output_panel(right_panel)

        # Bottom status bar
        self.status_label = tk.Label(
            self,
            text="System Ready",
            bd=1,
            relief=tk.FLAT,
            anchor=tk.W,
            bg='#1abc9c',
            fg='white',
            font=('Arial', 9),
            padx=10,
            pady=5
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def build_control_panel(self, parent):
        # File Operations Section
        file_section = tk.LabelFrame(
            parent,
            text=" File Operations ",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            bd=2
        )
        file_section.pack(fill=tk.X, padx=10, pady=10)

        btn_load = tk.Button(
            file_section,
            text="📁 Import Signal",
            command=self.import_signal_file,
            bg='#3498db',
            fg='white',
            font=('Arial', 10),
            cursor='hand2',
            relief=tk.FLAT,
            padx=15,
            pady=8
        )
        btn_load.pack(fill=tk.X, padx=10, pady=5)

        btn_view = tk.Button(
            file_section,
            text="📊 View Signal Plot",
            command=self.view_signal_plot,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10),
            cursor='hand2',
            relief=tk.FLAT,
            padx=15,
            pady=8
        )
        btn_view.pack(fill=tk.X, padx=10, pady=5)

        # Signal Generation Section
        gen_section = tk.LabelFrame(
            parent,
            text=" Signal Generator ",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            bd=2
        )
        gen_section.pack(fill=tk.X, padx=10, pady=10)

        btn_generate = tk.Button(
            gen_section,
            text="⚡ Create New Signal",
            command=self.open_generator_window,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10),
            cursor='hand2',
            relief=tk.FLAT,
            padx=15,
            pady=8
        )
        btn_generate.pack(fill=tk.X, padx=10, pady=5)

        # Mathematical Operations Section
        math_section = tk.LabelFrame(
            parent,
            text=" Mathematical Operations ",
            font=('Arial', 10, 'bold'),
            bg='darkgrey',
            fg='#ecf0f1',
            bd=2
        )
        math_section.pack(fill=tk.X, padx=10, pady=10)

        operations = [
            ("➕ Add Two Signals", self.perform_addition),
            ("➖ Subtract Signals", self.perform_subtraction),
            ("✖️ Scale by Constant", self.perform_scaling),
            ("² Square Signal", lambda: self.single_signal_op('square')),
            ("Σ Cumulative Sum", lambda: self.single_signal_op('accumulate')),
            ("⚖️ Normalize Signal", self.open_normalize_window),
            ("🔢 Quantize by Bits", self.perform_quantization_bits),
            ("🔢 Quantize by Levels", self.perform_quantization_levels)
        ]

        for text, cmd in operations:
            btn = tk.Button(
                math_section,
                text=text,
                command=cmd,
                bg='#16a085',
                fg='white',
                font=('Arial', 9),
                cursor='hand2',
                relief=tk.FLAT,
                padx=10,
                pady=5
            )
            btn.pack(fill=tk.X, padx=10, pady=3)

        # Frequency Domain Section
        freq_section = tk.LabelFrame(
            parent,
            text=" Frequency Domain ",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            bd=2
        )
        freq_section.pack(fill=tk.X, padx=10, pady=10)

        freq_operations = [
            ("🌊 Apply DFT", self.perform_dft),
            ("📊 Show Dominant Frequencies", self.show_dominant_frequencies),
            ("✏️ Modify Component", self.modify_frequency_component),
            ("🚫 Remove DC Component", self.perform_remove_dc),
            ("↩️ Apply IDFT", self.perform_idft)
        ]

        for text, cmd in freq_operations:
            btn = tk.Button(
                freq_section,
                text=text,
                command=cmd,
                bg='#8e44ad',
                fg='white',
                font=('Arial', 9),
                cursor='hand2',
                relief=tk.FLAT,
                padx=10,
                pady=5
            )
            btn.pack(fill=tk.X, padx=10, pady=3)

    def build_output_panel(self, parent):
        output_label = tk.Label(
            parent,
            text="Console Output",
            font=('Arial', 11, 'bold'),
            bg='#000000',
            fg='#ecf0f1'
        )
        output_label.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Text widget with scrollbar
        text_frame = tk.Frame(parent, bg='#34495e')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_output = tk.Text(
            text_frame,
            wrap=tk.WORD,
            height=30,
            bg='#ecf0f1',
            fg='#2c3e50',
            font=('Courier', 9),
            yscrollcommand=scrollbar.set
        )
        self.console_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.console_output.yview)

    def write_to_console(self, text):
        """Append text to console output area"""
        self.console_output.insert(tk.END, text + "\n")
        self.console_output.see(tk.END)
        self.status_label.config(text=text)

    def register_signal(self, name, sig_obj):
        """Store signal and log information"""
        signal_storage[name] = sig_obj
        info = f"✓ Registered '{name}' | Samples: {sig_obj.N1}"
        self.write_to_console(info)

    def ask_for_save(self, sig_obj, suggested_name):
        """Prompt user to save signal to file"""
        response = messagebox.askyesno("Save Output", "Would you like to export this signal to a file?")
        if response:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                initialfile=suggested_name
            )
            if filepath:
                # Pass the is_periodic flag if it exists in the signal object
                is_periodic = getattr(sig_obj, 'is_periodic', False)
                write_signal_file(sig_obj, filepath, is_periodic)
                self.write_to_console(f"✓ Exported to: {filepath}")

    def select_signal(self, instruction):
        """Dialog for selecting a signal from storage using a listbox"""
        if not signal_storage:
            messagebox.showerror("No Signals", "Please load or generate a signal first.")
            return None, None

        available = list(signal_storage.keys())
        
        # Create a custom dialog with listbox
        dialog = tk.Toplevel(self)
        dialog.title("Signal Selection")
        dialog.geometry("450x400")
        dialog.configure(bg='#ecf0f1')
        dialog.resizable(False, False)
        
        # Make it modal
        dialog.transient(self)
        dialog.grab_set()
        
        # Instruction label
        tk.Label(
            dialog,
            text=instruction,
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            wraplength=420,
            justify='left'
        ).pack(pady=(20, 10), padx=20)
        
        # Frame for listbox and scrollbar
        list_frame = tk.Frame(dialog, bg='#ecf0f1')
        list_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox
        listbox = tk.Listbox(
            list_frame,
            font=('Courier', 10),
            bg='white',
            fg='#2c3e50',
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            height=12
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox with signals
        for signal_name in available:
            sig = signal_storage[signal_name]
            # Add signal info (name and sample count)
            display_text = f"{signal_name} ({sig.N1} samples)"
            listbox.insert(tk.END, display_text)
        
        # Select first item by default
        if available:
            listbox.selection_set(0)
            listbox.activate(0)
        
        # Variable to store selection
        selected_signal = [None]
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                signal_name = available[index]
                selected_signal[0] = (signal_name, signal_storage[signal_name])
                dialog.destroy()
        
        def on_cancel():
            selected_signal[0] = None
            dialog.destroy()
        
        # Double-click to select
        listbox.bind('<Double-Button-1>', lambda e: on_select())
        
        # Button frame
        button_frame = tk.Frame(dialog, bg='#ecf0f1')
        button_frame.pack(pady=15, padx=20)
        
        # Select button
        tk.Button(
            button_frame,
            text="Select",
            command=on_select,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=12,
            cursor='hand2',
            relief=tk.FLAT,
            padx=10,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        tk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=12,
            cursor='hand2',
            relief=tk.FLAT,
            padx=10,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        # Center dialog on parent window
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Wait for dialog to close
        self.wait_window(dialog)
        
        if selected_signal[0] is not None:
            return selected_signal[0]
        else:
            return None, None

    # === FILE OPERATIONS ===
    def import_signal_file(self):
        """Load signal from file"""
        filepath = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Choose Signal File"
        )
        if not filepath:
            return

        base_name = os.path.basename(filepath).split('.')[0]

        # Handle name conflicts
        signal_name = base_name
        counter = 1
        while signal_name in signal_storage:
            signal_name = f"{base_name}_v{counter}"
            counter += 1

        self.write_to_console(f"Loading from: {filepath}")

        sig_obj = load_signal_file(filepath)

        if sig_obj:
            # Ask user if the signal is periodic
            periodic_dialog = tk.Toplevel(self)
            periodic_dialog.title("Signal Periodicity")
            periodic_dialog.geometry("300x150")
            periodic_dialog.configure(bg='#ecf0f1')
            periodic_dialog.transient(self)
            periodic_dialog.grab_set()
            
            # Center the dialog
            periodic_dialog.update_idletasks()
            width = periodic_dialog.winfo_width()
            height = periodic_dialog.winfo_height()
            x = (periodic_dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (periodic_dialog.winfo_screenheight() // 2) - (height // 2)
            periodic_dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            # Add a label
            tk.Label(
                periodic_dialog, 
                text="Is this a periodic signal?",
                font=('Arial', 11, 'bold'),
                bg='#ecf0f1',
                pady=10
            ).pack()
            
            # Add help text
            tk.Label(
                periodic_dialog, 
                text="This affects the periodic flag in the signal file\nand determines how the signal will be processed.",
                font=('Arial', 9),
                bg='#ecf0f1',
                fg='#7f8c8d',
                pady=5
            ).pack()
            
            # Button frame
            btn_frame = tk.Frame(periodic_dialog, bg='#ecf0f1')
            btn_frame.pack(pady=15)
            
            def set_periodic(is_periodic):
                sig_obj.is_periodic = is_periodic
                self.register_signal(signal_name, sig_obj)
                status = "periodic" if is_periodic else "non-periodic"
                self.write_to_console(f"✓ Import successful: '{signal_name}' (marked as {status})")
                periodic_dialog.destroy()
            
            tk.Button(
                btn_frame,
                text="Yes (Periodic)",
                command=lambda: set_periodic(True),
                bg='#2ecc71',
                fg='white',
                font=('Arial', 10),
                width=12,
                padx=10
            ).grid(row=0, column=0, padx=10)
            
            tk.Button(
                btn_frame,
                text="No (Aperiodic)",
                command=lambda: set_periodic(False),
                bg='#e74c3c',
                fg='white',
                font=('Arial', 10),
                width=12,
                padx=10
            ).grid(row=0, column=1, padx=10)
        else:
            messagebox.showerror("Import Failed", f"Could not load signal from {filepath}")

    def view_signal_plot(self):
        """Display selected signal"""
        name, sig = self.select_signal("Enter signal name to visualize:")
        if sig:
            self.write_to_console(f"Opening plot for '{name}'...")
            self.show_signal_visualization(sig, f"Signal Visualization: {name}")

    def show_signal_visualization(self, signal_obj, chart_title):
        """Display signal with interactive controls"""
        indices = np.array(sorted(signal_obj.samples.keys()))
        amplitudes = np.array([signal_obj.samples[i][0] for i in indices])

        # Check if the signal is periodic
        is_periodic = hasattr(signal_obj, 'is_periodic') and signal_obj.is_periodic

        # Create figure
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle(chart_title, fontsize=16, fontweight='bold')

        # Create subplots
        ax_discrete = fig.add_subplot(2, 1, 1)
        ax_continuous = fig.add_subplot(2, 1, 2)

        # Discrete plot
        ax_discrete.scatter(indices, amplitudes, marker='o', s=50, color='#3498db',
                            label='Discrete Samples', zorder=3)
        ax_discrete.set_title('Discrete Representation', fontsize=12, fontweight='bold')
        ax_discrete.set_xlabel('Sample Index', fontsize=10)
        ax_discrete.set_ylabel('Amplitude', fontsize=10)
        ax_discrete.grid(True, linestyle='--', alpha=0.5)
        ax_discrete.legend(loc='best')

        # Continuous plot - use enhanced visualization for periodic signals
        if is_periodic and len(indices) > 10:
            # For periodic signals, we can create a smoother continuous representation
            # by adding more interpolated points
            x_smooth = np.linspace(indices.min(), indices.max(), len(indices) * 10)
            # Use scipy's interp1d for a more accurate interpolation of periodic signals
            from scipy.interpolate import interp1d
            
            try:
                # Try cubic spline interpolation first for smoother curves
                f = interp1d(indices, amplitudes, kind='cubic')
                smooth_curve = True
            except (ValueError, ImportError):
                # Fall back to linear interpolation if cubic fails
                try:
                    f = interp1d(indices, amplitudes, kind='linear')
                    smooth_curve = True
                except (ValueError, ImportError):
                    smooth_curve = False
            
            if smooth_curve:
                # Plot the smooth interpolated line
                ax_continuous.plot(x_smooth, f(x_smooth), color='#e74c3c', linestyle='-',
                                linewidth=2, label='Continuous Approximation', zorder=2)
            else:
                # Fall back to original approach if interpolation fails
                ax_continuous.plot(indices, amplitudes, color='#e74c3c', linestyle='-',
                                linewidth=2, label='Continuous Approximation', zorder=2)
        else:
            # Standard plot for non-periodic signals
            ax_continuous.plot(indices, amplitudes, color='#e74c3c', linestyle='-',
                            linewidth=2, label='Continuous Approximation', zorder=2)
        
        # Always plot the actual data points
        ax_continuous.scatter(indices, amplitudes, marker='o', s=20, color='#c0392b',
                            zorder=3, alpha=0.7)
        
        ax_continuous.set_title('Continuous Representation', fontsize=12, fontweight='bold')
        ax_continuous.set_xlabel('Sample Index', fontsize=10)
        ax_continuous.set_ylabel('Amplitude', fontsize=10)
        ax_continuous.grid(True, linestyle='--', alpha=0.5)
        ax_continuous.legend(loc='best')
        
        # Add information for periodic signals
        if is_periodic:
            # Calculate approximate frequency if possible
            if len(indices) >= 2:
                # Estimate frequency from the data
                sample_period = indices[1] - indices[0] if len(indices) > 1 else 1
                if sample_period > 0:
                    # Try to detect the period from the data
                    try:
                        from scipy.signal import find_peaks
                        peaks, _ = find_peaks(amplitudes, distance=3)
                        if len(peaks) >= 2:
                            avg_period = np.mean(np.diff(indices[peaks]))
                            freq = 1/avg_period if avg_period > 0 else 0
                            info_text = f"Periodic Signal • Est. Frequency: {freq:.3f} cycles/sample"
                            ax_discrete.text(0.98, 0.02, info_text, transform=ax_discrete.transAxes, 
                                            ha='right', va='bottom', fontsize=9,
                                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
                    except (ImportError, ValueError, TypeError, IndexError) as e:
                        # Fallback if find_peaks fails or isn't available
                        info_text = "Periodic Signal"
                        ax_discrete.text(0.98, 0.02, info_text, transform=ax_discrete.transAxes, 
                                        ha='right', va='bottom', fontsize=9,
                                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

        fig.tight_layout(rect=[0, 0.08, 1, 0.96])

        # Create window
        plot_win = tk.Toplevel(self)
        plot_win.wm_title(f"Plot: {chart_title}")
        plot_win.geometry("1200x900")

        container = tk.Frame(plot_win)
        container.pack(fill=tk.BOTH, expand=True)

        # Embed plot
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, container)
        toolbar.update()

        # Control panel
        ctrl_panel = tk.Frame(container, bg='#95a5a6', relief=tk.RAISED, bd=2)
        ctrl_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        tk.Label(ctrl_panel, text="Navigation Controls:", font=('Arial', 10, 'bold'),
                 bg='#95a5a6').pack(side=tk.LEFT, padx=10)
                 
        # Add special controls for periodic signals
        if is_periodic:
            periodic_frame = tk.Frame(container, bg='#3498db', relief=tk.RAISED, bd=2)
            periodic_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=3)
            
            tk.Label(periodic_frame, text="Periodic Signal Controls:", font=('Arial', 10, 'bold'),
                    bg='#3498db', fg='white').pack(side=tk.LEFT, padx=10, pady=3)
            
            def show_full_cycles():
                # Auto-detect cycles and adjust view
                try:
                    from scipy.signal import find_peaks
                    # Find peaks to detect cycles
                    peaks, _ = find_peaks(amplitudes, distance=3)
                    
                    if len(peaks) >= 2:
                        # Get average cycle length
                        cycle_length = np.mean(np.diff(indices[peaks]))
                        # Show multiple cycles (between 2-5)
                        cycles_to_show = min(5, max(2, len(indices) // cycle_length))
                        
                        # Adjust view to show complete cycles
                        start_idx = indices[0]
                        end_idx = start_idx + cycle_length * cycles_to_show
                        
                        for ax in [ax_discrete, ax_continuous]:
                            ax.set_xlim(start_idx, end_idx)
                        canvas.draw()
                    else:
                        messagebox.showinfo("Information", "Could not detect clear cycles in this signal.")
                except Exception as e:
                    messagebox.showinfo("Information", f"Could not auto-detect cycles: {e}")
            
            tk.Button(periodic_frame, text="🔄 Show Full Cycles", command=show_full_cycles,
                    bg='#2980b9', fg='white', font=('Arial', 9, 'bold'),
                    width=15).pack(side=tk.LEFT, padx=5)
                    
            def toggle_continuous_fit():
                # Toggle between linear and advanced interpolation
                global smooth_interp
                smooth_interp = not smooth_interp
                
                # Redraw with new interpolation
                ax_continuous.clear()
                
                if smooth_interp:
                    # Create a smoother representation with more points
                    try:
                        x_smooth = np.linspace(indices.min(), indices.max(), len(indices) * 10)
                        from scipy.interpolate import interp1d
                        f = interp1d(indices, amplitudes, kind='cubic')
                        ax_continuous.plot(x_smooth, f(x_smooth), color='#e74c3c', linestyle='-',
                                        linewidth=2, label='Smooth Interpolation', zorder=2)
                        interp_type = "Cubic"
                    except (ValueError, ImportError, TypeError):
                        try:
                            x_smooth = np.linspace(indices.min(), indices.max(), len(indices) * 10)
                            from scipy.interpolate import interp1d
                            f = interp1d(indices, amplitudes, kind='linear')
                            ax_continuous.plot(x_smooth, f(x_smooth), color='#e74c3c', linestyle='-',
                                            linewidth=2, label='Linear Interpolation', zorder=2)
                            interp_type = "Linear"
                        except:
                            ax_continuous.plot(indices, amplitudes, color='#e74c3c', linestyle='-',
                                            linewidth=2, label='Continuous Approximation', zorder=2)
                            interp_type = "Basic"
                else:
                    ax_continuous.plot(indices, amplitudes, color='#e74c3c', linestyle='-',
                                    linewidth=2, label='Continuous Approximation', zorder=2)
                    interp_type = "Basic"
                
                # Always show the actual points
                ax_continuous.scatter(indices, amplitudes, marker='o', s=20, color='#c0392b',
                                    zorder=3, alpha=0.7)
                
                ax_continuous.set_title(f'Continuous Representation ({interp_type})', fontsize=12, fontweight='bold')
                ax_continuous.set_xlabel('Sample Index', fontsize=10)
                ax_continuous.set_ylabel('Amplitude', fontsize=10)
                ax_continuous.grid(True, linestyle='--', alpha=0.5)
                ax_continuous.legend(loc='best')
                
                # Maintain current view limits
                ax_continuous.set_xlim(ax_continuous.get_xlim())
                ax_continuous.set_ylim(ax_continuous.get_ylim())
                
                canvas.draw()
                
            # Initialize smooth interpolation state
            smooth_interp = True
                
            tk.Button(periodic_frame, text="🔀 Toggle Interpolation", command=toggle_continuous_fit,
                    bg='#2980b9', fg='white', font=('Arial', 9, 'bold'),
                    width=15).pack(side=tk.LEFT, padx=5)

        # Store initial limits
        init_xlim_disc = ax_discrete.get_xlim()
        init_ylim_disc = ax_discrete.get_ylim()
        init_xlim_cont = ax_continuous.get_xlim()
        init_ylim_cont = ax_continuous.get_ylim()

        def zoom_in_view():
            for ax in [ax_discrete, ax_continuous]:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_mid = (xlim[0] + xlim[1]) / 2
                y_mid = (ylim[0] + ylim[1]) / 2
                x_span = (xlim[1] - xlim[0]) * 0.75
                y_span = (ylim[1] - ylim[0]) * 0.75
                ax.set_xlim(x_mid - x_span / 2, x_mid + x_span / 2)
                ax.set_ylim(y_mid - y_span / 2, y_mid + y_span / 2)
            canvas.draw()

        def zoom_out_view():
            for ax in [ax_discrete, ax_continuous]:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_mid = (xlim[0] + xlim[1]) / 2
                y_mid = (ylim[0] + ylim[1]) / 2
                x_span = (xlim[1] - xlim[0]) * 1.33
                y_span = (ylim[1] - ylim[0]) * 1.33
                ax.set_xlim(x_mid - x_span / 2, x_mid + x_span / 2)
                ax.set_ylim(y_mid - y_span / 2, y_mid + y_span / 2)
            canvas.draw()

        def reset_view_limits():
            ax_discrete.set_xlim(init_xlim_disc)
            ax_discrete.set_ylim(init_ylim_disc)
            ax_continuous.set_xlim(init_xlim_cont)
            ax_continuous.set_ylim(init_ylim_cont)
            canvas.draw()
            
        def optimize_view():
            # Automatically optimize the view for better visualization
            # For large signals, show a reasonable portion
            if len(indices) > 100:
                # For large signals, show first 2-3 cycles for periodic or first 50-100 points
                if is_periodic:
                    try:
                        from scipy.signal import find_peaks
                        peaks, _ = find_peaks(amplitudes, distance=3)
                        if len(peaks) >= 2:
                            cycle_length = np.mean(np.diff(indices[peaks]))
                            end_idx = indices[0] + cycle_length * 3  # Show 3 cycles
                            end_idx = min(end_idx, indices[-1])  # Don't go beyond data
                            
                            for ax in [ax_discrete, ax_continuous]:
                                ax.set_xlim(indices[0], end_idx)
                            
                            # Reset y-axis to fit this range
                            visible_indices = indices[indices <= end_idx]
                            visible_amps = amplitudes[:len(visible_indices)]
                            y_min = min(visible_amps)
                            y_max = max(visible_amps)
                            y_padding = (y_max - y_min) * 0.1
                            
                            for ax in [ax_discrete, ax_continuous]:
                                ax.set_ylim(y_min - y_padding, y_max + y_padding)
                    except:
                        # If peak detection fails, show first 100 points
                        for ax in [ax_discrete, ax_continuous]:
                            ax.set_xlim(indices[0], indices[min(100, len(indices)-1)])
                else:
                    # For non-periodic, show first 50 points
                    for ax in [ax_discrete, ax_continuous]:
                        ax.set_xlim(indices[0], indices[min(50, len(indices)-1)])
            
            canvas.draw()
        
        # Call optimize_view on initial display for large signals
        if len(indices) > 100:
            optimize_view()

        def custom_range_zoom():
            dialog = tk.Toplevel(plot_win)
            dialog.title("Custom Range")
            dialog.geometry("300x150")

            tk.Label(dialog, text="X-axis range:", font=('Arial', 10, 'bold')).pack(pady=10)

            frame = tk.Frame(dialog)
            frame.pack(pady=5)

            tk.Label(frame, text="Min:").grid(row=0, column=0, padx=5)
            min_entry = tk.Entry(frame, width=10)
            min_entry.grid(row=0, column=1, padx=5)
            min_entry.insert(0, str(int(indices[0])))

            tk.Label(frame, text="Max:").grid(row=1, column=0, padx=5)
            max_entry = tk.Entry(frame, width=10)
            max_entry.grid(row=1, column=1, padx=5)
            max_entry.insert(0, str(int(indices[-1])))

            def apply_range():
                try:
                    x_min = float(min_entry.get())
                    x_max = float(max_entry.get())
                    if x_min >= x_max:
                        messagebox.showerror("Error", "Min must be less than Max")
                        return
                    ax_discrete.set_xlim(x_min, x_max)
                    ax_continuous.set_xlim(x_min, x_max)
                    canvas.draw()
                    dialog.destroy()
                except ValueError:
                    messagebox.showerror("Error", "Invalid numbers")

            tk.Button(dialog, text="Apply", command=apply_range, bg='#3498db',
                      fg='white').pack(pady=10)

        def shift_left():
            for ax in [ax_discrete, ax_continuous]:
                xlim = ax.get_xlim()
                shift = (xlim[1] - xlim[0]) * 0.25
                ax.set_xlim(xlim[0] - shift, xlim[1] - shift)
            canvas.draw()

        def shift_right():
            for ax in [ax_discrete, ax_continuous]:
                xlim = ax.get_xlim()
                shift = (xlim[1] - xlim[0]) * 0.25
                ax.set_xlim(xlim[0] + shift, xlim[1] + shift)
            canvas.draw()

        def shift_up():
            for ax in [ax_discrete, ax_continuous]:
                ylim = ax.get_ylim()
                shift = (ylim[1] - ylim[0]) * 0.25
                ax.set_ylim(ylim[0] + shift, ylim[1] + shift)
            canvas.draw()

        def shift_down():
            for ax in [ax_discrete, ax_continuous]:
                ylim = ax.get_ylim()
                shift = (ylim[1] - ylim[0]) * 0.25
                ax.set_ylim(ylim[0] - shift, ylim[1] - shift)
            canvas.draw()

        # Navigation arrows
        nav_frame = tk.Frame(ctrl_panel, bg='#95a5a6')
        nav_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(nav_frame, text="Pan:", font=('Arial', 9, 'bold'),
                 bg='#95a5a6').grid(row=0, column=0, columnspan=3, pady=2)

        tk.Button(nav_frame, text="↑", command=shift_up, bg='#bdc3c7',
                  font=('Arial', 11, 'bold'), width=3).grid(row=1, column=1, padx=1, pady=1)
        tk.Button(nav_frame, text="←", command=shift_left, bg='#bdc3c7',
                  font=('Arial', 11, 'bold'), width=3).grid(row=2, column=0, padx=1, pady=1)
        tk.Button(nav_frame, text="→", command=shift_right, bg='#bdc3c7',
                  font=('Arial', 11, 'bold'), width=3).grid(row=2, column=2, padx=1, pady=1)
        tk.Button(nav_frame, text="↓", command=shift_down, bg='#bdc3c7',
                  font=('Arial', 11, 'bold'), width=3).grid(row=3, column=1, padx=1, pady=1)

        tk.Frame(ctrl_panel, width=2, bg='gray').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Zoom buttons
        tk.Button(ctrl_panel, text="🔍+ Zoom In", command=zoom_in_view,
                  bg='#2ecc71', fg='white', font=('Arial', 9, 'bold'),
                  width=12).pack(side=tk.LEFT, padx=5)

        tk.Button(ctrl_panel, text="🔍- Zoom Out", command=zoom_out_view,
                  bg='#e74c3c', fg='white', font=('Arial', 9, 'bold'),
                  width=12).pack(side=tk.LEFT, padx=5)

        tk.Button(ctrl_panel, text="↻ Reset", command=reset_view_limits,
                  bg='#f39c12', fg='white', font=('Arial', 9, 'bold'),
                  width=12).pack(side=tk.LEFT, padx=5)
                  
        tk.Button(ctrl_panel, text="✨ Optimize View", command=optimize_view,
                  bg='#16a085', fg='white', font=('Arial', 9, 'bold'),
                  width=12).pack(side=tk.LEFT, padx=5)

        tk.Button(ctrl_panel, text="🎯 Set Range", command=custom_range_zoom,
                  bg='#9b59b6', fg='white', font=('Arial', 9, 'bold'),
                  width=13).pack(side=tk.LEFT, padx=5)

        # Info bar
        info_text = "💡 Use toolbar for box zoom • Arrow buttons to pan • Mouse wheel to zoom"
        
        # Add more detailed info for different signal types
        if is_periodic:
            # Calculate basic signal statistics
            min_val = np.min(amplitudes)
            max_val = np.max(amplitudes)
            amplitude_val = (max_val - min_val) / 2
            
            # Add more info for sine/cosine signals
            if chart_title.lower().find('sine') >= 0 or chart_title.lower().find('cosine') >= 0:
                info_text += f" • Amplitude: {amplitude_val:.2f} • Range: [{min_val:.2f}, {max_val:.2f}]"
        else:
            # For non-periodic signals, show basic statistics
            avg_val = np.mean(amplitudes)
            std_val = np.std(amplitudes)
            info_text += f" • Mean: {avg_val:.2f} • Std Dev: {std_val:.2f}"
            
        info = tk.Label(container, text=info_text, bg='#ecf0f1', 
                        font=('Arial', 9), relief=tk.FLAT, pady=3)
        info.pack(side=tk.BOTTOM, fill=tk.X)

        # Mouse wheel zoom
        def handle_scroll(event):
            ax = event.inaxes
            if ax is None:
                return

            scale = 1.2 if event.button == 'down' else 0.8
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
            y_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * scale
            y_range = (ylim[1] - ylim[0]) * scale
            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
            canvas.draw()

        canvas.mpl_connect('scroll_event', handle_scroll)

        # Keyboard navigation
        def handle_keypress(event):
            if event.key == 'left':
                shift_left()
            elif event.key == 'right':
                shift_right()
            elif event.key == 'up':
                shift_up()
            elif event.key == 'down':
                shift_down()
            elif event.key == 'r':
                reset_view_limits()
            elif event.key in ['+', '=']:
                zoom_in_view()
            elif event.key in ['-', '_']:
                zoom_out_view()

        canvas.mpl_connect('key_press_event', handle_keypress)
        canvas_widget.focus_set()

        plot_win.protocol("WM_DELETE_WINDOW", plot_win.destroy)

    # === SIGNAL GENERATION ===
    def open_generator_window(self):
        """Open dialog for signal generation parameters"""
        win = tk.Toplevel(self)
        win.title("Signal Generator Configuration")
        win.geometry("400x420")  # Made taller to accommodate the new checkbox
        win.configure(bg='#ecf0f1')

        params = {}

        fields = [
            ("Amplitude:", 5.0, 'A'),
            ("Phase (radians):", 0.0, 'theta'),
            ("Frequency (Hz):", 1.0, 'F'),
            ("Sample Rate (Hz):", 20.0, 'Fs'),
            ("Duration (seconds):", 1.0, 'duration')
        ]

        for idx, (label_text, default, key) in enumerate(fields):
            tk.Label(win, text=label_text, bg='#ecf0f1', font=('Arial', 10)).grid(
                row=idx, column=0, padx=15, pady=8, sticky='e'
            )
            entry = tk.Entry(win, width=15, font=('Arial', 10))
            entry.insert(0, str(default))
            entry.grid(row=idx, column=1, padx=15, pady=8)
            params[key] = entry

        tk.Label(win, text="Waveform:", bg='#ecf0f1', font=('Arial', 10)).grid(
            row=len(fields), column=0, padx=15, pady=8, sticky='e'
        )
        wave_var = tk.StringVar(win)
        wave_var.set('cosine')

        wave_menu = ttk.Combobox(win, textvariable=wave_var, values=['sine', 'cosine'],
                                 state='readonly', width=13, font=('Arial', 10))
        wave_menu.grid(row=len(fields), column=1, padx=15, pady=8)
        
        # Add periodic checkbox - place it after the waveform dropdown (but before the Generate button)
        periodic_var = tk.BooleanVar(win)
        periodic_var.set(True)  # Default to True for sine/cosine waves
        
        # Create a styled frame for the checkbox
        periodic_frame = tk.Frame(win, bg='#d6eaf8', bd=1, relief=tk.GROOVE)
        periodic_frame.grid(row=len(fields)+1, column=0, columnspan=2, padx=15, pady=10, sticky='ew')
        
        periodic_check = tk.Checkbutton(
            periodic_frame, 
            text="Signal is Periodic", 
            variable=periodic_var,
            bg='#d6eaf8',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5,
            activebackground='#aed6f1'
        )
        periodic_check.pack(fill=tk.X)
        
        # Add info tooltip/label
        info_label = tk.Label(
            periodic_frame,
            text="Enable to mark signal as periodic in the output file",
            font=('Arial', 8, 'italic'),
            fg='#2980b9',
            bg='#d6eaf8',
            pady=2
        )
        info_label.pack()

        def execute_generation():
            try:
                amplitude = float(params['A'].get())
                phase = float(params['theta'].get())
                freq = float(params['F'].get())
                sample_rate = float(params['Fs'].get())
                duration = float(params['duration'].get())
                wave_type = wave_var.get()
                is_periodic = periodic_var.get()  # Get the periodic checkbox value

                if sample_rate <= 0 or freq <= 0 or duration <= 0:
                    messagebox.showerror("Invalid Input", "Frequency, sample rate, and duration must be positive.")
                    return

                if sample_rate < 2 * freq:
                    messagebox.showwarning(
                        "Nyquist Warning",
                        f"Sample rate ({sample_rate} Hz) is below Nyquist limit (2×{freq} = {2 * freq} Hz).\nAliasing may occur!"
                    )

                # Generate the signal
                result = generate_sin_cos(wave_type, amplitude, phase, freq, sample_rate, duration)

                # Set the periodic flag based on the checkbox
                if result:
                    result.is_periodic = is_periodic
                    periodicity_str = "periodic" if is_periodic else "non-periodic"
                    sig_name = f"{wave_type}_{len(signal_storage)}"
                    self.register_signal(sig_name, result)
                    self.ask_for_save(result, f"{wave_type}_{freq}Hz_{duration}s_{periodicity_str}.txt")
                    win.destroy()

            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

        tk.Button(
            win,
            text="Generate Signal",
            command=execute_generation,
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            padx=20,
            pady=10
        ).grid(row=len(fields) + 2, column=0, columnspan=2, pady=20)

    # === ARITHMETIC OPERATIONS ===
    def perform_addition(self):
        """Add two signals together"""
        name1, sig1 = self.select_signal("Select first signal:")
        if not sig1:
            return

        name2, sig2 = self.select_signal("Select second signal:")
        if not sig2:
            return

        result = add_signals(sig1, sig2)

        if result:
            result_name = f"sum_{len(signal_storage)}"
            self.register_signal(result_name, result)
            self.ask_for_save(result, f"sum_{name1}_{name2}.txt")

    def perform_subtraction(self):
        """Subtract one signal from another"""
        name1, sig1 = self.select_signal("Select minuend (signal to subtract from):")
        if not sig1:
            return

        name2, sig2 = self.select_signal("Select subtrahend (signal to subtract):")
        if not sig2:
            return

        result = subtract_signals(sig1, sig2)

        if result:
            result_name = f"diff_{len(signal_storage)}"
            self.register_signal(result_name, result)
            self.ask_for_save(result, f"difference_{name1}_{name2}.txt")

    def perform_scaling(self):
        """Multiply signal by constant factor"""
        name, sig = self.select_signal("Select signal to scale:")
        if not sig:
            return

        try:
            factor = simpledialog.askfloat("Scaling Factor", "Enter multiplication factor:")
            if factor is None:
                return
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")
            return

        result = multiply_signal(sig, factor)

        if result:
            result_name = f"scaled_{len(signal_storage)}"
            self.register_signal(result_name, result)
            self.ask_for_save(result, f"{name}_times_{factor}.txt")

    def single_signal_op(self, operation):
        """Handle single-signal operations"""
        name, sig = self.select_signal(f"Select signal for {operation} operation:")
        if not sig:
            return

        if operation == 'square':
            result = square_signal(sig)
            filename = f"{name}_squared.txt"
        elif operation == 'accumulate':
            result = accumulate_signal(sig)
            filename = f"{name}_accumulated.txt"
        else:
            return

        if result:
            result_name = f"{operation}_{len(signal_storage)}"
            self.register_signal(result_name, result)
            self.ask_for_save(result, filename)

    def open_normalize_window(self):
        """Open normalization range selection dialog"""
        name, sig = self.select_signal("Select signal to normalize:")
        if not sig:
            return
            
    def perform_quantization_bits(self):
        """Quantize a signal based on number of bits"""
        name, sig = self.select_signal("Select signal to quantize:")
        if not sig:
            return
            
        # Ask user for the number of bits
        num_bits = simpledialog.askinteger("Quantization Parameters", 
                                           "Enter the number of bits:", 
                                           minvalue=1, maxvalue=16)
        if not num_bits:  # User cancelled or entered invalid value
            return
        
        self.write_to_console(f"Quantizing signal '{name}' with {num_bits} bits...")
        
        # Perform quantization
        encoded_values, quantized_values = quantize_signal_bits(sig, num_bits)
        
        # Create result signal
        result_signal = Signal()
        sorted_indices = sorted(sig.samples.keys())
        
        for i, idx in enumerate(sorted_indices):
            result_signal.samples[idx] = [quantized_values[i]]
        
        result_signal.N1 = len(result_signal.samples)
        
        # Register result signal
        result_name = f"{name}_quantized_{num_bits}bits"
        self.register_signal(result_name, result_signal)
        
        # Display results in the console
        self.write_to_console(f"\nQuantization Results for {name} with {num_bits} bits:")
        self.write_to_console(f"{'Index':<10}{'Original':<15}{'Encoded':<15}{'Quantized':<15}")
        self.write_to_console("-" * 50)
        
        for i, idx in enumerate(sorted_indices):
            original_val = sig.samples[idx][0]
            encoded = encoded_values[i]
            quantized = quantized_values[i]
            self.write_to_console(f"{idx:<10}{original_val:<15.4f}{encoded:<15}{quantized:<15.4f}")
        
        # Ask if user wants to save results to file
        if messagebox.askyesno("Save Results", "Would you like to save the quantization results to a file?"):
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save Quantization Results"
            )
            if filepath:
                with open(filepath, 'w') as f:
                    f.write("0\n")  # SignalType: 0 for Time domain
                    f.write("0\n")  # IsPeriodic: 0 for non-periodic
                    f.write(f"{len(sorted_indices)}\n")  # Number of samples
                    
                    for i, idx in enumerate(sorted_indices):
                        f.write(f"{encoded_values[i]} {quantized_values[i]:.2f}\n")
                
                self.write_to_console(f"Results saved to {filepath}")
        
        # Show the result signal
        self.show_signal_visualization(result_signal, f"Quantized Signal ({num_bits} bits)")
        
    def perform_quantization_levels(self):
        """Quantize a signal based on number of levels"""
        name, sig = self.select_signal("Select signal to quantize:")
        if not sig:
            return
            
        # Ask user for the number of levels
        num_levels = simpledialog.askinteger("Quantization Parameters", 
                                            "Enter the number of levels:", 
                                            minvalue=2, maxvalue=256)
        if not num_levels:  # User cancelled or entered invalid value
            return
        
        self.write_to_console(f"Quantizing signal '{name}' with {num_levels} levels...")
        
        # Perform quantization
        interval_indices, encoded_values, quantized_values, sampled_errors = quantize_signal_levels(sig, num_levels)
        
        # Create result signal
        result_signal = Signal()
        sorted_indices = sorted(sig.samples.keys())
        
        for i, idx in enumerate(sorted_indices):
            result_signal.samples[idx] = [quantized_values[i]]
        
        result_signal.N1 = len(result_signal.samples)
        
        # Register result signal
        result_name = f"{name}_quantized_{num_levels}levels"
        self.register_signal(result_name, result_signal)
        
        # Display results in the console
        self.write_to_console(f"\nQuantization Results for {name} with {num_levels} levels:")
        self.write_to_console(f"{'Index':<10}{'Original':<15}{'Interval':<10}{'Encoded':<10}{'Quantized':<15}{'Error':<15}")
        self.write_to_console("-" * 70)
        
        for i, idx in enumerate(sorted_indices):
            original_val = sig.samples[idx][0]
            interval = interval_indices[i]
            encoded = encoded_values[i]
            quantized = quantized_values[i]
            error = sampled_errors[i]
            
            self.write_to_console(
                f"{idx:<10}{original_val:<15.4f}{interval:<10}{encoded:<10}{quantized:<15.4f}{error:<15.4f}"
            )
        
        # Ask if user wants to save results to file
        if messagebox.askyesno("Save Results", "Would you like to save the quantization results to a file?"):
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save Quantization Results"
            )
            if filepath:
                with open(filepath, 'w') as f:
                    f.write("0\n")  # SignalType: 0 for Time domain
                    f.write("0\n")  # IsPeriodic: 0 for non-periodic
                    f.write(f"{len(sorted_indices)}\n")  # Number of samples
                    
                    for i, idx in enumerate(sorted_indices):
                        f.write(f"{interval_indices[i]} {encoded_values[i]} {quantized_values[i]:.3f} {sampled_errors[i]:.3f}\n")
                
                self.write_to_console(f"Results saved to {filepath}")
        
        # Show the result signal
        self.show_signal_visualization(result_signal, f"Quantized Signal ({num_levels} levels)")

        win = tk.Toplevel(self)
        win.title("Normalization Settings")
        win.geometry("320x180")
        win.configure(bg='#ecf0f1')

        tk.Label(
            win,
            text="Choose normalization range:",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1'
        ).pack(pady=20)

        def apply_normalization(range_type):
            result = normalize_signal(sig, range_type)
            if result:
                result_name = f"norm_{range_type.replace('_to_', '')}_{len(signal_storage)}"
                self.register_signal(result_name, result)
                self.ask_for_save(result, f"{name}_normalized_{range_type}.txt")
            win.destroy()

        tk.Button(
            win,
            text="Normalize to [0, 1]",
            command=lambda: apply_normalization('0_to_1'),
            bg='#3498db',
            fg='white',
            font=('Arial', 10),
            width=20,
            pady=8
        ).pack(pady=8)

        tk.Button(
            win,
            text="Normalize to [-1, 1]",
            command=lambda: apply_normalization('-1_to_1'),
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10),
            width=20,
            pady=8
        ).pack(pady=8)

    # === FREQUENCY DOMAIN OPERATIONS ===
    def perform_dft(self):
        """Apply Discrete Fourier Transform to a signal"""
        name, sig = self.select_signal("Select signal for DFT:")
        if not sig:
            return
        
        # Ask for sampling frequency
        sampling_freq = simpledialog.askfloat(
            "DFT Parameters",
            "Enter the sampling frequency (Hz):",
            minvalue=0.1
        )
        if not sampling_freq:
            return
        
        self.write_to_console(f"Computing DFT for '{name}' with Fs = {sampling_freq} Hz...")
        
        # Compute DFT
        frequencies, amplitudes, phase_shifts = dft(sig, sampling_freq)
        
        # Normalize amplitudes to [0, 1]
        max_amp = np.max(amplitudes)
        if max_amp > 0:
            normalized_amplitudes = amplitudes / max_amp
        else:
            normalized_amplitudes = amplitudes
        
        # Store frequency domain data as a new signal
        freq_signal = Signal()
        freq_signal.N1 = len(frequencies)
        freq_signal.samples = {}
        
        # Store frequency domain data with index as key
        for i in range(len(frequencies)):
            # Store as [amplitude, phase] for frequency domain signals
            freq_signal.samples[i] = [amplitudes[i], phase_shifts[i]]
        
        # Mark this as a frequency domain signal
        freq_signal.is_periodic = False
        
        # Store additional metadata
        freq_signal.frequencies = frequencies
        freq_signal.amplitudes = amplitudes
        freq_signal.phase_shifts = phase_shifts
        freq_signal.normalized_amplitudes = normalized_amplitudes
        freq_signal.sampling_frequency = sampling_freq
        
        # Register the frequency domain signal
        freq_name = f"{name}_DFT"
        self.register_signal(freq_name, freq_signal)
        
        # Display results
        self.write_to_console(f"\nDFT Results for '{name}':")
        self.write_to_console(f"{'Index':<8}{'Freq (Hz)':<15}{'Amplitude':<15}{'Norm. Amp':<15}{'Phase (rad)':<15}")
        self.write_to_console("-" * 70)
        
        for i in range(len(frequencies)):
            self.write_to_console(
                f"{i:<8}{frequencies[i]:<15.3f}{amplitudes[i]:<15.3f}{normalized_amplitudes[i]:<15.3f}{phase_shifts[i]:<15.3f}"
            )
        
        # Ask if user wants to save DFT results to file
        if messagebox.askyesno("Save DFT Results", "Would you like to save the DFT results to a file?"):
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save DFT Results"
            )
            if filepath:
                write_signal_file(freq_signal, filepath, is_periodic=False)
                self.write_to_console(f"DFT results saved to {filepath}")
        
        # Visualize frequency domain
        self.show_frequency_domain_plot(frequencies, normalized_amplitudes, phase_shifts, f"DFT of {name}")
        
    def show_frequency_domain_plot(self, frequencies, amplitudes, phase_shifts, title):
        """Display frequency domain plots (amplitude and phase vs frequency)"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Amplitude vs Frequency
        ax1.stem(frequencies, amplitudes, basefmt=' ', linefmt='b-', markerfmt='bo')
        ax1.set_title('Amplitude vs Frequency', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frequency (Hz)', fontsize=10)
        ax1.set_ylabel('Normalized Amplitude', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_ylim([0, max(1.1, np.max(amplitudes) * 1.1)])
        
        # Phase vs Frequency
        ax2.stem(frequencies, phase_shifts, basefmt=' ', linefmt='r-', markerfmt='ro')
        ax2.set_title('Phase vs Frequency', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frequency (Hz)', fontsize=10)
        ax2.set_ylabel('Phase (radians)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        
        # Create window
        plot_win = tk.Toplevel(self)
        plot_win.wm_title(f"Frequency Domain: {title}")
        plot_win.geometry("1000x800")
        
        container = tk.Frame(plot_win)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Embed plot
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, container)
        toolbar.update()
    
    def show_dominant_frequencies(self):
        """Display frequencies with normalized amplitude > 0.5"""
        name, sig = self.select_signal("Select DFT signal to analyze:")
        if not sig:
            return
        
        # Check if this is a frequency domain signal
        if not hasattr(sig, 'frequencies'):
            messagebox.showerror("Error", "Please select a signal that has been transformed with DFT.")
            return
        
        # Get normalized amplitudes
        normalized_amps = sig.normalized_amplitudes
        frequencies = sig.frequencies
        
        # Find dominant frequencies (normalized amplitude > 0.5)
        dominant_indices = np.where(normalized_amps > 0.5)[0]
        
        if len(dominant_indices) == 0:
            self.write_to_console(f"No dominant frequencies found (amplitude > 0.5) in '{name}'")
            messagebox.showinfo("Dominant Frequencies", "No dominant frequencies found with amplitude > 0.5")
            return
        
        # Display results
        self.write_to_console(f"\nDominant Frequencies in '{name}' (Normalized Amplitude > 0.5):")
        self.write_to_console(f"{'Index':<8}{'Frequency (Hz)':<20}{'Normalized Amplitude':<20}")
        self.write_to_console("-" * 50)
        
        for idx in dominant_indices:
            self.write_to_console(
                f"{idx:<8}{frequencies[idx]:<20.3f}{normalized_amps[idx]:<20.3f}"
            )
        
        # Show message box with summary
        summary = f"Found {len(dominant_indices)} dominant frequency component(s):\n\n"
        for idx in dominant_indices:
            summary += f"• {frequencies[idx]:.3f} Hz (Amplitude: {normalized_amps[idx]:.3f})\n"
        
        messagebox.showinfo("Dominant Frequencies", summary)
    
    def modify_frequency_component(self):
        """Allow user to modify amplitude and phase of specific frequency components"""
        name, sig = self.select_signal("Select DFT signal to modify:")
        if not sig:
            return
        
        # Check if this is a frequency domain signal
        if not hasattr(sig, 'frequencies'):
            messagebox.showerror("Error", "Please select a signal that has been transformed with DFT.")
            return
        
        # Ask for index to modify
        max_index = len(sig.frequencies) - 1
        index = simpledialog.askinteger(
            "Modify Component",
            f"Enter the index to modify (0 to {max_index}):",
            minvalue=0,
            maxvalue=max_index
        )
        
        if index is None:
            return
        
        # Show current values
        current_amp = sig.amplitudes[index]
        current_phase = sig.phase_shifts[index]
        
        self.write_to_console(f"\nCurrent values at index {index}:")
        self.write_to_console(f"  Frequency: {sig.frequencies[index]:.3f} Hz")
        self.write_to_console(f"  Amplitude: {current_amp:.3f}")
        self.write_to_console(f"  Phase: {current_phase:.3f} rad")
        
        # Ask for new amplitude
        new_amp = simpledialog.askfloat(
            "Modify Amplitude",
            f"Enter new amplitude for index {index}\n(Current: {current_amp:.3f}):",
            initialvalue=current_amp
        )
        
        if new_amp is None:
            return
        
        # Ask for new phase
        new_phase = simpledialog.askfloat(
            "Modify Phase",
            f"Enter new phase (radians) for index {index}\n(Current: {current_phase:.3f}):",
            initialvalue=current_phase
        )
        
        if new_phase is None:
            return
        
        # Create modified signal
        modified_sig = Signal()
        modified_sig.N1 = sig.N1
        modified_sig.frequencies = sig.frequencies.copy()
        modified_sig.amplitudes = sig.amplitudes.copy()
        modified_sig.phase_shifts = sig.phase_shifts.copy()
        modified_sig.sampling_frequency = sig.sampling_frequency
        
        # Modify the specified component
        modified_sig.amplitudes[index] = new_amp
        modified_sig.phase_shifts[index] = new_phase
        
        # Recalculate normalized amplitudes
        max_amp = np.max(modified_sig.amplitudes)
        if max_amp > 0:
            modified_sig.normalized_amplitudes = modified_sig.amplitudes / max_amp
        else:
            modified_sig.normalized_amplitudes = modified_sig.amplitudes
        
        # Update samples dictionary
        modified_sig.samples = {}
        for i in range(len(modified_sig.frequencies)):
            modified_sig.samples[i] = [modified_sig.amplitudes[i], modified_sig.phase_shifts[i]]
        
        # Register modified signal
        modified_name = f"{name}_modified"
        self.register_signal(modified_name, modified_sig)
        
        self.write_to_console(f"\nModified component at index {index}:")
        self.write_to_console(f"  New Amplitude: {new_amp:.3f}")
        self.write_to_console(f"  New Phase: {new_phase:.3f} rad")
        self.write_to_console(f"Signal '{modified_name}' has been created.")
        
        # Show visualization
        self.show_frequency_domain_plot(
            modified_sig.frequencies,
            modified_sig.normalized_amplitudes,
            modified_sig.phase_shifts,
            f"Modified: {name}"
        )
    
    def perform_remove_dc(self):
        """Remove DC component from signal (works with both time and frequency domain)"""
        name, sig = self.select_signal("Select signal to remove DC component:")
        if not sig:
            return
        
        # Check if this is a frequency domain signal or time domain
        is_frequency_domain = hasattr(sig, 'frequencies')
        
        if is_frequency_domain:
            # Work with frequency domain signal directly
            self.write_to_console(f"Removing DC component from frequency domain signal '{name}'...")
            
            # Remove DC component
            modified_amplitudes, modified_phase_shifts = remove_dc_component(
                sig.amplitudes,
                sig.phase_shifts
            )
            
            # Create new signal without DC
            no_dc_sig = Signal()
            no_dc_sig.N1 = sig.N1
            no_dc_sig.frequencies = sig.frequencies.copy()
            no_dc_sig.amplitudes = modified_amplitudes
            no_dc_sig.phase_shifts = modified_phase_shifts
            no_dc_sig.sampling_frequency = sig.sampling_frequency
            
            # Recalculate normalized amplitudes
            max_amp = np.max(no_dc_sig.amplitudes)
            if max_amp > 0:
                no_dc_sig.normalized_amplitudes = no_dc_sig.amplitudes / max_amp
            else:
                no_dc_sig.normalized_amplitudes = no_dc_sig.amplitudes.copy()
            
            # Update samples dictionary
            no_dc_sig.samples = {}
            for i in range(len(no_dc_sig.frequencies)):
                no_dc_sig.samples[i] = [no_dc_sig.amplitudes[i], no_dc_sig.phase_shifts[i]]
            
            # Register new signal
            no_dc_name = f"{name}_no_DC"
            self.register_signal(no_dc_name, no_dc_sig)
            
            self.write_to_console(f"DC component removed. Original DC amplitude: {sig.amplitudes[0]:.3f}")
            self.write_to_console(f"Signal '{no_dc_name}' has been created.")
            
            # Show visualization
            self.show_frequency_domain_plot(
                no_dc_sig.frequencies,
                no_dc_sig.normalized_amplitudes,
                no_dc_sig.phase_shifts,
                f"No DC: {name}"
            )
            
        else:
            # Time domain signal - apply full DFT -> Remove DC -> IDFT pipeline
            self.write_to_console(f"Removing DC component from time domain signal '{name}'...")
            
            # Ask for sampling frequency (default 1 Hz)
            sampling_freq = simpledialog.askfloat(
                "Sampling Frequency",
                "Enter the sampling frequency (Hz):",
                initialvalue=1.0,
                minvalue=0.1
            )
            if not sampling_freq:
                return
            
            self.write_to_console(f"Step 1: Applying DFT with Fs = {sampling_freq} Hz...")
            
            # Step 1: Apply DFT
            frequencies, amplitudes, phase_shifts = dft(sig, sampling_freq)
            
            self.write_to_console(f"  DC component (F[0]) amplitude: {amplitudes[0]:.4f}")
            
            # Step 2: Remove DC component
            self.write_to_console(f"Step 2: Removing DC component...")
            modified_amplitudes, modified_phase_shifts = remove_dc_component(amplitudes, phase_shifts)
            
            self.write_to_console(f"  DC component set to: {modified_amplitudes[0]:.4f}")
            
            # Step 3: Apply IDFT to reconstruct time domain signal
            self.write_to_console(f"Step 3: Applying IDFT to reconstruct signal...")
            reconstructed_signal = idft(frequencies, modified_amplitudes, modified_phase_shifts)
            
            # Register reconstructed signal
            no_dc_name = f"{name}_no_DC"
            self.register_signal(no_dc_name, reconstructed_signal)
            
            self.write_to_console(f"✓ DC component removed successfully!")
            self.write_to_console(f"Signal '{no_dc_name}' has been created (time domain).")
            
            # Display comparison table
            self.write_to_console(f"\nComparison (first 5 samples):")
            self.write_to_console(f"{'Index':<8}{'Original':<15}{'No DC':<15}{'Difference':<15}")
            self.write_to_console("-" * 50)
            
            sorted_indices = sorted(sig.samples.keys())
            for i in range(min(5, len(sorted_indices))):
                idx = sorted_indices[i]
                original = sig.samples[idx][0]
                no_dc_val = reconstructed_signal.samples[idx][0]
                diff = original - no_dc_val
                self.write_to_console(f"{idx:<8}{original:<15.4f}{no_dc_val:<15.4f}{diff:<15.4f}")
            
            # Ask if user wants to save
            if messagebox.askyesno("Save Results", "Would you like to save the signal with DC removed?"):
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")],
                    title="Save Signal (DC Removed)"
                )
                if filepath:
                    write_signal_file(reconstructed_signal, filepath, is_periodic=False)
                    self.write_to_console(f"Signal saved to {filepath}")
            
            # Show visualization of reconstructed signal
            self.show_signal_visualization(reconstructed_signal, f"DC Removed: {no_dc_name}")
    
    def perform_idft(self):
        """Apply Inverse Discrete Fourier Transform"""
        name, sig = self.select_signal("Select DFT signal for IDFT:")
        if not sig:
            return
        
        # Check if this is a frequency domain signal
        if not hasattr(sig, 'frequencies'):
            messagebox.showerror("Error", "Please select a signal that has been transformed with DFT.")
            return
        
        self.write_to_console(f"Computing IDFT for '{name}'...")
        
        # Compute IDFT
        time_signal = idft(sig.frequencies, sig.amplitudes, sig.phase_shifts)
        
        # Register reconstructed time domain signal
        time_name = f"{name}_IDFT"
        self.register_signal(time_name, time_signal)
        
        self.write_to_console(f"IDFT completed. Signal '{time_name}' has been reconstructed.")
        
        # Ask if user wants to save IDFT results to file
        if messagebox.askyesno("Save IDFT Results", "Would you like to save the reconstructed signal to a file?"):
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save IDFT Results"
            )
            if filepath:
                write_signal_file(time_signal, filepath, is_periodic=False)
                self.write_to_console(f"IDFT results saved to {filepath}")
        
        # Display reconstructed signal
        self.show_signal_visualization(time_signal, f"IDFT: {time_name}")



if __name__ == "__main__":
    app = SignalProcessorGUI()
    app.mainloop()