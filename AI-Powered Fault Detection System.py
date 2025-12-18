import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks

# ================ Full Feature Extraction Function ================
def extract_comprehensive_time_series_features(signal, signal_name=""):
    """
    Extract comprehensive statistical, frequency, and non-linear features from time series data
    """
    features = {}
    n = len(signal)
    
    # Basic Statistical Features
    features[f'{signal_name}_mean'] = np.mean(signal)
    features[f'{signal_name}_std'] = np.std(signal)
    features[f'{signal_name}_var'] = np.var(signal)
    features[f'{signal_name}_max'] = np.max(signal)
    features[f'{signal_name}_min'] = np.min(signal)
    features[f'{signal_name}_median'] = np.median(signal)
    features[f'{signal_name}_rms'] = np.sqrt(np.mean(signal**2))
    features[f'{signal_name}_peak_to_peak'] = np.max(signal) - np.min(signal)
    features[f'{signal_name}_skewness'] = stats.skew(signal) if n > 2 else 0
    features[f'{signal_name}_kurtosis'] = stats.kurtosis(signal) if n > 3 else 0
    features[f'{signal_name}_q25'] = np.percentile(signal, 25)
    features[f'{signal_name}_q75'] = np.percentile(signal, 75)
    features[f'{signal_name}_iqr'] = features[f'{signal_name}_q75'] - features[f'{signal_name}_q25']
    
    # Advanced Statistical Features
    signal_mean = np.mean(signal)
    features[f'{signal_name}_cv'] = np.std(signal) / signal_mean if signal_mean != 0 else 0
    features[f'{signal_name}_mad'] = np.median(np.abs(signal - np.median(signal)))
    features[f'{signal_name}_mean_abs_dev'] = np.mean(np.abs(signal - signal_mean))
    
    if n > 1:
        features[f'{signal_name}_std_abs_diff'] = np.std(np.abs(np.diff(signal)))
    else:
        features[f'{signal_name}_std_abs_diff'] = 0
    
    # Shape Indicators
    abs_signal = np.abs(signal)
    rms = features[f'{signal_name}_rms']
    signal_mean_abs = np.mean(abs_signal)
    
    features[f'{signal_name}_crest_factor'] = np.max(abs_signal) / rms if rms != 0 else 0
    features[f'{signal_name}_impulse_factor'] = np.max(abs_signal) / signal_mean_abs if signal_mean_abs != 0 else 0
    features[f'{signal_name}_shape_factor'] = rms / signal_mean_abs if signal_mean_abs != 0 else 0
    
    # Clearance Factor (Lucas Factor)
    sqrt_abs = np.sqrt(abs_signal[abs_signal > 0])
    if len(sqrt_abs) > 0:
        features[f'{signal_name}_clearance_factor'] = np.max(abs_signal) / (np.mean(sqrt_abs)**2)
    else:
        features[f'{signal_name}_clearance_factor'] = 0
    
    # Regularity Measures
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    features[f'{signal_name}_zcr'] = zero_crossings / n if n > 0 else 0
    
    # Peak Analysis
    try:
        peaks, properties = find_peaks(abs_signal, height=np.std(signal), distance=max(1, n//100))
        features[f'{signal_name}_peak_count'] = len(peaks)
        features[f'{signal_name}_peak_rate'] = len(peaks) / n if n > 0 else 0
        if len(peaks) > 0:
            features[f'{signal_name}_mean_peak_height'] = np.mean(properties['peak_heights'])
            features[f'{signal_name}_std_peak_height'] = np.std(properties['peak_heights'])
        else:
            features[f'{signal_name}_mean_peak_height'] = 0
            features[f'{signal_name}_std_peak_height'] = 0
    except:
        features[f'{signal_name}_peak_count'] = 0
        features[f'{signal_name}_peak_rate'] = 0
        features[f'{signal_name}_mean_peak_height'] = 0
        features[f'{signal_name}_std_peak_height'] = 0
    
    # Rate of Change Measures
    if n > 1:
        diff_signal = np.diff(signal)
        features[f'{signal_name}_mean_roc'] = np.mean(np.abs(diff_signal))
        features[f'{signal_name}_std_roc'] = np.std(diff_signal)
    else:
        features[f'{signal_name}_mean_roc'] = 0
        features[f'{signal_name}_std_roc'] = 0
    
    # Frequency Domain Features
    try:
        fft_values = np.abs(fft(signal))
        frequencies = np.fft.fftfreq(n, d=1)
        positive_freq_mask = frequencies >= 0
        fft_pos = fft_values[positive_freq_mask]
        freq_pos = frequencies[positive_freq_mask]
        total_power = np.sum(fft_pos**2)
        
        if total_power > 0 and len(freq_pos) > 0:
            features[f'{signal_name}_spectral_centroid'] = np.sum(freq_pos * fft_pos) / total_power
            centroid = features[f'{signal_name}_spectral_centroid']
            features[f'{signal_name}_spectral_bandwidth'] = np.sqrt(
                np.sum(((freq_pos - centroid)**2) * fft_pos) / total_power
            )
            
            psd = fft_pos**2 / n
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]
            features[f'{signal_name}_spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
            
            if len(fft_pos) >= 3:
                top_freq_indices = np.argsort(fft_pos)[-3:][::-1]
                for i, idx in enumerate(top_freq_indices):
                    features[f'{signal_name}_dominant_freq_{i+1}'] = freq_pos[idx]
                    features[f'{signal_name}_dominant_amp_{i+1}'] = fft_pos[idx]
            else:
                for i in range(3):
                    features[f'{signal_name}_dominant_freq_{i+1}'] = 0
                    features[f'{signal_name}_dominant_amp_{i+1}'] = 0
        else:
            features[f'{signal_name}_spectral_centroid'] = 0
            features[f'{signal_name}_spectral_bandwidth'] = 0
            features[f'{signal_name}_spectral_entropy'] = 0
            for i in range(3):
                features[f'{signal_name}_dominant_freq_{i+1}'] = 0
                features[f'{signal_name}_dominant_amp_{i+1}'] = 0
                
    except Exception as e:
        print(f"Warning: Error in frequency analysis: {e}")
        features[f'{signal_name}_spectral_centroid'] = np.nan
        features[f'{signal_name}_spectral_bandwidth'] = np.nan
        features[f'{signal_name}_spectral_entropy'] = np.nan
        for i in range(3):
            features[f'{signal_name}_dominant_freq_{i+1}'] = np.nan
            features[f'{signal_name}_dominant_amp_{i+1}'] = np.nan
    
    # Simple Autocorrelation
    try:
        def simple_autocorr(x, lag=1):
            if len(x) <= lag:
                return 0
            x_mean = np.mean(x)
            numerator = np.sum((x[lag:] - x_mean) * (x[:-lag] - x_mean))
            denominator = np.sum((x - x_mean)**2)
            return numerator / denominator if denominator != 0 else 0
        
        features[f'{signal_name}_autocorr_lag1'] = simple_autocorr(signal, lag=1)
        features[f'{signal_name}_autocorr_lag2'] = simple_autocorr(signal, lag=2)
    except:
        features[f'{signal_name}_autocorr_lag1'] = np.nan
        features[f'{signal_name}_autocorr_lag2'] = np.nan
    
    # Non-linear Features
    def higuchi_fractal_dimension_simple(ts, k_max=10):
        n = len(ts)
        if n < 10:
            return 1.0
        
        L = []
        for k in range(1, min(k_max, n//2) + 1):
            Lk = 0
            for m in range(k):
                idx = np.arange(m, n, k)
                if len(idx) > 1:
                    Lkm = np.sum(np.abs(np.diff(ts[idx])))
                    nk = len(idx) - 1
                    if nk > 0:
                        Lkm = Lkm * (n - 1) / (nk * k)
                        Lk += Lkm
            if Lk > 0:
                L.append(np.log(Lk / k))
            else:
                L.append(0)
        
        if len(L) > 1:
            x = np.log(1.0 / np.arange(1, len(L) + 1))
            y = np.array(L)
            if np.all(y == 0):
                return 1.0
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 1.0
    
    try:
        features[f'{signal_name}_fractal_dim'] = higuchi_fractal_dimension_simple(signal)
    except:
        features[f'{signal_name}_fractal_dim'] = np.nan
    
    # Signal Quality Features
    non_zero_vals = abs_signal[abs_signal > 0]
    if len(non_zero_vals) > 0:
        min_val = np.min(non_zero_vals)
        max_val = np.max(abs_signal)
        features[f'{signal_name}_dynamic_range'] = 20 * np.log10(max_val / min_val) if min_val > 0 else 0
    else:
        features[f'{signal_name}_dynamic_range'] = 0
    
    # Trend Features
    try:
        if n > 1:
            time = np.arange(n)
            slope, intercept, r_value, p_value, std_err = stats.linregress(time, signal)
            features[f'{signal_name}_trend_slope'] = slope
            features[f'{signal_name}_trend_r2'] = r_value**2
        else:
            features[f'{signal_name}_trend_slope'] = 0
            features[f'{signal_name}_trend_r2'] = 0
    except:
        features[f'{signal_name}_trend_slope'] = np.nan
        features[f'{signal_name}_trend_r2'] = np.nan
    
    # Complexity Measures
    if n > 1:
        features[f'{signal_name}_complexity_var_diff'] = np.var(np.diff(signal))
    else:
        features[f'{signal_name}_complexity_var_diff'] = 0
    
    return features

class SimpleFaultDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fault Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load the model
        self.load_model()
        
        # Setup UI
        self.setup_ui()
        
        # Initialize variables
        self.current_file_path = None
        self.raw_data = None
        self.extracted_features = None
        
    def load_model(self):
        """Load the saved model"""
        try:
            # Look for model files
            model_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'fault_detection' in f.lower()]
            
            if not model_files:
                raise FileNotFoundError("Model file not found")
            
            model_path = model_files[0]
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder_fault']
            self.feature_names = model_data['feature_names']
            
            print(f"✅ Model loaded: {model_path}")
            print(f"   Model type: {type(self.model).__name__}")
            print(f"   Expected features: {len(self.feature_names)}")
            print(f"   Number of classes: {len(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            messagebox.showerror("Error", f"Cannot load model: {str(e)}")
            self.root.destroy()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#ffffff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🔬 AI-Powered Fault Detection System",
            font=('Arial', 20, 'bold'),
            bg='#ffffff',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # File loading section
        self.setup_file_section(main_frame)
        
        # Results section
        self.setup_results_section(main_frame)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select data file")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#34495e',
            fg='white'
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_file_section(self, parent):
        """Setup file loading section"""
        file_frame = tk.LabelFrame(
            parent,
            text="📁 Load Data File",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#2c3e50'
        )
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Path row
        path_frame = tk.Frame(file_frame, bg='#ffffff')
        path_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            path_frame,
            text="CSV File Path:",
            font=('Arial', 10),
            bg='#ffffff'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        path_entry = tk.Entry(
            path_frame,
            textvariable=self.file_path_var,
            width=50,
            font=('Arial', 10)
        )
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Buttons
        button_frame = tk.Frame(file_frame, bg='#ffffff')
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Button(
            button_frame,
            text="Browse",
            command=self.browse_file,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=10
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.process_btn = tk.Button(
            button_frame,
            text="🚀 Analyze Data & Detect Fault",
            command=self.process_file,
            bg='#2ecc71',
            fg='white',
            font=('Arial', 11, 'bold'),
            state=tk.DISABLED,
            height=2
        )
        self.process_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Data preview
        preview_frame = tk.LabelFrame(
            file_frame,
            text="Data Preview",
            font=('Arial', 10),
            bg='#ffffff'
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.preview_text = scrolledtext.ScrolledText(
            preview_frame,
            height=8,
            font=('Consolas', 9),
            wrap=tk.NONE
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_results_section(self, parent):
        """Setup results section"""
        results_frame = tk.LabelFrame(
            parent,
            text="🎯 Analysis Results",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#2c3e50'
        )
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Main result
        result_top_frame = tk.Frame(results_frame, bg='#ffffff')
        result_top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            result_top_frame,
            text="Diagnosis:",
            font=('Arial', 14, 'bold'),
            bg='#ffffff'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.result_label = tk.Label(
            result_top_frame,
            text="No analysis performed yet",
            font=('Arial', 16, 'bold'),
            bg='#f8f9fa',
            fg='#7f8c8d',
            relief=tk.RIDGE,
            width=40,
            anchor='w',
            padx=10
        )
        self.result_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Confidence
        confidence_frame = tk.Frame(results_frame, bg='#ffffff')
        confidence_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Label(
            confidence_frame,
            text="Confidence:",
            font=('Arial', 12),
            bg='#ffffff'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.confidence_var = tk.StringVar(value="0%")
        confidence_label = tk.Label(
            confidence_frame,
            textvariable=self.confidence_var,
            font=('Arial', 14, 'bold'),
            bg='#ffffff',
            fg='#27ae60'
        )
        confidence_label.pack(side=tk.LEFT)
        
        # Probabilities
        prob_frame = tk.Frame(results_frame, bg='#ffffff')
        prob_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        tk.Label(
            prob_frame,
            text="Fault Probabilities:",
            font=('Arial', 12, 'bold'),
            bg='#ffffff'
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Create Treeview for probabilities
        columns = ('Rank', 'Fault Type', 'Probability', 'Percentage')
        self.prob_tree = ttk.Treeview(
            prob_frame,
            columns=columns,
            show='headings',
            height=10
        )
        
        # Set headings
        self.prob_tree.heading('Rank', text='Rank')
        self.prob_tree.heading('Fault Type', text='Fault Type')
        self.prob_tree.heading('Probability', text='Probability')
        self.prob_tree.heading('Percentage', text='Percentage %')
        
        # Set column widths
        self.prob_tree.column('Rank', width=80, anchor='center')
        self.prob_tree.column('Fault Type', width=250, anchor='w')
        self.prob_tree.column('Probability', width=120, anchor='center')
        self.prob_tree.column('Percentage', width=100, anchor='center')
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(prob_frame, orient=tk.VERTICAL, command=self.prob_tree.yview)
        self.prob_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.prob_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Additional buttons
        button_frame = tk.Frame(results_frame, bg='#ffffff')
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(
            button_frame,
            text="💾 Save Results",
            command=self.save_results,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            button_frame,
            text="🔄 Analyze New File",
            command=self.reset_analysis,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side=tk.LEFT)
        
        tk.Button(
            button_frame,
            text="❓ Help",
            command=self.show_help,
            bg='#f39c12',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side=tk.RIGHT)
    
    def browse_file(self):
        """Open file dialog to select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set(f"File selected: {os.path.basename(file_path)}")
            self.load_file_preview(file_path)
            self.process_btn.config(state=tk.NORMAL)
    
    def load_file_preview(self, file_path):
        """Load file preview"""
        try:
            # Read first 100 rows for preview
            df = pd.read_csv(file_path, nrows=100)
            
            self.preview_text.delete(1.0, tk.END)
            
            # Show basic info
            info_text = f"📊 File Information:\n"
            info_text += f"File Name: {os.path.basename(file_path)}\n"
            info_text += f"Rows: {len(df)}\n"
            info_text += f"Columns: {len(df.columns)}\n"
            info_text += f"Columns: {', '.join(df.columns.tolist())}\n"
            info_text += "="*50 + "\n\n"
            info_text += "Data Sample:\n"
            
            self.preview_text.insert(tk.END, info_text)
            
            # Show first 5 rows
            preview_df = df.head(5)
            self.preview_text.insert(tk.END, preview_df.to_string())
            
        except Exception as e:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, f"Error reading file: {str(e)}")
    
    def process_file(self):
        """Process the file and analyze data"""
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file first!")
            return
        
        try:
            self.status_var.set("Loading data...")
            self.root.update()
            
            # Load complete data
            df = pd.read_csv(file_path)
            
            # Check data structure
            if df.shape[1] == 8:
                expected_columns = [
                    'tachometer', 'underhang_axial', 'underhang_radial',
                    'underhang_tangential', 'overhang_axial', 'overhang_radial',
                    'overhang_tangential', 'microphone'
                ]
                df.columns = expected_columns
            
            self.raw_data = df
            self.current_file_path = file_path
            
            # Extract features
            self.status_var.set("Extracting features...")
            self.root.update()
            
            self.extracted_features = self.extract_all_features(df)
            
            # Make prediction
            self.status_var.set("Analyzing data and predicting...")
            self.root.update()
            
            self.make_prediction()
            
            self.status_var.set("Analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file: {str(e)}")
            self.status_var.set("Processing error")
    
    def extract_all_features(self, df):
        """Extract all features from all sensors"""
        all_features = {}
        
        for column in df.columns:
            signal = df[column].values
            signal = signal[~np.isnan(signal)]  # Remove missing values
            
            if len(signal) < 10:
                print(f"⚠️ Signal too short in {column}: {len(signal)} samples")
                continue
            
            features = extract_comprehensive_time_series_features(signal, column)
            all_features.update(features)
        
        return pd.DataFrame([all_features])
    
    def make_prediction(self):
        """Make prediction using the model"""
        if self.extracted_features is None:
            return
        
        try:
            # Prepare features for model
            prepared_features = self.prepare_features_for_model()
            
            if prepared_features is None:
                return
            
            # Normalize features
            features_scaled = self.scaler.transform(prepared_features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            prediction_label = self.label_encoder.inverse_transform(prediction)[0]
            
            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                self.display_results(prediction_label, probabilities)
            else:
                self.display_simple_result(prediction_label)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def prepare_features_for_model(self):
        """Prepare features for model input"""
        # Get available features
        available_features = [f for f in self.feature_names 
                            if f in self.extracted_features.columns]
        
        if len(available_features) < len(self.feature_names):
            missing = len(self.feature_names) - len(available_features)
            print(f"⚠️ Missing features: {missing}")
            
            # Can add warning here if needed
        
        # Create DataFrame with same feature order
        prepared_df = pd.DataFrame(columns=self.feature_names)
        
        # Fill available features
        for feature in available_features:
            prepared_df[feature] = self.extracted_features[feature]
        
        # Fill missing features with 0
        prepared_df = prepared_df.fillna(0)
        
        return prepared_df
    
    def display_results(self, prediction, probabilities):
        """Display results"""
        # Update main result
        self.result_label.config(
            text=prediction,
            fg='#27ae60' if prediction.lower() == 'normal' else '#e74c3c'
        )
        
        # Update confidence
        top_prob = np.max(probabilities) * 100
        self.confidence_var.set(f"{top_prob:.1f}%")
        
        # Update probabilities table
        self.update_probabilities_table(probabilities, prediction)
    
    def update_probabilities_table(self, probabilities, top_prediction):
        """Update probabilities table"""
        # Clear current table
        for item in self.prob_tree.get_children():
            self.prob_tree.delete(item)
        
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Sort probabilities descending
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Add rows to table
        for rank, idx in enumerate(sorted_indices, 1):
            fault_type = class_names[idx]
            prob = probabilities[idx]
            percentage = prob * 100
            
            # Determine color based on rank
            if fault_type == top_prediction:
                tag = 'top'
            elif rank <= 3:
                tag = 'high'
            else:
                tag = ''
            
            self.prob_tree.insert('', 'end', values=(
                rank,
                fault_type,
                f"{prob:.6f}",
                f"{percentage:.2f}%"
            ), tags=(tag,))
        
        # Format rows
        self.prob_tree.tag_configure('top', background='#2ecc71', foreground='white')
        self.prob_tree.tag_configure('high', background='#f1c40f')
    
    def display_simple_result(self, prediction):
        """Display simple result (when no probabilities)"""
        self.result_label.config(
            text=prediction,
            fg='#27ae60' if prediction.lower() == 'normal' else '#e74c3c'
        )
        self.confidence_var.set("N/A")
        
        # Clear table
        for item in self.prob_tree.get_children():
            self.prob_tree.delete(item)
        
        # Add single result
        self.prob_tree.insert('', 'end', values=(
            '1', prediction, 'N/A', 'N/A'
        ), tags=('top',))
        
        self.prob_tree.tag_configure('top', background='#2ecc71', foreground='white')
    
    def save_results(self):
        """Save results"""
        if not hasattr(self, 'prediction_result'):
            messagebox.showwarning("Warning", "No results to save!")
            return
        
        try:
            # Create DataFrame for results
            results_data = {
                'Analysis_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'File_Name': [os.path.basename(self.current_file_path)],
                'Diagnosis': [self.result_label.cget("text")],
                'Confidence': [self.confidence_var.get()],
                'Features_Used': [len(self.extracted_features.columns) if self.extracted_features is not None else 0]
            }
            
            results_df = pd.DataFrame(results_data)
            
            # Save as CSV file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if file_path:
                results_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results saved to:\n{file_path}")
                self.status_var.set("Results saved")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def reset_analysis(self):
        """Reset analysis"""
        self.file_path_var.set("")
        self.preview_text.delete(1.0, tk.END)
        self.result_label.config(
            text="No analysis performed yet",
            fg='#7f8c8d'
        )
        self.confidence_var.set("0%")
        
        # Clear probabilities table
        for item in self.prob_tree.get_children():
            self.prob_tree.delete(item)
        
        self.process_btn.config(state=tk.DISABLED)
        self.status_var.set("Ready - Select data file")
    
    def show_help(self):
        """Show help window"""
        help_text = """
📖 User Guide:

1. Select a CSV file containing sensor readings
2. Click "Analyze Data & Detect Fault" button
3. Wait for analysis to complete
4. View results:
   - Final diagnosis
   - Confidence percentage
   - Probabilities for all fault types

📝 Notes:
- CSV file should have at least 8 columns
- System supports multiple fault types detection
- Results can be saved as CSV file for later reference

For additional support, please contact the support team.
        """
        
        messagebox.showinfo("Help", help_text)

def main():
    """Main function"""
    root = tk.Tk()
    app = SimpleFaultDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()