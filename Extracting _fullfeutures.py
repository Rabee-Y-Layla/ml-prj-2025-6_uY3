import os
import warnings
import numpy as np
import pandas as pd

from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")


# -------- feature extraction --------

def extract_signal_features(x, name=""):
    f = {}
    n = len(x)

    if n == 0:
        return f

    mean_x = np.mean(x)
    std_x = np.std(x)

    f[f"{name}_mean"] = mean_x
    f[f"{name}_std"] = std_x
    f[f"{name}_var"] = np.var(x)
    f[f"{name}_max"] = np.max(x)
    f[f"{name}_min"] = np.min(x)
    f[f"{name}_median"] = np.median(x)

    rms = np.sqrt(np.mean(x ** 2))
    f[f"{name}_rms"] = rms
    f[f"{name}_peak_to_peak"] = np.max(x) - np.min(x)

    f[f"{name}_skewness"] = stats.skew(x) if n > 2 else 0
    f[f"{name}_kurtosis"] = stats.kurtosis(x) if n > 3 else 0

    q25, q75 = np.percentile(x, [25, 75])
    f[f"{name}_q25"] = q25
    f[f"{name}_q75"] = q75
    f[f"{name}_iqr"] = q75 - q25

    f[f"{name}_cv"] = std_x / mean_x if mean_x != 0 else 0
    f[f"{name}_mad"] = np.median(np.abs(x - np.median(x)))
    f[f"{name}_mean_abs_dev"] = np.mean(np.abs(x - mean_x))

    f[f"{name}_std_abs_diff"] = np.std(np.abs(np.diff(x))) if n > 1 else 0

    abs_x = np.abs(x)
    mean_abs = np.mean(abs_x)

    f[f"{name}_crest_factor"] = np.max(abs_x) / rms if rms != 0 else 0
    f[f"{name}_impulse_factor"] = np.max(abs_x) / mean_abs if mean_abs != 0 else 0
    f[f"{name}_shape_factor"] = rms / mean_abs if mean_abs != 0 else 0

    nz = abs_x[abs_x > 0]
    f[f"{name}_clearance_factor"] = (
        np.max(abs_x) / (np.mean(np.sqrt(nz)) ** 2)
        if len(nz) > 0 else 0
    )

    f[f"{name}_zcr"] = np.sum(np.diff(np.sign(x)) != 0) / n

    try:
        peaks, props = find_peaks(abs_x, height=std_x, distance=max(1, n // 100))
        f[f"{name}_peak_count"] = len(peaks)
        f[f"{name}_peak_rate"] = len(peaks) / n
        if len(peaks) > 0:
            f[f"{name}_mean_peak_height"] = np.mean(props["peak_heights"])
            f[f"{name}_std_peak_height"] = np.std(props["peak_heights"])
        else:
            f[f"{name}_mean_peak_height"] = 0
            f[f"{name}_std_peak_height"] = 0
    except:
        f[f"{name}_peak_count"] = 0
        f[f"{name}_peak_rate"] = 0
        f[f"{name}_mean_peak_height"] = 0
        f[f"{name}_std_peak_height"] = 0

    if n > 1:
        dx = np.diff(x)
        f[f"{name}_mean_roc"] = np.mean(np.abs(dx))
        f[f"{name}_std_roc"] = np.std(dx)
    else:
        f[f"{name}_mean_roc"] = 0
        f[f"{name}_std_roc"] = 0

    try:
        spec = np.abs(fft(x))
        freqs = np.fft.fftfreq(n, d=1)

        mask = freqs >= 0
        spec = spec[mask]
        freqs = freqs[mask]

        power = np.sum(spec ** 2)

        if power > 0:
            centroid = np.sum(freqs * spec) / power
            f[f"{name}_spectral_centroid"] = centroid
            f[f"{name}_spectral_bandwidth"] = np.sqrt(
                np.sum(((freqs - centroid) ** 2) * spec) / power
            )

            psd = (spec ** 2) / n
            psd = psd / np.sum(psd)
            psd = psd[psd > 0]
            f[f"{name}_spectral_entropy"] = -np.sum(psd * np.log2(psd))

            top = np.argsort(spec)[-3:][::-1]
            for i, idx in enumerate(top):
                f[f"{name}_dominant_freq_{i+1}"] = freqs[idx]
                f[f"{name}_dominant_amp_{i+1}"] = spec[idx]
        else:
            for k in ["spectral_centroid", "spectral_bandwidth", "spectral_entropy"]:
                f[f"{name}_{k}"] = 0
            for i in range(3):
                f[f"{name}_dominant_freq_{i+1}"] = 0
                f[f"{name}_dominant_amp_{i+1}"] = 0
    except:
        pass

    def autocorr(v, lag):
        if len(v) <= lag:
            return 0
        m = np.mean(v)
        num = np.sum((v[lag:] - m) * (v[:-lag] - m))
        den = np.sum((v - m) ** 2)
        return num / den if den != 0 else 0

    f[f"{name}_autocorr_lag1"] = autocorr(x, 1)
    f[f"{name}_autocorr_lag2"] = autocorr(x, 2)

    def higuchi(ts, kmax=10):
        N = len(ts)
        if N < 10:
            return 1.0
        L = []
        for k in range(1, min(kmax, N // 2) + 1):
            Lk = 0
            for m in range(k):
                idx = np.arange(m, N, k)
                if len(idx) > 1:
                    diff = np.sum(np.abs(np.diff(ts[idx])))
                    nk = len(idx) - 1
                    if nk > 0:
                        Lk += diff * (N - 1) / (nk * k)
            if Lk > 0:
                L.append(np.log(Lk / k))
        if len(L) > 1:
            xlog = np.log(1.0 / np.arange(1, len(L) + 1))
            return np.polyfit(xlog, L, 1)[0]
        return 1.0

    f[f"{name}_fractal_dim"] = higuchi(x)

    f[f"{name}_dynamic_range"] = (
        20 * np.log10(np.max(abs_x) / np.min(nz)) if len(nz) > 0 else 0
    )

    if n > 1:
        t = np.arange(n)
        slope, _, r, _, _ = stats.linregress(t, x)
        f[f"{name}_trend_slope"] = slope
        f[f"{name}_trend_r2"] = r ** 2
        f[f"{name}_complexity_var_diff"] = np.var(np.diff(x))
    else:
        f[f"{name}_trend_slope"] = 0
        f[f"{name}_trend_r2"] = 0
        f[f"{name}_complexity_var_diff"] = 0

    try:
        ac_vals = [autocorr(x, i) for i in range(1, min(5, n // 2))]
        f[f"{name}_periodicity_var"] = np.var(ac_vals) if ac_vals else 0
    except:
        f[f"{name}_periodicity_var"] = 0

    return f


# -------- dataset processing --------

def process_csv(path, label):
    try:
        df = pd.read_csv(path)
    except:
        return None

    if df.shape[1] == 8:
        df.columns = [
            "tachometer",
            "underhang_axial",
            "underhang_radial",
            "underhang_tangential",
            "overhang_axial",
            "overhang_radial",
            "overhang_tangential",
            "microphone",
        ]

    row = {}
    for col in df.columns:
        sig = df[col].dropna().values
        if len(sig) < 10:
            continue
        row.update(extract_signal_features(sig, col))

    if row:
        row["fault_type"] = label
        row["file"] = os.path.basename(path)
        return row

    return None


# -------- main --------

if __name__ == "__main__":

    BASE_PATH = "D:/3 семестер/Машинное обучение"

    classes = [
        "normal",
        "imbalance",
        "horizontal-misalignment",
        "vertical-misalignment",
        "underhang",
        "overhang",
    ]

    data = []

    for cls in classes:
        folder = os.path.join(BASE_PATH, cls)
        if not os.path.isdir(folder):
            continue

        for root, _, files in os.walk(folder):
            for f in files:
                if not f.endswith(".csv"):
                    continue

                path = os.path.join(root, f)
                feats = process_csv(path, cls)
                if feats:
                    data.append(feats)

        print("done:", cls)

    if data:
        df = pd.DataFrame(data)
        df.to_csv("fault_features.csv", index=False)
        print("saved:", df.shape)
    else:
        print("no data found")