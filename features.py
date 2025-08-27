# features.py
import numpy as np
import pandas as pd
from scipy.signal import welch, get_window
from scipy.stats import skew, kurtosis
import io, csv

def sliding_windows(x, fs, win_sec=1.0, overlap=0.5):
    n = int(win_sec * fs)
    step = int(n * (1 - overlap))
    for start in range(0, len(x) - n + 1, step):
        yield start, x[start:start+n]

def _fft_topk(sig, fs, k=10):
    w = get_window("hann", len(sig), fftbins=True)
    sp = np.fft.rfft((sig - sig.mean()) * w)
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    amp = np.abs(sp)
    # игнорируем DC и слишком высокие
    amp[:2] = 0.0
    idx = np.argpartition(amp, -k)[-k:]
    top = idx[np.argsort(-amp[idx])]
    return freqs[top], amp[top]

def _psd_bands(sig, fs, bands=((0,100),(100,200),(200,300),(300,400),(400,500),(500,700))):
    f, Pxx = welch(sig, fs=fs, nperseg=min(2048, len(sig)), noverlap=0)
    out=[]
    for lo, hi in bands:
        mask = (f>=lo)&(f<hi)
        out.append(np.trapz(Pxx[mask], f[mask]) if mask.any() else 0.0)
    return np.array(out, dtype=np.float32)

def phase_features(sig, fs):
    f_top, a_top = _fft_topk(sig, fs, k=10)         # 10 частот + 10 амплитуд
    psd6 = _psd_bands(sig, fs)                      # 6 энергий
    rms = np.sqrt(np.mean(sig**2) + 1e-12)
    crest = (np.max(np.abs(sig)) / (rms + 1e-9))
    return np.concatenate([
        f_top.astype(np.float32), a_top.astype(np.float32), psd6,
        np.array([rms, crest, skew(sig), kurtosis(sig, fisher=True)], np.float32)
    ])

def extract_features_window(win3, fs):
    # win3: np.ndarray shape (n, 3)
    feats = [phase_features(win3[:,i], fs) for i in range(3)]
    return np.concatenate(feats)                    # 90 чисел = 30 на 3

def _sniff_sep(sample: bytes):
    try:
        dialect = csv.Sniffer().sniff(sample.decode('utf-8', 'ignore'), delimiters=",; \t")
        return dialect.delimiter
    except Exception:
        return ','

def read_csv_3phase(path_or_bytes):
    # читаем первые ~8 КБ для определения формата
    if isinstance(path_or_bytes, (str, bytes, io.BytesIO)):
        if isinstance(path_or_bytes, str):
            with open(path_or_bytes, 'rb') as f:
                head = f.read(8192)
        elif isinstance(path_or_bytes, bytes):
            head = path_or_bytes[:8192]
        else:
            pos = path_or_bytes.tell()
            head = path_or_bytes.read(8192); path_or_bytes.seek(pos)
    else:
        raise ValueError("unsupported input type")

    sep = _sniff_sep(head)
    # пытаемся понять, есть ли запятая как десятичная
    decimal = ',' if b',' in head and b'.' not in head.splitlines()[0] and sep != ',' else '.'

    # читаем c заголовком; если его нет — зададим
    try:
        df = pd.read_csv(path_or_bytes, sep=sep, decimal=decimal, engine='c')
    except Exception:
        df = pd.read_csv(path_or_bytes, sep=sep, decimal=decimal, header=None, engine='c')
        df.columns = ['A','B','C'][:df.shape[1]]

    # если есть столбец времени — отбросим его
    # оставим первые три числовые колонки
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[num_cols].iloc[:, :3].copy()

    if df.shape[1] < 3:
        raise ValueError(f"Нужно 3 числовых столбца, а получили {df.shape}")

    return df.values.astype(np.float32)

def decimate(sig3, factor):
    # простая децимация с предварительным антиалиасом (IIR безопаснее, но оставим FIR-хук)
    from scipy.signal import decimate as _dec
    return np.column_stack([_dec(sig3[:,i], factor, ftype='fir', zero_phase=True) for i in range(3)])
