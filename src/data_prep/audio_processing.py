import librosa
import numpy as np
import scipy.signal

def extract_glottal_residual(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Aplica Filtrado Inverso Glótico (GIF) mediante Linear Predictive Coding (LPC).
    Aísla la onda de excitación glótica primaria e(t) producida por las cuerdas vocales.
    """
    # Pre-énfasis para estabilizar el cálculo del LPC
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # Orden del filtro LPC (18 para audios de 16 kHz)
    lpc_order = int(2 + sr / 1000)
    
    # Coeficientes de resonancia del tracto vocal
    a = librosa.lpc(y_pre, order=lpc_order)
    
    # Inversión del filtro para extraer el residuo
    residual = scipy.signal.lfilter(a, [1.0], y_pre)
    
    # Convierte y asegura explícitamente el tipo de retorno np.ndarray 
    return np.asarray(residual, dtype=np.float32)