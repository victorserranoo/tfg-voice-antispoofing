import librosa
import numpy as np
import scipy.signal as signal
import torch

class DigitalCymaticsExtractor:
    """
    Transforma señales de audio 1D en representaciones geométricas 2D 
    (CQT y Espectrogramas de Fase) para detección de artefactos de IA.
    """
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, n_bins=84):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_bins

    def extract_cqt(self, y: np.ndarray) -> np.ndarray:
        """
        Extrae la Transformada Constant-Q (CQT).
        Revela discontinuidades en la estructura armónica de vocoders sintéticos.
        """
        cqt = librosa.cqt(
            y, 
            sr=self.sample_rate, 
            hop_length=self.hop_length, 
            n_bins=self.n_bins
        )
        # Convertir a escala de decibelios (amplitud logarítmica)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return cqt_db

    def extract_unwrapped_phase(self, y: np.ndarray) -> np.ndarray:
        """
        Calcula el espectrograma de fase continua (Unwrapped Phase Spectrogram).
        Las IAs de clonación suelen fallar en mantener la coherencia de fase espacial.
        """
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        phase = np.angle(stft)
        # Desenrollar la fase a lo largo del eje del tiempo (fase continua)
        unwrapped_phase = np.unwrap(phase, axis=1)
        return unwrapped_phase

    def get_cymatic_tensor(self, audio_path: str, duration: float = 2.0) -> torch.Tensor:
        """
        Carga un audio, extrae CQT y Fase, y los empaqueta como un Tensor 2D de 2 canales
        (similar a una imagen RGB pero con 2 canales: [CQT, Fase]).
        """
        # Cargar audio fijando la tasa de muestreo
        y, _ = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
        
        # Normalizar duración (padding si es corto, truncar si es largo)
        target_len = int(self.sample_rate * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]

        # Extraer mapas 2D
        cqt_map = self.extract_cqt(y)
        phase_map = self.extract_unwrapped_phase(y)

        # Ajustar dimensiones de la fase para que coincidan con CQT
        # Redimensionado mediante interpolación básica para empaquetar en canales
        import cv2
        phase_resized = cv2.resize(phase_map, (cqt_map.shape[1], cqt_map.shape[0]))

        # Normalización min-max por canal a rango [0, 1]
        cqt_norm = (cqt_map - cqt_map.min()) / (cqt_map.max() - cqt_map.min() + 1e-8)
        phase_norm = (phase_resized - phase_resized.min()) / (phase_resized.max() - phase_resized.min() + 1e-8)

        # Apilar en un tensor de forma (Canales, Alto, Ancho) -> (2, Frecuencia, Tiempo)
        cymatic_image = np.stack([cqt_norm, phase_norm], axis=0)
        return torch.tensor(cymatic_image, dtype=torch.float32)