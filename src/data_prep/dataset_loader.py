import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.features.cymatics import DigitalCymaticsExtractor

class ASVspoofCymaticDataset(Dataset):
    """
    Dataset de PyTorch para cargar audios y transformarlos al vuelo 
    en Tensores 2D (CQT + Fase) usando DigitalCymaticsExtractor.
    """
    def __init__(self, manifest_path: str, audio_dir: str, duration: float = 2.0):
        """
        manifest_path: Ruta a un CSV con columnas ['filename', 'label']
                       donde label: 0 (Spoof/IA), 1 (Bonafide/Humano)
        audio_dir: Carpeta contenedora de los audios (.wav o .flac)
        """
        self.audio_dir = audio_dir
        self.duration = duration
        self.extractor = DigitalCymaticsExtractor()
        
        # Cargar tabla de etiquetas
        self.df = pd.read_csv(manifest_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['filename'])
        label = int(row['label'])
        
        try:
            # Extracción del tensor 2D [2, 84, 201]
            tensor_2d = self.extractor.get_cymatic_tensor(audio_path, duration=self.duration)
        except Exception:
            # En caso de audio corrupto, devuelve un tensor de ceros para no interrumpir el entrenamiento
            tensor_2d = torch.zeros((2, 84, 201), dtype=torch.float32)

        return tensor_2d, torch.tensor(label, dtype=torch.long)