import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Asegurar la raíz del proyecto en sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_prep.dataset_loader import ASVspoofCymaticDataset
from src.models.vision_detector import AntiSpoofingResNet

def compute_min_dcf(labels, bonafide_scores, p_spf=0.05, c_miss=1.0, c_fa=10.0):
    """
    Calcula el min DCF oficial para ASVspoof 5 Track 1.
    p_spf=0.05, c_miss=1, c_fa=10 -> beta ~ 1.90
    """
    fpr, tpr, thresholds = roc_curve(labels, bonafide_scores, pos_label=0) # pos_label=0 es bonafide
    fnr = 1.0 - tpr  # P_miss (falsos rechazos de bonafide)
    
    beta = (c_miss / c_fa) * ((1.0 - p_spf) / p_spf)
    dcf = beta * fnr + fpr
    
    # Normalización predeterminada de ASVspoof
    dcf_norm = dcf / min(beta, 1.0)
    return np.min(dcf_norm)

def create_asvspoof5_manifest(protocol_path: str, output_csv_path: str) -> str:
    """Parsea el archivo de metadatos oficial de ASVspoof 5."""
    data = []
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[1] if parts[1].endswith('.flac') else f"{parts[1]}.flac"
                label_str = parts[-1].lower()
                if label_str in ['spoof', 'bonafide']:
                    label = 1 if label_str == 'spoof' else 0
                    data.append([filename, label])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df = pd.DataFrame(data, columns=["filename", "label"])
    df.to_csv(output_csv_path, index=False)
    print(f"Manifiesto ASVspoof 5 generado con {len(df)} entradas.")
    return output_csv_path

def evaluate_asvspoof5(manifest_path: str, audio_dir: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Evaluando ASVspoof 5 (Track 1) en dispositivo: {device}")

    dataset = ASVspoofCymaticDataset(manifest_path=manifest_path, audio_dir=audio_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    model = AntiSpoofingResNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    bonafide_scores = []

    print("\nInferencia sobre ASVspoof 5 con capa GIF")
    with torch.no_grad():
        for i, (tensors, labels) in enumerate(loader):
            tensors = tensors.to(device)
            outputs = model(tensors)
            
            # Convención ASVspoof 5: Score más alto indica Bona Fide (Clase 0)
            probs = F.softmax(outputs, dim=1)
            p_bonafide = probs[:, 0].cpu().numpy()
            
            all_labels.extend(labels.numpy())
            bonafide_scores.extend(p_bonafide)
            
            if (i + 1) % 100 == 0:
                print(f"Procesados {i+1}/{len(loader)} lotes...")

    # 1. Cálculo de EER
    fpr, tpr, _ = roc_curve(all_labels, bonafide_scores, pos_label=0)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # 2. Cálculo de min DCF oficial (ASVspoof 5)
    min_dcf = compute_min_dcf(np.array(all_labels), np.array(bonafide_scores))

    print(f"\n==========================================")
    print(f"RESULTADOS ASVSPOOF 5 TRACK 1 (MODELO V2 GIF):")
    print(f"   • Equal Error Rate (EER):  {eer * 100:.4f}%")
    print(f"   • Minimum DCF (min DCF):   {min_dcf:.4f}")
    print(f"==========================================")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    protocol_file = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input/asvspoof5-bucket/ASVspoof5.dev.metadata.txt"
    audio_dir = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/input/asvspoof5-bucket/flac"
    checkpoint = "/kaggle/working/models_checkpoints/antispoofing_resnet_v2_glottal.pth"
    
    manifest_file = os.path.join(project_root, "data/asvspoof5_manifest.csv")
    create_asvspoof5_manifest(protocol_file, manifest_file)
    evaluate_asvspoof5(manifest_file, audio_dir, checkpoint)