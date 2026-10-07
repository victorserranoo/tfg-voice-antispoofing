import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from src.features.cymatics import DigitalCymaticsExtractor

def generate_dummy_audio_files():
    """Genera dos archivos de audio sintéticos si no existen datos reales aún."""
    os.makedirs("data", exist_ok=True)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Audio "Humano" Simulado: Onda continua con armónicos suaves
    audio_real = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    audio_real += 0.01 * np.random.normal(size=t.shape)

    # Audio "IA Clonada" Simulado: Discontinuidades de fase y artefactos de trama (vocoder)
    audio_spoof = audio_real.copy()
    hop = int(sr * 0.05) # Tramas de 50ms
    for i in range(0, len(audio_spoof), hop):
        if (i // hop) % 2 == 0:
            audio_spoof[i:i+hop] *= -1.0 # Salto de fase brusco
            audio_spoof[i:i+hop] += 0.08 * np.random.uniform(-1, 1, size=min(hop, len(audio_spoof)-i))

    real_path = "data/audioPersona.wav"
    spoof_path = "data/audioGenerado.wav"
    
    sf.write(real_path, audio_real, sr)
    sf.write(spoof_path, audio_spoof, sr)
    return real_path, spoof_path

def plot_cymatics_comparison(real_path: str, spoof_path: str, output_image: str = "data/cymatics_comparison2.png"):
    extractor = DigitalCymaticsExtractor(sample_rate=16000)

    # Extraer tensores (Canal 0: CQT, Canal 1: Fase)
    tensor_real = extractor.get_cymatic_tensor(real_path, duration=2.0).numpy()
    tensor_spoof = extractor.get_cymatic_tensor(spoof_path, duration=2.0).numpy()

    cqt_real, phase_real = tensor_real[0], tensor_real[1]
    cqt_spoof, phase_spoof = tensor_spoof[0], tensor_spoof[1]

    # Crear la figura comparativa (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Análisis de Cimática Digital: Humano vs. IA Clonada", fontsize=14, fontweight='bold')

    # Row 1: CQT
    im1 = axes[0, 0].imshow(cqt_real, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title("Audio REAL (Bonafide) - CQT")
    axes[0, 0].set_ylabel("Frecuencia (Bins)")
    fig.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 0].imshow(cqt_real, aspect='auto', origin='lower', cmap='viridis')
    im2 = axes[0, 1].imshow(cqt_spoof, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title("Audio IA (Clonado / Spoof) - CQT")
    fig.colorbar(im2, ax=axes[0, 1])

    # Row 2: Espectrograma de Fase Continua
    im3 = axes[1, 0].imshow(phase_real, aspect='auto', origin='lower', cmap='magma')
    axes[1, 0].set_title("Audio REAL - Espectrograma de Fase Continua")
    axes[1, 0].set_xlabel("Tiempo (Tramas)")
    axes[1, 0].set_ylabel("Frecuencia (Bins)")
    fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(phase_spoof, aspect='auto', origin='lower', cmap='magma')
    axes[1, 1].set_title("Audio IA - Espectrograma de Fase Continua")
    axes[1, 1].set_xlabel("Tiempo (Tramas)")
    fig.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Imagen comparativa guardada con éxito en: {output_image}")

if __name__ == "__main__":
    # Si tienes audios reales de ASVspoof, pon aquí sus rutas.
    # Si no, el script creará dos de prueba automáticamente.
    real_audio = "data/audioPersona.wav"
    spoof_audio = "data/audioGenerado.wav"

    if not (os.path.exists(real_audio) and os.path.exists(spoof_audio)):
        print("No se encontraron audios locales. Generando muestras sintéticas de prueba...")
        real_audio, spoof_audio = generate_dummy_audio_files()

    plot_cymatics_comparison(real_audio, spoof_audio)