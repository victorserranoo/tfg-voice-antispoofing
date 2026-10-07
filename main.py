import torch
from src.features.cymatics import DigitalCymaticsExtractor
from src.models.vision_detector import AntiSpoofingResNet

if __name__ == "__main__":
    print("1. Inicializando extractor de características...")
    extractor = DigitalCymaticsExtractor()
    
    # Usamos uno de los audios de prueba que generamos antes
    audio_test = "data/audioPersona.wav" 
    
    try:
        # Extraer el tensor (El "ojo")
        tensor_2d = extractor.get_cymatic_tensor(audio_test, duration=2.0)
        print(f"   -> Tensor extraído con éxito. Forma: {tensor_2d.shape}")
        
        # Añadir la dimensión del "Batch" (La red espera [Batch, Canales, Alto, Ancho])
        # Al pasar un solo audio, el batch es 1.
        tensor_batch = tensor_2d.unsqueeze(0) 
        
        print("\n2. Inicializando Red Neuronal (ResNet18 modificada)...")
        model = AntiSpoofingResNet(pretrained=False) # False para probar más rápido en local
        
        # Pasar el tensor por la red neuronal (El "cerebro")
        print("   -> Realizando inferencia...")
        model.eval() # Modo evaluación
        with torch.no_grad():
            output = model(tensor_batch)
            
        print(f"\n¡Éxito! La red ha procesado el audio.")
        print(f"Salida bruta de la red (Logits): {output}")
        print(f"Forma de la salida: {output.shape} -> (1 audio, 2 clases: [Fake, Real])")
        
    except Exception as e:
        print(f"Error en la ejecución: {e}")