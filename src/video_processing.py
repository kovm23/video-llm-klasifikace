import cv2
import numpy as np
import base64
from PIL import Image
import io

def extract_frames(video_path, num_frames=10):
    """
    Načte video ze zadané cesty a extrahuje 'num_frames' rovnoměrně
    rozložených snímků.
    
    Vrací:
        list: Seznam objektů PIL.Image
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"CHYBA: Nelze otevřít video soubor: {video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            # Pokud má video méně snímků, než chceme, vezmeme je všechny
            frame_indices = np.arange(total_frames)
        else:
            # Rovnoměrný výběr snímků
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Převedeme OpenCV (BGR) na PIL (RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                print(f"VAROVÁNÍ: Nelze přečíst snímek {idx} z {video_path}")
        
        cap.release()
        
    except Exception as e:
        print(f"CHYBA při zpracování videa {video_path}: {e}")
    
    return frames


def pil_to_base64(image, format="jpeg"):
    """
    Převede objekt PIL.Image na textový řetězec base64.
    To je formát, který vyžaduje OpenAI API pro posílání obrázků.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')