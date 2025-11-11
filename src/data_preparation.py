import os
import shutil
from tqdm import tqdm
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit

def create_mini_dataset(base_data_path, selected_classes, full_dataset_folder="UCF101_full/UCF-101", mini_dataset_folder="UCF101_mini"):
    """
    Zkopíruje vybrané třídy z plného datasetu UCF101 do nové složky 'UCF101_mini'.
    """
    full_dataset_path = os.path.join(base_data_path, full_dataset_folder)
    mini_dataset_path = os.path.join(base_data_path, mini_dataset_folder)
    
    print(f"--- Spouštím create_mini_dataset ---")
    print(f"Zdroj: {full_dataset_path}")
    print(f"Cíl: {mini_dataset_path}")

    if not os.path.exists(mini_dataset_path):
        os.makedirs(mini_dataset_path)
        print(f"Vytvořena složka: {mini_dataset_path}")
    else:
        print(f"Složka {mini_dataset_path} již existuje.")

    copied_count = 0
    for class_name in tqdm(selected_classes, desc="Kopírování tříd"):
        source_class_path = os.path.join(full_dataset_path, class_name)
        dest_class_path = os.path.join(mini_dataset_path, class_name)
        
        if not os.path.exists(dest_class_path):
            os.makedirs(dest_class_path)
        
        if not os.path.exists(source_class_path):
            print(f"VAROVÁNÍ: Zdrojová složka nenalezena: {source_class_path}")
            continue
            
        video_files = os.listdir(source_class_path)
        for video_file in video_files:
            source_video_path = os.path.join(source_class_path, video_file)
            dest_video_path = os.path.join(dest_class_path, video_file)
            
            if not os.path.exists(dest_video_path):
                shutil.copy(source_video_path, dest_video_path)
                copied_count += 1
    
    if copied_count > 0:
        print(f"Zkopírováno {copied_count} nových souborů.")
    else:
        print("Všechny soubory již byly zkopírovány.")
    print("--- Kopírování pro mini-dataset dokončeno ---")
    return mini_dataset_path


def create_group_splits(base_data_path, mini_dataset_folder="UCF101_mini", test_size=0.2, random_state=42):
    """
    Načte videa z 'UCF101_mini', rozdělí je pomocí GroupShuffleSplit
    a uloží RELATIVNÍ CESTY do 'train_split.csv' a 'test_split.csv'.
    """
    print(f"\n--- Spouštím create_group_splits ---")
    mini_dataset_path = os.path.join(base_data_path, mini_dataset_folder)
    video_files_data = []

    print(f"Načítám soubory z: {mini_dataset_path}")

    for class_name in os.listdir(mini_dataset_path):
        class_path = os.path.join(mini_dataset_path, class_name)
        if os.path.isdir(class_path):
            for video_name in os.listdir(class_path):
                
                # Toto je plná, absolutní cesta (např. /Users/kovy/.../video.avi)
                absolute_video_path = os.path.join(class_path, video_name)
                
                # --- NOVÁ ZMĚNA ZDE ---
                # Vytvoříme relativní cestu (např. UCF101_mini/ApplyEyeMakeup/video.avi)
                # 'base_data_path' je např. /Users/kovy/.../data
                relative_video_path = os.path.relpath(absolute_video_path, base_data_path)
                # ---------------------
                
                match = re.search(r'_g(\d+)_', video_name)
                
                if match:
                    group_id = f"g{match.group(1)}"
                    video_files_data.append({
                        "relative_path": relative_video_path, # <-- ULOŽÍME RELATIVNÍ
                        "class_name": class_name,
                        "group_id": group_id
                    })
                else:
                    print(f"VAROVÁNÍ: Nelze najít group_id pro: {video_name} (Přeskakuji)")

    df_videos = pd.DataFrame(video_files_data)
    if df_videos.empty:
        print("CHYBA: Nebyla nalezena žádná videa. Ověřte cestu a formát názvů souborů.")
        return

    print(f"Nalezeno celkem {len(df_videos)} videí.")
    print("Ukázka uložených cest:")
    print(df_videos.head()) # Zkontrolujte, že cesty jsou relativní

    # Rozdělení
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split_generator = splitter.split(df_videos, groups=df_videos['group_id'])
    train_indices, test_indices = next(split_generator)

    train_df = df_videos.iloc[train_indices]
    test_df = df_videos.iloc[test_indices]

    print(f"Počet trénovacích videí: {len(train_df)}")
    print(f"Počet testovacích videí: {len(test_df)}")

    # Uložení
    train_csv_path = os.path.join(base_data_path, "train_split.csv")
    test_csv_path = os.path.join(base_data_path, "test_split.csv")

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"Trénovací data uložena do: {train_csv_path}")
    print(f"Testovací data uložena do: {test_csv_path}")
    print("--- Vytvoření splitů dokončeno ---")