# Obsah souboru: src/llm_generation.py

import os
import time
import pandas as pd
from tqdm import tqdm
import openai

# Importujeme naše vlastní moduly
import src.video_processing as vp

# --- Definice Promptů (zůstávají stejné) ---

PHRASE_PROMPT = """You are an expert video analyst. Describe the main action performed by the person in this frame. 
Include key objects they interact with and the scene context. 
Formulate your description in natural, flowing sentences."""

WORDSPLIT_PROMPT_1 = """Briefly list all key objects, actions, and scene attributes visible in the frame. 
Use bullet points."""

WORDSPLIT_PROMPT_2_REFINEMENT = """Convert the following list of observations into a continuous description using affirmative language. 
Do not use negations (no 'not', 'without', 'lacks'). Instead, describe what IS present.
Observation list:
{observation_list}
"""

# --- Funkce pro volání API ---

def get_llm_description(client, frame_image, prompt_text, model="gpt-4o-mini", max_tokens=150, temp=0.0):
    """
    Zavolá OpenAI API pro jeden snímek a jeden prompt.
    Vrací textový popis.
    """
    
    # Připravíme zprávu pro API
    base64_image = vp.pil_to_base64(frame_image)
    messages_payload = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            max_tokens=max_tokens,
            temperature=temp  # 0.0 pro reprodukovatelnost (dle AY 2025, kap. 5.8)
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"CHYBA API: {e}. Čekám 5 sekund a zkusím to znovu...")
        time.sleep(5)
        # Zkusíme to ještě jednou (s opraveným obsahem)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages_payload, # <-- OPRAVENO: Použije stejný payload
                max_tokens=max_tokens,
                temperature=temp
            )
            return response.choices[0].message.content
        except Exception as e2:
            print(f"CHYBA API (opakovaná): {e2}. Vracím prázdný popis.")
            return "" # Vrátíme prázdný řetězec, abychom nezastavili celý proces


# --- Hlavní řídící funkce ---

def generate_all_descriptions(data_path, num_frames_per_video=5, rate_limit_pause=1.0):
    """
    Hlavní smyčka, která projde train_split.csv a test_split.csv,
    spojí relativní cesty s 'data_path' a progresivně ukládá výsledky.
    """
    
    # ... (Kód pro načtení API klíče zůstává stejný) ...
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or len(api_key) < 10:
             raise ValueError("OPENAI_API_KEY nenalezen v .env souboru.")
        client = openai.OpenAI(api_key=api_key)
        print("OpenAI klient úspěšně vytvořen.")
    except Exception as e:
        print(f"CHYBA: {e}. Zastavuji.")
        return

    # 2. Definice cest
    train_csv_path = os.path.join(data_path, "train_split.csv")
    test_csv_path = os.path.join(data_path, "test_split.csv")
    output_csv_path = os.path.join(data_path, "generated_descriptions.csv")

    # 3. Načtení dat k zpracování
    # ... (kód pro načtení full_df zůstává stejný) ...
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        print(f"CHYBA: {train_csv_path} nebo {test_csv_path} nenalezen. Spusťte prosím Fázi 1.")
        return
    train_df = pd.read_csv(train_csv_path)
    train_df['split'] = 'train'
    test_df = pd.read_csv(test_csv_path)
    test_df['split'] = 'test'
    full_df = pd.concat([train_df, test_df], ignore_index=True)


    # 4. Logika pro pokračování
    final_results = []
    processed_videos = set() # Použijeme relativní cesty jako klíč

    if os.path.exists(output_csv_path):
        print(f"Nalezen existující soubor: {output_csv_path}. Načítám již zpracovaná data...")
        try:
            df_existing = pd.read_csv(output_csv_path)
            df_existing['phrase_descs'] = df_existing['phrase_descs'].apply(eval)
            df_existing['wordsplit_descs'] = df_existing['wordsplit_descs'].apply(eval)
            final_results = df_existing.to_dict('records')
            processed_videos = set(df_existing['relative_path']) # <-- ZMĚNA: Klíč je 'relative_path'
            print(f"Načteno {len(processed_videos)} již zpracovaných videí.")
        except Exception as e:
            print(f"CHYBA při načítání existujícího souboru: {e}. Začínám od nuly.")
            final_results = []
            processed_videos = set()

    # 5. Hlavní smyčka zpracování
    print(f"Celkem videí v datasetu: {len(full_df)}")
    print(f"Zbývá zpracovat: {len(full_df) - len(processed_videos)}")
    
    for index, row in tqdm(full_df.iterrows(), total=full_df.shape[0], desc="Generování popisů"):
        
        # Načteme RELATIVNÍ cestu z CSV
        relative_video_path = row['relative_path'] # <-- ZMĚNA: Přejmenovali jsme sloupec
        
        if relative_video_path in processed_videos:
            continue

        # --- NOVÁ ZMĚNA ZDE ---
        # Spojíme 'data_path' (který je správný pro aktuální stroj)
        # s relativní cestou ze CSV.
        absolute_video_path = os.path.join(data_path, relative_video_path)
        # ---------------------
        
        # A. Extrahuje snímky (volá Modul 1)
        video_frames = vp.extract_frames(absolute_video_path, num_frames=num_frames_per_video)
        
        if not video_frames:
            print(f"VAROVÁNÍ: Přeskakuji video, nelze extrahovat snímky: {absolute_video_path}")
            continue

        # ... (zbytek smyčky pro generování popisů je stejný) ...
        frame_phrase_descs = []
        frame_wordsplit_descs = []
        for frame in video_frames:
            phrase_desc = get_llm_description(client, frame, PHRASE_PROMPT)
            frame_phrase_descs.append(phrase_desc)
            time.sleep(rate_limit_pause) 
            obs_list = get_llm_description(client, frame, WORDSPLIT_PROMPT_1, max_tokens=100)
            time.sleep(rate_limit_pause) 
            refined_desc = get_llm_description(
                client, 
                frame, 
                WORDSPLIT_PROMPT_2_REFINEMENT.format(observation_list=obs_list),
                max_tokens=150
            )
            frame_wordsplit_descs.append(refined_desc)
            time.sleep(rate_limit_pause)


        # C. Připraví data pro uložení
        video_result = {
            "relative_path": relative_video_path, # <-- ZMĚNA: Uložíme relativní cestu
            "class_name": row['class_name'],
            "group_id": row['group_id'],
            "split": row['split'],
            "phrase_descs": frame_phrase_descs,
            "wordsplit_descs": frame_wordsplit_descs
        }
        
        final_results.append(video_result)
        
        # D. Progresivní ukládání
        try:
            pd.DataFrame(final_results).to_csv(output_csv_path, index=False)
        except Exception as e:
            print(f"Kritická CHYBA při ukládání do CSV: {e}")

    print("\n--- Generování všech popisů dokončeno ---")
    print(f"Výsledky uloženy do: {output_csv_path}")