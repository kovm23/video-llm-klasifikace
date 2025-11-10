import os
import time
import pandas as pd
from tqdm import tqdm
from google.colab import userdata
import openai

import src.video_processing as vp

# --- Definice Promptů  ---

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
    try:
        # Převedeme PIL obrázek na base64
        base64_image = vp.pil_to_base64(frame_image)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
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
            ],
            max_tokens=max_tokens,
            temperature=temp  # 0.0 pro reprodukovatelnost (dle AY 2025, kap. 5.8)
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"CHYBA API: {e}. Čekám 5 sekund a zkusím to znovu...")
        time.sleep(5)
        # Zkusíme to ještě jednou
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[...], # (Stejný obsah jako výše)
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
    extrahuje snímky, zavolá LLM pro oba režimy (Phrase a Word-split)
    a progresivně ukládá výsledky.
    """
    
    # 1. Načtení API klíče ze Secrets
    try:
        api_key = userdata.get('APIKEY')
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        print("CHYBA: Nelze načíst 'OPENAI_API_KEY' z Colab Secrets. Zastavuji.")
        print("Ujistěte se, že jste klíč správně přidal a zapnul 'Notebook access'.")
        return

    # 2. Definice cest
    train_csv_path = os.path.join(data_path, "train_split.csv")
    test_csv_path = os.path.join(data_path, "test_split.csv")
    output_csv_path = os.path.join(data_path, "generated_descriptions.csv")

    # 3. Načtení dat k zpracování
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        print(f"CHYBA: {train_csv_path} nebo {test_csv_path} nenalezen. Spusťte prosím Fázi 1.")
        return

    train_df = pd.read_csv(train_csv_path)
    train_df['split'] = 'train'
    test_df = pd.read_csv(test_csv_path)
    test_df['split'] = 'test'
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 4. Logika pro pokračování (aby se nepřepisovala data)
    final_results = []
    processed_videos = set()

    if os.path.exists(output_csv_path):
        print(f"Nalezen existující soubor: {output_csv_path}. Načítám již zpracovaná data...")
        try:
            df_existing = pd.read_csv(output_csv_path)
            # Ukládáme výsledky jako seznam řetězců, musíme je převést zpět na seznam
            df_existing['phrase_descs'] = df_existing['phrase_descs'].apply(eval)
            df_existing['wordsplit_descs'] = df_existing['wordsplit_descs'].apply(eval)
            final_results = df_existing.to_dict('records')
            processed_videos = set(df_existing['video_path'])
            print(f"Načteno {len(processed_videos)} již zpracovaných videí.")
        except Exception as e:
            print(f"CHYBA při načítání existujícího souboru (možná je poškozený): {e}. Začínám od nuly.")
            final_results = []
            processed_videos = set()

    # 5. Hlavní smyčka zpracování
    print(f"Celkem videí ke zpracování: {len(full_df)}")
    
    # Použijeme tqdm pro hezký progress bar
    for index, row in tqdm(full_df.iterrows(), total=full_df.shape[0], desc="Generování popisů"):
        
        video_path = row['video_path']
        
        # Přeskočíme videa, která už máme v souboru
        if video_path in processed_videos:
            continue

        # A. Extrahuje snímky (volá Modul 1)
        video_frames = vp.extract_frames(video_path, num_frames=num_frames_per_video)
        
        if not video_frames:
            print(f"VAROVÁNÍ: Přeskakuji video, nelze extrahovat snímky: {video_path}")
            continue

        frame_phrase_descs = []
        frame_wordsplit_descs = []

        # B. Projde každý snímek a zavolá API
        for frame in video_frames:
            
            # --- Režim Phrase ---
            phrase_desc = get_llm_description(client, frame, PHRASE_PROMPT)
            frame_phrase_descs.append(phrase_desc)
            
            # Pauza kvůli rate limitu
            time.sleep(rate_limit_pause) 

            # --- Režim Word-split (2 kroky) ---
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
            "video_path": video_path,
            "class_name": row['class_name'],
            "group_id": row['group_id'],
            "split": row['split'],
            "phrase_descs": frame_phrase_descs,       # Uloží seznam popisů
            "wordsplit_descs": frame_wordsplit_descs  # Uloží seznam popisů
        }
        
        final_results.append(video_result)
        
        # D. Progresivní ukládání po KAŽDÉM videu
        # To je záchrana, pokud Colab spadne
        try:
            pd.DataFrame(final_results).to_csv(output_csv_path, index=False)
        except Exception as e:
            print(f"Kritická CHYBA při ukládání do CSV: {e}")

    print("\n--- Generování všech popisů dokončeno ---")
    print(f"Výsledky uloženy do: {output_csv_path}")