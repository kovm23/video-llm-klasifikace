import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import openai


import src.video_processing as vp
import src.llm_generation as lg
import src.feature_engineering as fe

def get_raw_data_for_video(video_path, llm_client, embed_client, num_frames=3):
    """
    Znovu spustí Fázi 2 a 3 pro JEDNO video, aby získal surová data pro perturbaci.
    """
    # 1. Extrakce snímků
    video_frames = vp.extract_frames(video_path, num_frames=num_frames)
    if not video_frames:
        print(f"XAI CHYBA: Nelze extrahovat snímky z {video_path}")
        return None
    
    # 2. Generování popisů (jako v Fázi 2)
    frame_phrase_descs = []
    frame_wordsplit_descs = []
    
    for frame in video_frames:
        phrase_desc = lg.get_llm_description(llm_client, frame, lg.PHRASE_PROMPT)
        frame_phrase_descs.append(phrase_desc)
        
        obs_list = lg.get_llm_description(llm_client, frame, lg.WORDSPLIT_PROMPT_1, max_tokens=100)
        refined_desc = lg.get_llm_description(
            llm_client, frame, 
            lg.WORDSPLIT_PROMPT_2_REFINEMENT.format(observation_list=obs_list),
            max_tokens=150
        )
        frame_wordsplit_descs.append(refined_desc)
        
    # 3. Zpracování textů (Fragmentace) (jako v Fázi 3)
    # Vytvoříme 'row' objekt, který funkce 'process_texts' očekává
    fake_row = {
        'phrase_descs': str(frame_phrase_descs),
        'wordsplit_descs': str(frame_wordsplit_descs)
    }
    phrase_frags, wordsplit_frags = fe.process_texts(fake_row)
    
    # 4. Vytvoření vnoření (Embeddings) (jako v Fázi 3)
    phrase_embeds = fe.get_text_embeddings(embed_client, phrase_frags)
    wordsplit_embeds = fe.get_text_embeddings(embed_client, wordsplit_frags)
    
    # 5. Vytvoření vnoření pro jednotlivé snímky (pro doménové příznaky)
    # (Zjednodušená verze z Fáze 3)
    frame_phrase_texts = [desc if (desc and isinstance(desc, str) and desc.strip()) else "none" for desc in frame_phrase_descs]
    frame_phrase_embeds_list = fe.get_text_embeddings(embed_client, frame_phrase_texts)
    frame_phrase_embeds = [emb for emb in frame_phrase_embeds_list if emb is not None and len(emb) > 0]

    frame_wordsplit_texts = [desc if (desc and isinstance(desc, str) and desc.strip()) else "none" for desc in frame_wordsplit_descs]
    frame_wordsplit_embeds_list = fe.get_text_embeddings(embed_client, frame_wordsplit_texts)
    frame_wordsplit_embeds = [emb for emb in frame_wordsplit_embeds_list if emb is not None and len(emb) > 0]

    return {
        "phrase_frags": phrase_frags,
        "phrase_embeds": phrase_embeds,
        "frame_phrase_embeds": frame_phrase_embeds,
        "wordsplit_frags": wordsplit_frags,
        "wordsplit_embeds": wordsplit_embeds,
        "frame_wordsplit_embeds": frame_wordsplit_embeds
    }

def run_perturbation_analysis(model, xai_data, tfidf_data, target_class_index, mode='phrase', feature_set='phrase_combined_domain'):
    """
    Provede "Leave-one-out" perturbaci pro jedno video.
    (Inspirováno AY 2025, kap. 10.4.5)
    """
    
    if mode == 'phrase':
        fragments = xai_data['phrase_frags']
        embeddings = xai_data['phrase_embeds']
        frame_embeddings = xai_data['frame_phrase_embeds']
        tfidf_scores, tfidf_vocab = tfidf_data
    else: # wordsplit
        fragments = xai_data['wordsplit_frags']
        embeddings = xai_data['wordsplit_embeds']
        frame_embeddings = xai_data['frame_wordsplit_embeds']
        tfidf_scores, tfidf_vocab = tfidf_data

    if not fragments or not embeddings:
        print("Žádné fragmenty nebo vnoření k analýze.")
        return [], 0.0

    # 1. Získáme původní predikci
    # Agregujeme všechny fragmenty
    full_features = fe.aggregate_features(embeddings, fragments, frame_embeddings, tfidf_scores, tfidf_vocab)
    
    # Vybereme správný feature set, např. 'phrase_combined_domain' -> 'combined_domain'
    vector_key = "_".join(feature_set.split('_')[1:]) 
    X_original = full_features[vector_key].reshape(1, -1)
    
    # Získáme původní pravděpodobnost pro cílovou třídu
    original_prob = model.predict_proba(X_original)[0][target_class_index]
    
    # 2. Perturbace (Leave-one-out)
    importances = []
    for i in tqdm(range(len(fragments)), desc="Provádím perturbaci", leave=False):
        
        # Vytvoříme 'n-1' verzi
        temp_frags = fragments[:i] + fragments[i+1:]
        temp_embeds = embeddings[:i] + embeddings[i+1:]
        
        # Znovu agregujeme (to je výpočetně náročné)
        perturbed_features = fe.aggregate_features(temp_embeds, temp_frags, frame_embeddings, tfidf_scores, tfidf_vocab)
        X_perturbed = perturbed_features[vector_key].reshape(1, -1)
        
        # Získáme novou pravděpodobnost
        perturbed_prob = model.predict_proba(X_perturbed)[0][target_class_index]
        
        # Důležitost = o kolik klesla pravděpodobnost
        importance = original_prob - perturbed_prob
        importances.append((fragments[i], importance))
        
    # Seřadíme od nejdůležitějšího
    importances.sort(key=lambda x: x[1], reverse=True)
    
    return importances, original_prob

def calculate_aopc(model, video_list, label_list, xai_data_list, tfidf_data, mode, feature_set, k_max=10):
    """
    Vypočítá AOPC (Area Over the Perturbation Curve) pro seznam videí.
    (Inspirováno AY 2025, kap. 10.4.5)
    """
    
    tfidf_scores, tfidf_vocab = tfidf_data
    all_perturbation_results = [] # Seznam pro [ {k: 1, drop: 0.1}, {k: 2, drop: 0.15}, ... ]

    # Projdeme každé video v testovací sadě (nebo jejím vzorku)
    for i in tqdm(range(len(video_list)), desc="Počítám AOPC"):
        xai_data = xai_data_list[i]
        target_class_index = label_list[i]
        
        if mode == 'phrase':
            fragments = xai_data['phrase_frags']
            embeddings = xai_data['phrase_embeds']
            frame_embeddings = xai_data['frame_phrase_embeds']
        else: # wordsplit
            fragments = xai_data['wordsplit_frags']
            embeddings = xai_data['wordsplit_embeds']
            frame_embeddings = xai_data['frame_wordsplit_embeds']
        
        if not fragments or not embeddings:
            continue

        # 1. Získáme seřazený seznam důležitosti (stejně jako v 'run_perturbation_analysis')
        full_features = fe.aggregate_features(embeddings, fragments, frame_embeddings, tfidf_scores, tfidf_vocab)
        vector_key = "_".join(feature_set.split('_')[1:])
        X_original = full_features[vector_key].reshape(1, -1)
        original_prob = model.predict_proba(X_original)[0][target_class_index]
        
        importances = []
        for j in range(len(fragments)):
            temp_frags = fragments[:j] + fragments[j+1:]
            temp_embeds = embeddings[:j] + embeddings[j+1:]
            
            perturbed_features = fe.aggregate_features(temp_embeds, temp_frags, frame_embeddings, tfidf_scores, tfidf_vocab)
            X_perturbed = perturbed_features[vector_key].reshape(1, -1)
            perturbed_prob = model.predict_proba(X_perturbed)[0][target_class_index]
            importance = original_prob - perturbed_prob
            importances.append((fragments[j], importance, j)) # Uložíme i původní index
            
        importances.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Spočítáme poklesy pro K=1, K=2, ...
        # Vezmeme si původní fragmenty a vnoření
        current_frags = list(fragments)
        current_embeds = list(embeddings)
        
        # Iterujeme až do K
        for k in range(1, k_max + 1):
            if not importances: # Pokud už nejsou žádné fragmenty k odstranění
                break
                
            # Najdeme index fragmentu, který je teď na řadě (nejdůležitější)
            frag_to_remove, imp_value, original_index = importances.pop(0)
            
            # Musíme ho najít v našem *aktuálním* seznamu (jeho pozice se mění)
            try:
                # Najdeme ho podle textu
                idx_in_current = current_frags.index(frag_to_remove)
                # Odstraníme ho
                current_frags.pop(idx_in_current)
                current_embeds.pop(idx_in_current)
            except ValueError:
                # Už byl odstraněn (např. duplicitní slovo)
                continue # Přeskočíme na další K
            
            # 3. Získáme novou predikci s K odstraněnými fragmenty
            perturbed_features = fe.aggregate_features(current_embeds, current_frags, frame_embeddings, tfidf_scores, tfidf_vocab)
            X_perturbed = perturbed_features[vector_key].reshape(1, -1)
            perturbed_prob = model.predict_proba(X_perturbed)[0][target_class_index]
            
            probability_drop = original_prob - perturbed_prob
            
            # Uložíme výsledek
            all_perturbation_results.append({
                "video_index": i,
                "k_removed": k,
                "probability_drop": probability_drop
            })

    return pd.DataFrame(all_perturbation_results)