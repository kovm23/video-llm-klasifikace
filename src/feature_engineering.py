import os
import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import openai

# --- 1. ZPRACOVÁNÍ TEXTU (Fragmentace) ---

# Načteme NLP model pro 'Phrase' režim (extrakce frází)
# Použijeme 'en_core_sci_sm', který jsme nainstalovali s scispacy
print("Načítám spaCy model (en_core_sci_sm)...")
try:
    nlp = spacy.load("en_core_sci_sm")
except IOError:
    print("CHYBA: Model 'en_core_sci_sm' nenalezen.")
    print("Prosím, spusťte v terminálu (s aktivním venv):")
    print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz")
    raise
print("spaCy model načten.")


def process_texts(row):
    """
    Zpracuje řádek z CSV a vrátí seznamy fragmentů pro oba režimy.
    """
    # Režim Phrase: Extrahujeme podstatná jména a jejich části
    phrase_fragments = []
    # Spojíme popisy všech snímků do jednoho velkého textu
    full_phrase_text = " ".join(eval(row['phrase_descs'])) 
    doc = nlp(full_phrase_text)
    # Extrahujeme "noun chunks" (smysluplné skupiny podst. jmen) 
    for chunk in doc.noun_chunks:
        phrase_fragments.append(chunk.text.lower())
    # Pokud by nebyly žádné, použijeme alespoň jednotlivá slova
    if not phrase_fragments:
        phrase_fragments = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

    # Režim Word-split: Jen rozdělíme podle mezer
    wordsplit_fragments = []
    full_wordsplit_text = " ".join(eval(row['wordsplit_descs']))
    wordsplit_fragments = [word.lower() for word in full_wordsplit_text.split() if word.isalnum()] # Jen alfanumerická slova
    
    return phrase_fragments, wordsplit_fragments

# --- 2. TVORBA VNOŘENÍ (Embeddings) ---

def get_text_embeddings(client, text_list, model="text-embedding-3-small"):
    """
    Převede seznam textů na seznam vnoření (embeddings) pomocí OpenAI API.
    """
    if not text_list:
        return [] # Vrací prázdný seznam, pokud nejsou fragmenty
    try:
        # Nahradíme prázdné řetězce (API by selhalo)
        text_list = [text if text.strip() else "none" for text in text_list]
        
        response = client.embeddings.create(input=text_list, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"CHYBA API (Embeddings): {e}. Vracím prázdný seznam.")
        return []

# --- 3. DOMÉNOVÉ PŘÍZNAKY (Vaše inovace) ---

def get_domain_features(fragments, frame_embeddings_list, tfidf_scores, tfidf_vocab):
    """
    Vypočítá doménové příznaky pro jedno video.
    (Inspirováno AY 2025, kap. 8.4.5 a vaší metodikou)
    """
    # 1. Adaptované příznaky
    fragment_count = len(fragments) # [cite: 1986]
    
    # Slovník {slovo: idf_váha}
    tfidf_weights = dict(zip(tfidf_vocab, tfidf_scores))
    
    # TF-IDF váhy pro fragmenty v tomto videu
    fragment_tfidf_scores = [tfidf_weights.get(f, 0) for f in fragments]
    
    # Entropie (AY 2025, kap. 8.4.5) [cite: 1991]
    tfidf_entropy = entropy(fragment_tfidf_scores) if fragment_tfidf_scores else 0

    # 2. Nové video příznaky (dle metodiky, sekce V)
    # 'frame_embeddings_list' je seznam vnoření (např. 1536 dim) pro KAŽDÝ snímek
    temporal_variance = 0.0
    if len(frame_embeddings_list) > 1:
        similarities = []
        for i in range(len(frame_embeddings_list) - 1):
            sim = cosine_similarity(
                [frame_embeddings_list[i]], 
                [frame_embeddings_list[i+1]]
            )[0][0]
            similarities.append(sim)
        
        # Průměrná nepodobnost mezi snímky
        temporal_variance = 1.0 - np.mean(similarities) if similarities else 0
    
    # Vrátí vektor příznaků
    return np.array([
        fragment_count, 
        tfidf_entropy, 
        temporal_variance
    ])

# --- 4. AGREGAČNÍ STRATEGIE (Pooling) ---

def aggregate_features(fragment_embeddings, fragment_list, frame_embeddings_list, tfidf_scores, tfidf_vocab):
    """
    Vytvoří 4 finální vektory příznaků pro jedno video.
    (Inspirováno AY 2025, kap. 8.4)
    """
    
    # Převedeme na numpy pole pro snadnější výpočty
    if not fragment_embeddings:
        # Případ, kdy video nemá žádné fragmenty
        # Vytvoříme nulové vektory správné délky (pro text-embedding-3-small) [cite: 1672]
        # Toto je důležité, aby modely neselhaly
        D_MODEL = 1536 
        D_DOMAIN = 3 # Počet našich doménových příznaků
        
        emb_array = np.zeros((1, D_MODEL))
        domain_features = np.zeros(D_DOMAIN)
        
        mean_vec = np.zeros(D_MODEL)
        max_vec = np.zeros(D_MODEL)
        weighted_mean_vec = np.zeros(D_MODEL)
        
    else:
        emb_array = np.array(fragment_embeddings, dtype=np.float32)
    
        # Strategie 1: Mean pooling (AY 2025, kap. 8.4.1) [cite: 1889]
        mean_vec = np.mean(emb_array, axis=0)
        
        # Strategie 2: Max pooling (AY 2025, kap. 8.4.2) [cite: 1890]
        max_vec = np.max(emb_array, axis=0)
        
        # Strategie 3: Combined (část 1) - Vážený průměr
        tfidf_weights = dict(zip(tfidf_vocab, tfidf_scores))
        weights = [tfidf_weights.get(f, 1.0) for f in fragment_list] # 1.0 jako default váha [cite: 1926]
        
        # Zajistíme, aby váhy nebyly nulové
        weights = [w if w > 0 else 1.0 for w in weights]
        
        weighted_mean_vec = np.average(emb_array, axis=0, weights=weights)

        # Doménové příznaky
        domain_features = get_domain_features(
            fragment_list, 
            frame_embeddings_list, 
            tfidf_scores, 
            tfidf_vocab
        )

    # --- Sestavení finálních vektorů ---
    
    # 1. Mean
    final_mean = mean_vec
    
    # 2. Max
    final_max = max_vec
    
    # 3. Combined (AY 2025, kap. 8.4.3) [cite: 1891]
    final_combined = np.concatenate([mean_vec, max_vec, weighted_mean_vec])
    
    # 4. Combined + Domain (AY 2025, kap. 8.4.4) [cite: 1892]
    final_combined_domain = np.concatenate([mean_vec, max_vec, weighted_mean_vec, domain_features])
    
    return {
        "mean": final_mean,
        "max": final_max,
        "combined": final_combined,
        "combined_domain": final_combined_domain
    }

# --- Hlavní řídící funkce ---

def process_and_embed_all(data_path):
    """
    Hlavní smyčka Fáze 3: Načte 'generated_descriptions.csv',
    zpracuje texty, vytvoří vnoření, agreguje příznaky a uloží
    finální datasety připravené pro trénování.
    """
    
    # 1. Načtení API klíče a dat
    try:
        # ZMĚNA: Načtení z .env souboru (očekává, že load_dotenv() proběhlo v notebooku)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or len(api_key) < 10:
             raise ValueError("OPENAI_API_KEY nenalezen v .env souboru.")
        
        client = openai.OpenAI(api_key=api_key)
        print("OpenAI klient úspěšně vytvořen.")
    except Exception as e:
        print(f"CHYBA: {e}. Zastavuji.")
        print("Ujistěte se, že máte v kořenové složce projektu soubor .env s vaším OPENAI_API_KEY.")
        return

    input_csv_path = os.path.join(data_path, "generated_descriptions.csv")
    if not os.path.exists(input_csv_path):
        print(f"CHYBA: {input_csv_path} nenalezen. Spusťte prosím Fázi 2.")
        return

    df = pd.read_csv(input_csv_path)
    print(f"Načteno {len(df)} videí k zpracování...")

    # 2. Zpracování textů (Fragmentace)
    print("Zpracovávám texty (fragmentace)...")
    fragments = df.apply(process_texts, axis=1)
    df['phrase_fragments'] = [f[0] for f in fragments]
    df['wordsplit_fragments'] = [f[1] for f in fragments]

    # 3. Fitování TF-IDF (pouze na trénovacích datech!)
    # (Dle AY 2025, kap. 8.4.3)
    print("Fituji TF-IDF na trénovacích datech...")
    train_df = df[df['split'] == 'train']
    
    # Vytvoříme korpus pro oba režimy
    phrase_corpus = [" ".join(fragments) for fragments in train_df['phrase_fragments']]
    wordsplit_corpus = [" ".join(fragments) for fragments in train_df['wordsplit_fragments']]
    
    tfidf_vectorizer_phrase = TfidfVectorizer(min_df=5).fit(phrase_corpus)
    tfidf_vectorizer_wordsplit = TfidfVectorizer(min_df=5).fit(wordsplit_corpus)
    
    # Uložíme si váhy a slovník
    tfidf_phrase = (tfidf_vectorizer_phrase.idf_, tfidf_vectorizer_phrase.get_feature_names_out())
    tfidf_wordsplit = (tfidf_vectorizer_wordsplit.idf_, tfidf_vectorizer_wordsplit.get_feature_names_out())
    print("TF-IDF fitování dokončeno.")

    # 4. Hlavní smyčka: Vytvoření vnoření a agregace
    data_for_modeling = {
        "phrase_mean": [], "phrase_max": [], "phrase_combined": [], "phrase_combined_domain": [],
        "wordsplit_mean": [], "wordsplit_max": [], "wordsplit_combined": [], "wordsplit_combined_domain": [],
        "labels": [], "splits": [], "video_paths": []
    }

    # TODO: Implementovat logiku pro 'resume' (pokračování), pokud je to nutné

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Vytváření vnoření a příznaků"):
        
        # --- A. Režim PHRASE ---
        phrase_frags = row['phrase_fragments']
        phrase_embeds = get_text_embeddings(client, phrase_frags)
        
        # Pro doménový příznak 'temporal_variance' potřebujeme vnoření *jednotlivých snímků*
        frame_phrase_texts = [" ".join(chunk.text for chunk in nlp(desc).noun_chunks) if (desc and isinstance(desc, str) and desc.strip()) else "none" for desc in eval(row['phrase_descs'])]
        frame_phrase_embeds = get_text_embeddings(client, frame_phrase_texts)
        
        if phrase_embeds: # Jen pokud se podařilo vytvořit vnoření
            features_phrase = aggregate_features(
                phrase_embeds, 
                phrase_frags,
                frame_phrase_embeds, # Pro 'temporal_variance'
                tfidf_phrase[0], 
                tfidf_phrase[1]
            )
            data_for_modeling["phrase_mean"].append(features_phrase["mean"])
            data_for_modeling["phrase_max"].append(features_phrase["max"])
            data_for_modeling["phrase_combined"].append(features_phrase["combined"])
            data_for_modeling["phrase_combined_domain"].append(features_phrase["combined_domain"])
        
        # --- B. Režim WORDSPLIT ---
        wordsplit_frags = row['wordsplit_fragments']
        wordsplit_embeds = get_text_embeddings(client, wordsplit_frags)
        
        # Vnoření pro jednotlivé snímky
        frame_wordsplit_texts = [" ".join(desc.split()) if (desc and isinstance(desc, str) and desc.strip()) else "none" for desc in eval(row['wordsplit_descs'])]
        frame_wordsplit_embeds = get_text_embeddings(client, frame_wordsplit_texts)

        if wordsplit_embeds:
            features_wordsplit = aggregate_features(
                wordsplit_embeds,
                wordsplit_frags,
                frame_wordsplit_embeds,
                tfidf_wordsplit[0],
                tfidf_wordsplit[1]
            )
            data_for_modeling["wordsplit_mean"].append(features_wordsplit["mean"])
            data_for_modeling["wordsplit_max"].append(features_wordsplit["max"])
            data_for_modeling["wordsplit_combined"].append(features_wordsplit["combined"])
            data_for_modeling["wordsplit_combined_domain"].append(features_wordsplit["combined_domain"])

        # --- C. Metadata (pouze pokud oba režimy selhaly, přeskočíme) ---
        if not phrase_embeds or not wordsplit_embeds:
            print(f"VAROVÁNÍ: Přeskakuji video {row['video_path']} kvůli chybě vnoření.")
            if phrase_embeds and not wordsplit_embeds: # Jen pro jistotu, kdyby jeden prošel a druhý ne
                data_for_modeling["phrase_mean"].pop()
                data_for_modeling["phrase_max"].pop()
                data_for_modeling["phrase_combined"].pop()
                data_for_modeling["phrase_combined_domain"].pop()
            continue
            
        data_for_modeling["labels"].append(row['class_name'])
        data_for_modeling["splits"].append(row['split'])
        data_for_modeling["video_paths"].append(row['video_path'])

    # 5. Uložení finálních dat
    print("Ukládám finální zpracovaná data...")
    output_dir = os.path.join(data_path, "model_inputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Uložíme každý dataset zvlášť
    for key, data in data_for_modeling.items():
        if key in ["labels", "splits", "video_paths"]: continue # Metadata uložíme zvlášť
        
        # Převedeme seznam vektorů na 2D numpy pole
        X = np.array(data)
        np.save(os.path.join(output_dir, f"{key}_X.npy"), X)
        print(f"Uloženo: {key}_X.npy s tvarem {X.shape}")

    # Uložíme labely a splity
    np.save(os.path.join(output_dir, "labels.npy"), np.array(data_for_modeling["labels"]))
    np.save(os.path.join(output_dir, "splits.npy"), np.array(data_for_modeling["splits"]))
    np.save(os.path.join(output_dir, "video_paths.npy"), np.array(data_for_modeling["video_paths"]))
    
    print(f"Všechna data pro modelování uložena do: {output_dir}")
    print("\n--- Fáze 3 (Feature Engineering) je kompletní ---")