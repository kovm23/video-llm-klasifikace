import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle # Pro ukládání modelů

def load_data_for_modeling(model_inputs_path, data_path):
    """
    Načte všechny .npy soubory a připraví je pro trénování.
    Vrátí slovník s X, y a skupinami.
    """
    print(f"Načítám .npy data pro modelování z: {model_inputs_path}")
    data = {}
    
    # 1. Načtení X matic
    feature_sets = [
        "phrase_mean", "phrase_max", "phrase_combined", "phrase_combined_domain",
        "wordsplit_mean", "wordsplit_max", "wordsplit_combined", "wordsplit_combined_domain"
    ]
    for key in feature_sets:
        filepath = os.path.join(model_inputs_path, f"{key}_X.npy")
        if os.path.exists(filepath):
            data[f"{key}_X"] = np.load(filepath)
        else:
            print(f"VAROVÁNÍ: Soubor nenalezen: {filepath}")
            
    # 2. Načtení y (labels), splitů a cest
    labels_path = os.path.join(model_inputs_path, "labels.npy")
    splits_path = os.path.join(model_inputs_path, "splits.npy")
    paths_path = os.path.join(model_inputs_path, "relative_paths.npy")
    
    if not all(os.path.exists(p) for p in [labels_path, splits_path, paths_path]):
        print(f"CHYBA: Chybí jeden ze základních .npy souborů (labels, splits, or relative_paths).")
        print("Prosím, spusťte znovu Fázi 3.")
        return None
        
    labels = np.load(labels_path)
    splits = np.load(splits_path)
    relative_paths = np.load(paths_path, allow_pickle=True)
    
    # 3. Převedení textových labelů na čísla
    le = LabelEncoder()
    data['y'] = le.fit_transform(labels)
    data['labels_encoder'] = le # Uložíme enkodér pro pozdější použití
    
    # 4. Rozdělení na train/test
    train_mask = (splits == 'train')
    test_mask = (splits == 'test')
    
    data['y_train'] = data['y'][train_mask]
    data['y_test'] = data['y'][test_mask]
    
    # Rozdělíme i X matice
    for key in feature_sets:
        if f"{key}_X" in data:
            data[f"{key}_X_train"] = data[f"{key}_X"][train_mask]
            data[f"{key}_X_test"] = data[f"{key}_X"][test_mask]

    # 5. Načtení 'group_id' pro GroupKFold (klíčové pro správnou validaci)
    train_split_path = os.path.join(data_path, "train_split.csv")
    train_df_original = pd.read_csv(train_split_path)
    
    # Potřebujeme jen ty video cesty, které jsou v našem trénovacím .npy
    relative_paths_train_list = relative_paths[train_mask]
    
    # Vytvoříme mapu {relativní_cesta: group_id}
    video_path_to_group = dict(zip(train_df_original['relative_path'], train_df_original['group_id']))
    
    # Přiřadíme group_id ke každému řádku v X_train
    data['train_groups'] = [video_path_to_group.get(path) for path in relative_paths_train_list]
    
    if None in data['train_groups']:
        print("CHYBA: Nepodařilo se spárovat group_id pro všechna trénovací videa.")
    elif len(data['y_train']) != len(data['train_groups']):
         print(f"CHYBA: Nesouhlasí počet trénovacích vzorků ({len(data['y_train'])}) a skupin ({len(data['train_groups'])})")
    
    print("Data pro modelování úspěšně načtena.")
    return data

def get_classifiers():
    """
    Vrátí slovník klasifikátorů a jejich hyperparametrů pro GridSearchCV.
    (Inspirováno AY 2025, kap. 9.1 a 9.2)
    """
    classifiers = {
        "LogReg": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('clf', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
            ]),
            "params": {
                'pca__n_components': [0.8, 0.9, 0.95],
                'clf__C': [0.1, 1.0, 10.0]
            }
        },
        "RandomForest": {
            "pipeline": Pipeline([
                # RF nepotřebuje scaler ani PCA
                ('clf', RandomForestClassifier(random_state=42))
            ]),
            "params": {
                'clf__n_estimators': [100, 300],
                'clf__max_depth': [10, 50, None]
            }
        },
        "SVM": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('clf', SVC(probability=True, random_state=42)) # probability=True je nutné pro XAI
            ]),
            "params": {
                'pca__n_components': [0.8, 0.9, 0.95],
                'clf__C': [0.1, 1.0, 10.0]
            }
        },
        "XGBoost": {
            "pipeline": Pipeline([
                # XGBoost nepotřebuje scaler ani PCA
                ('clf', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
            ]),
            "params": {
                'clf__n_estimators': [100, 300],
                'clf__learning_rate': [0.01, 0.1]
            }
        }
    }
    return classifiers

def run_modeling_pipeline(data, models_output_path, results_output_path):
    """
    Spustí celý proces trénování a evaluace pro všechny kombinace.
    Uloží modely a vrátí DataFrame s výsledky.
    """
    # Seznam všech 8 sad příznaků, které jsme vygenerovali
    feature_sets = [
        "phrase_mean", "phrase_max", "phrase_combined", "phrase_combined_domain",
        "wordsplit_mean", "wordsplit_max", "wordsplit_combined", "wordsplit_combined_domain"
    ]
    classifiers = get_classifiers()
    
    y_train = data['y_train']
    y_test = data['y_test']
    train_groups = data['train_groups']
    
    # Ověření, že máme skupiny pro všechna trénovací data
    if len(y_train) != len(train_groups):
        print(f"CHYBA: Počet trénovacích vzorků ({len(y_train)}) se neshoduje s počtem skupin ({len(train_groups)})")
        return pd.DataFrame() # Vrátíme prázdný dataframe

    # validace (dle metodiky AY 2025, kap. 6.1.2)
    group_kfold = GroupKFold(n_splits=5)
    
    all_results = []
    
    if not os.path.exists(models_output_path):
        os.makedirs(models_output_path)

    for feature_set in tqdm(feature_sets, desc="Trénování sad příznaků"):
        # Přeskočíme, pokud pro set neexistují data
        if f"{feature_set}_X_train" not in data:
            print(f"Přeskakuji {feature_set}, nenalezena data.")
            continue
            
        X_train = data[f"{feature_set}_X_train"]
        X_test = data[f"{feature_set}_X_test"]
        
        for clf_name, clf_info in classifiers.items():
            
            print(f"\n--- Trénuji: {feature_set} | Klasifikátor: {clf_name} ---")
            
            # Použijeme GridSearchCV s GroupKFold (Robustní metoda ladění)
            grid_search = GridSearchCV(
                clf_info["pipeline"], 
                clf_info["params"], 
                cv=group_kfold, 
                scoring='f1_weighted', # Metrika dle metodiky
                n_jobs=-1 # Použije všechna jádra
            )
            
            # Trénujeme (GridSearch si sám vezme skupiny)
            grid_search.fit(X_train, y_train, groups=train_groups)
            
            best_model = grid_search.best_estimator_
            
            # ULOŽÍME MODEL pro pozdější XAI
            model_filename = f"{feature_set}_{clf_name}.pkl"
            with open(os.path.join(models_output_path, model_filename), 'wb') as f:
                pickle.dump(best_model, f)
            
            # Evaluace na testovací sadě
            y_pred = best_model.predict(X_test)
            
            # Uložení reportu
            report = classification_report(
                y_test, 
                y_pred, 
                output_dict=True, 
                target_names=data['labels_encoder'].classes_,
                zero_division=0
            )
            
            result_row = {
                "Feature Set": feature_set,
                "Classifier": clf_name,
                "Accuracy": report["accuracy"],
                "F1 (Weighted)": report["weighted avg"]["f1-score"],
                "Precision (Weighted)": report["weighted avg"]["precision"],
                "Recall (Weighted)": report["weighted avg"]["recall"],
                "Model Path": os.path.join(models_output_path, model_filename)
            }
            all_results.append(result_row)
            
            print(f"Hotovo. F1 (Weighted): {report['weighted avg']['f1-score']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_output_path, index=False)
    print("\n--- Modelování dokončeno ---")
    print(f"Výsledky uloženy do: {results_output_path}")
    return results_df