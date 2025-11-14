import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_results_heatmap(results_df):
    """
    Vytvoří heatmapu srovnávající výkon modelů.
    """
    
    # Připravíme data pro heatmapu
    # Chceme F1 skóre jako hlavní metriku
    try:
        heatmap_data = results_df.pivot(
            index="Classifier", 
            columns="Feature Set", 
            values="F1 (Weighted)"
        )
        
        # Seřadíme sloupce logicky
        col_order = [
            "phrase_mean", "phrase_max", "phrase_combined", "phrase_combined_domain",
            "wordsplit_mean", "wordsplit_max", "wordsplit_combined", "wordsplit_combined_domain"
        ]
        # Ponecháme jen ty sloupce, které reálně existují
        heatmap_data = heatmap_data[[col for col in col_order if col in heatmap_data.columns]]
        
    except Exception as e:
        print(f"CHYBA při vytváření pivot tabulky: {e}")
        print("Ujistěte se, že 'results_df' obsahuje sloupce 'Classifier', 'Feature Set' a 'F1 (Weighted)'.")
        return

    plt.figure(figsize=(16, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".3f", 
        cmap="viridis", 
        linewidths=.5
    )
    plt.title("Srovnání výkonu modelů (Weighted F1-Score)", fontsize=16)
    plt.xlabel("Sada příznaků (Feature Set)")
    plt.ylabel("Klasifikátor")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

def plot_aopc_curve(aopc_results, title="AOPC Curve"):
    """
    Vykreslí AOPC křivku (Area Over the Perturbation Curve).
    (Inspirováno AY 2025, Obr. 10.7)
    
    'aopc_results' je DataFrame se sloupci 'k_removed' a 'probability_drop'
    """
    
    plt.figure(figsize=(10, 6))
    # 'errorbar="sd"' automaticky spočítá a vykreslí průměr a směrodatnou odchylku
    sns.lineplot(
        data=aopc_results, 
        x='k_removed', 
        y='probability_drop', 
        marker='o',
        errorbar='sd' 
    )
    
    # Výpočet AOPC (Area Over the Perturbation Curve)
    # Je to jednoduše průměr všech poklesů
    aopc_score = np.mean(aopc_results['probability_drop'])
    
    plt.title(f"{title}\nPrůměrný pokles (AOPC): {aopc_score:.4f}", fontsize=16)
    plt.xlabel("Počet odstraněných nejlepších fragmentů (K)", fontsize=12)
    plt.ylabel("Průměrný pokles pravděpodobnosti", fontsize=12)
    plt.grid(True)
    plt.show()