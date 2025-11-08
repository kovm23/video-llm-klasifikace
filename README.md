# Klasifikace videa na základě popisů generovaných velkými jazykovými modely

**Autor:** Matěj Koval
**Vedoucí práce:** Tomáš Klieger
**Studijní program:** Aplikovaná informatika
**Instituce:** Vysoká škola ekonomická v Praze, Fakulta informatiky a statistiky

---

## 🎯 Cíl práce

Cílem této práce je navrhnout a implementovat softwarové řešení pro klasifikaci videa (konkrétně datasetu **UCF101**), které převádí vizuální data na textové popisy pomocí multimodálních LLM. Následně využívá vnoření (embeddings) těchto popisů jako **interpretovatelné příznaky** pro trénink klasifikačních modelů.

Projekt metodologicky navazuje na práci AY (2025) a Balek et al. (2025) a aplikuje jejich principy (např. režimy *Phrase* vs. *Word-split*, agregační strategie) na doménu videa.

---

## ⚙️ Metodika (Pipeline)

Softwarové řešení je implementováno jako sekvenční pipeline v Pythonu:

1.  **Zpracování videa**: Extrakce reprezentativních snímků z videí (dataset UCF101) pomocí OpenCV.

2.  **Generování popisů**: Každý snímek je popsán pomocí multimodálního LLM (např. Qwen-VL, GPT-4o). Experimentuje se se dvěma režimy:
    * **Phrase**: Generování přirozených diagnostických frází.
    * **Word-split**: Generování afirmativních popisů rozdělených na jednotlivá slova.

3.  **Tvorba vnoření (Embeddings)**: Textové fragmenty (slova/fráze) jsou převedeny na numerické vektory (vnoření) pomocí modelů jako `text-embedding-3-small` nebo `Bio_ClinicalBERT`.

4.  **Agregace (Pooling)**: Všechny vektory fragmentů z jednoho videa jsou sloučeny do jediného vektoru příznaků pomocí strategií *Mean*, *Max*, *Combined* a *Combined+Domain*.

5.  **Klasifikace a evaluace**: Výsledné vektory jsou použity pro trénink a evaluaci vysvětlitelných klasifikátorů (např. Logistická regrese, Random Forest) ze `scikit-learn`. Výkon je srovnán s baseline modelem (3D-CNN).

---

## 🚀 Jak spustit experiment

> **Poznámka:** Projekt je navržen pro běh v prostředí Google Colab.

### 1. Klonování repozitáře

```bash
git clone <URL_REPOZITÁŘE>
cd video-llm-klasifikace
