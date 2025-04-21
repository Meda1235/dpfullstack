import os
import json
import math
import logging

from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Query
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union

#  Statistika
from scipy.stats import (
    boxcox, pearsonr, spearmanr, kendalltau, norm, shapiro, normaltest, kstest, zscore,
    ttest_ind, mannwhitneyu, f_oneway, kruskal, ttest_rel, wilcoxon, levene,
    chi2_contingency, fisher_exact
)
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit, Logit
from statsmodels.tools.sm_exceptions import PerfectSeparationError

#  Analýza faktorů a PCA
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 🔍 Machine Learning – klasifikace, regrese, shlukování
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report, silhouette_score
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 🧮 Práce s daty
import pandas as pd
import numpy as np
import requests


app = FastAPI()
load_dotenv()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassificationMetricsInterpretation(BaseModel):
    accuracy: float
    precision: float # Weighted precision
    recall: float    # Weighted recall
    f1_score: float  # Weighted F1

class ClassificationInterpretationRequest(BaseModel):
    analysis_type: str = "classification"
    algorithm_used: str
    target_variable: str
    features_used: List[str]
    metrics: ClassificationMetricsInterpretation
    has_feature_importances: bool
    number_of_classes: int

# --- Nový endpoint pro AI interpretaci klasifikace ---
@app.post("/api/interpret_classification")
async def interpret_classification(req: ClassificationInterpretationRequest):
    # --- System Prompt pro LLM ---
    system_prompt = (
        "Jsi AI asistent specializující se na analýzu dat. Uživatel provedl klasifikační analýzu a poskytne ti její klíčové výsledky.\n\n"
        "Tvým úkolem je interpretovat tyto výsledky v češtině, jednoduchým a srozumitelným jazykem.\n\n"
        f"Použitý algoritmus byl: **{req.algorithm_used}**. Cílová proměnná (kterou se snažíme predikovat) je '{req.target_variable}'. Počet tříd cílové proměnné: {req.number_of_classes}.\n\n"
        "**Interpretace klíčových metrik (Accuracy, Precision, Recall, F1-Score):**\n"
        "- **Accuracy (Přesnost):** Jaké procento všech predikcí bylo správných? (např. Accuracy 0.85 znamená 85% správných predikcí celkově). Je to dobrá metrika, pokud jsou třídy vyvážené.\n"
        "- **Precision (Přesnost pozitivních predikcí):** Z těch případů, které model označil jako pozitivní (pro danou třídu), kolik jich bylo skutečně pozitivních? Vysoká precision znamená málo falešně pozitivních výsledků.\n"
        "- **Recall (Senzitivita, Úplnost):** Z těch případů, které byly skutečně pozitivní (pro danou třídu), kolik jich model správně identifikoval? Vysoký recall znamená málo falešně negativních výsledků.\n"
        "- **F1-Score:** Harmonický průměr Precision a Recall. Dobrá metrika, pokud hledáme rovnováhu mezi Precision a Recall, nebo pokud jsou třídy nevyvážené.\n"
        f"(Poznámka: Poskytnuté metriky Precision, Recall a F1 jsou vážené průměry přes všechny {req.number_of_classes} třídy, což zohledňuje jejich velikost.)\n\n"
        "**Celkové zhodnocení modelu:**\n"
        "- Na základě hodnot metrik (typicky F1 nebo Accuracy) zhodnoť, jak dobře model funguje. Hodnoty blízko 1 jsou ideální, hodnoty kolem 0.5 u binární klasifikace mohou znamenat, že model není o moc lepší než náhodné hádání.\n"
        "- Zmínit, že interpretace metrik závisí na kontextu problému (např. v medicíně může být důležitější vysoký Recall než Precision).\n\n"
        "**Další informace:**\n"
        f"- {'Model poskytl informaci o důležitosti příznaků (feature importances).' if req.has_feature_importances else 'Model neposkytl informaci o důležitosti příznaků.'} Pokud ano, znamená to, že některé vstupní proměnné měly větší vliv na rozhodování modelu než jiné.\n"
        "- Byla zmíněna i Confusion Matrix (matice záměn), která detailně ukazuje, jaké typy chyb model dělal (které třídy si pletl s kterými).\n\n"
        "Pravidla:\n"
        "- Odpovídej v češtině.\n"
        "- Buď srozumitelný pro někoho bez hlubokých znalostí statistiky.\n"
        "- Vysvětli význam klíčových metrik jednoduše.\n"
        "- Neuváděj vzorce ani kód.\n"
        "- Formátuj odpověď pro dobrou čitelnost."
    )

    # --- Sestavení User Promptu ---
    user_prompt_parts = [
        f"Provedl jsem klasifikační analýzu pomocí algoritmu '{req.algorithm_used}'.",
        f"Cílová proměnná: '{req.target_variable}' ({req.number_of_classes} třídy).",
        f"Použité příznaky: {', '.join(req.features_used)}.",
        "\nSouhrnné metriky modelu (vážený průměr):"
        f"- Přesnost (Accuracy): {req.metrics.accuracy:.4f}",
        f"- Precision: {req.metrics.precision:.4f}",
        f"- Recall: {req.metrics.recall:.4f}",
        f"- F1-Score: {req.metrics.f1_score:.4f}",
        f"\nModel {'poskytl' if req.has_feature_importances else 'neposkytl'} informaci o důležitosti příznaků.",
        "\nProsím, interpretuj tyto výsledky."
    ]
    full_user_prompt = "\n".join(user_prompt_parts)

    # --- Volání LLM API ---
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
             raise HTTPException(status_code=500, detail="Chybí konfigurace API klíče pro LLM.")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free", # Nebo jiný model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                "max_tokens": 600
            },
            timeout=60
        )
        response.raise_for_status()

        llm_data = response.json()
        interpretation_text = llm_data.get("choices", [{}])[0].get("message", {}).get("content")

        if not interpretation_text:
             raise HTTPException(status_code=500, detail="AI nevrátila platnou interpretaci.")

        return {"interpretation": interpretation_text}

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Chyba komunikace s LLM API: {req_err}")
        raise HTTPException(status_code=503, detail=f"Chyba při komunikaci s AI službou: {req_err}")
    except Exception as e:
        logger.error(f"Neočekávaná chyba při interpretaci klasifikace: {e}", exc_info=True)
        # print(f"User prompt was: {full_user_prompt}") # Pro debugging
        raise HTTPException(status_code=500, detail=f"Interní chyba serveru při generování interpretace: {str(e)}")
# Nastavení logování
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Nový Pydantic Model pro Klasifikaci ---
class ClassificationRequest(BaseModel):
    feature_columns: List[str] = Field(..., min_items=1)
    target_column: str
    algorithm: Literal["auto", "logistic_regression", "knn", "decision_tree", "random_forest", "naive_bayes"] = "auto"
    standardize: bool = True
    test_size: float = Field(default=0.25, ge=0.1, le=0.5) # Testovací sada 10-50%
    knn_neighbors: Optional[int] = Field(default=5, ge=1)
    random_state: int = 42 # Pro reprodukovatelnost

# --- Nový Endpoint pro Klasifikaci ---
@app.post("/api/classification_analysis")
async def classification_analysis(req: ClassificationRequest):
    logger.info(f"Classification request received: {req.dict()}")
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    headers = stored_datasets["latest"]["headers"]
    data = stored_datasets["latest"]["data"]
    df = pd.DataFrame(data, columns=headers)

    # --- Validace vstupů ---
    if req.target_column in req.feature_columns:
        raise HTTPException(status_code=400, detail="Cílová proměnná nemůže být zároveň příznakem.")
    if not req.feature_columns:
         raise HTTPException(status_code=400, detail="Musíte vybrat alespoň jeden příznak (feature column).")

    all_selected_columns = req.feature_columns + [req.target_column]
    for col in all_selected_columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Sloupec '{col}' nebyl v datech nalezen.")

    # --- Příprava dat ---
    try:
        df_subset = df[all_selected_columns].copy()

        # Základní čištění - odstranění řádků s jakoukoli chybějící hodnotou v vybraných sloupcích
        initial_rows = len(df_subset)
        df_subset.dropna(inplace=True)
        if len(df_subset) < initial_rows:
             logger.warning(f"Odstraněno {initial_rows - len(df_subset)} řádků kvůli chybějícím hodnotám.")

        if df_subset.empty:
            raise ValueError("Po odstranění chybějících hodnot nezůstala žádná data.")

        # Kontrola cílové proměnné
        target_series = df_subset[req.target_column]
        if target_series.nunique() < 2:
             raise ValueError(f"Cílová proměnná '{req.target_column}' musí mít alespoň 2 unikátní hodnoty pro klasifikaci.")
        if target_series.nunique() > 50: # Varování pro příliš mnoho tříd
             logger.warning(f"Cílová proměnná '{req.target_column}' má {target_series.nunique()} unikátních hodnot. To může být pro klasifikaci příliš mnoho.")
        # Nepřevádíme na kategorii explicitně, necháme sklearn, aby si poradil

        X = df_subset[req.feature_columns]
        y = target_series

        # Identifikace numerických a kategorických příznaků
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")

        # --- Preprocessing Pipeline ---
        transformers = []
        if numeric_features:
            numeric_transformer_steps = []
            if req.standardize:
                numeric_transformer_steps.append(('scaler', StandardScaler()))
            # Můžeme přidat imputer, pokud bychom nechtěli dropovat NA
            # numeric_transformer_steps.append(('imputer', SimpleImputer(strategy='median')))
            if numeric_transformer_steps:
                 transformers.append(('num', Pipeline(steps=numeric_transformer_steps), numeric_features))

        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                # ('imputer', SimpleImputer(strategy='most_frequent')), # Pro případné NA v kategoriích
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False pro snazší manipulaci
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        if not transformers:
             raise ValueError("Nebyly nalezeny žádné vhodné příznaky (numerické nebo kategorické) pro zpracování.")

        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough') # 'passthrough' zachová sloupce, které nejsou explicitně transformovány (pokud by nějaké byly)

        # Rozdělení dat PŘED aplikací preprocessingu (fit jen na train)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=req.random_state, stratify=y # Stratifikace je důležitá
        )
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    except ValueError as e:
         logger.error(f"Chyba při přípravě dat: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Chyba při přípravě dat: {str(e)}")
    except Exception as e:
         logger.error(f"Neočekávaná chyba při přípravě dat: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Neočekávaná chyba serveru: {str(e)}")


    # --- Výběr a trénování modelu ---
    used_algorithm = req.algorithm
    reason = ""

    # Jednoduchá auto logika (může být sofistikovanější)
    if used_algorithm == "auto":
        if len(df_subset) < 1000 and not categorical_features:
             used_algorithm = "knn"
             reason = "Malý dataset bez kategorických příznaků -> KNN."
        elif categorical_features:
             used_algorithm = "random_forest" # Random Forest si dobře poradí s mixem typů
             reason = "Dataset obsahuje kategorické příznaky -> Random Forest."
        else:
             used_algorithm = "logistic_regression"
             reason = "Standardní volba -> Logistická regrese."
        logger.info(f"Auto algorithm selected: {used_algorithm}")


    model_instance: Any # Pro typovou kontrolu
    if used_algorithm == "logistic_regression":
        model_instance = LogisticRegression(random_state=req.random_state, max_iter=1000) # Zvýšení iterací pro konvergenci
    elif used_algorithm == "knn":
         if not numeric_features and not req.standardize:
             logger.warning("KNN použito bez numerických příznaků nebo standardizace. Výsledky mohou být neoptimální.")
         model_instance = KNeighborsClassifier(n_neighbors=req.knn_neighbors or 5)
    elif used_algorithm == "decision_tree":
        model_instance = DecisionTreeClassifier(random_state=req.random_state)
    elif used_algorithm == "random_forest":
        model_instance = RandomForestClassifier(random_state=req.random_state)
    elif used_algorithm == "naive_bayes":
         if categorical_features:
              # Pro mix typů by byl lepší CategoricalNB nebo smíšený přístup, GaussianNB předpokládá Gaussovské rozdělení
              logger.warning("Používáte GaussianNB s kategorickými příznaky po OneHotEncode. Zvažte vhodnější Naive Bayes variantu pro smíšená data.")
         if not req.standardize and numeric_features:
              logger.warning("GaussianNB použit bez standardizace numerických příznaků.")
         model_instance = GaussianNB()
    else:
         raise HTTPException(status_code=400, detail=f"Neznámý algoritmus: {used_algorithm}")


    # Vytvoření kompletní pipeline: Preprocessing -> Model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', model_instance)])

    try:
        # Trénování pipeline
        logger.info(f"Trénování modelu: {used_algorithm}...")
        pipeline.fit(X_train, y_train)
        logger.info("Trénování dokončeno.")

        # Predikce na testovacích datech
        y_pred = pipeline.predict(X_test)
        logger.info("Predikce na testovací sadě dokončena.")

        # --- Vyhodnocení ---
        accuracy = accuracy_score(y_test, y_pred)
        # Použití 'weighted' pro průměrování metrik v multi-class problémech, 'macro' je další možnost
        # zero_division=0 zabrání chybě, pokud některá třída nemá predikce/skutečné hodnoty v test setu (což by bylo divné)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Získání názvů tříd pro confusion matrix
        class_labels = sorted(y.unique().astype(str).tolist()) # Unikátní třídy z původních 'y', seřazené
        cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_) # Použijeme třídy z pipeline pro správné pořadí
        cm_list = cm.tolist()

        # Classification report jako slovník
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0, labels=pipeline.classes_, target_names=[str(cls) for cls in pipeline.classes_])


        # --- Feature Importances (pokud jsou dostupné) ---
        feature_importances = None
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = pipeline.named_steps['classifier'].feature_importances_
            # Získání názvů příznaků PO transformaci (např. po OneHotEncode)
            try:
                # Získání transformerů z ColumnTransformer
                preprocessor_fitted = pipeline.named_steps['preprocessor']
                feature_names_out = preprocessor_fitted.get_feature_names_out()
                if len(importances) == len(feature_names_out):
                    feature_importances = [{"feature": name, "importance": round(float(imp), 4)}
                                           for name, imp in zip(feature_names_out, importances)]
                    # Seřazení podle důležitosti
                    feature_importances.sort(key=lambda x: x["importance"], reverse=True)
                else:
                    logger.warning("Nesoulad počtu importances a názvů příznaků po transformaci.")
            except Exception as e:
                 logger.warning(f"Nepodařilo se získat názvy příznaků pro feature importances: {e}")


    except ValueError as e:
         # Specifické chyby z fit/predict/metrics
         logger.error(f"Chyba během trénování/vyhodnocení: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Chyba klasifikace: {str(e)}")
    except Exception as e:
         logger.error(f"Neočekávaná chyba během trénování/vyhodnocení: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Neočekávaná chyba serveru: {str(e)}")


    # --- Sestavení odpovědi ---
    return {
        "algorithm_used": used_algorithm,
        "reason": reason,
        "standardized": req.standardize if numeric_features else None, # Relevantní jen pokud byly numerické feat.
        "test_size": req.test_size,
        "knn_neighbors": req.knn_neighbors if used_algorithm == "knn" else None,
        "feature_columns_used": req.feature_columns, # Původní seznam
        "target_column": req.target_column,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_weighted": round(precision, 4),
            "recall_weighted": round(recall, 4),
            "f1_weighted": round(f1, 4),
        },
        "classification_report": report_dict, # Detailní report
        "confusion_matrix": cm_list,
        "confusion_matrix_labels": [str(cls) for cls in pipeline.classes_], # Labels pro osu matice
        "feature_importances": feature_importances # Může být None
    }


class DependencyTestRequest(BaseModel):
    columns: List[str]
    method: Optional[str] = "auto"
    paired: Optional[bool] = False

def run_shapiro(series):
    series = series.dropna()
    if len(series) < 3 or len(series.unique()) < 2: return 0.0
    try:
        stat, p_value = shapiro(series)
        if np.isnan(p_value): return 0.0
        return p_value
    except Exception: return 0.0

def run_levene(*groups):
    valid_groups = [g.dropna() for g in groups if len(g.dropna()) >= 3]
    if len(valid_groups) < 2: return 0.0
    try:
        stat, p_value = levene(*valid_groups)
        if np.isnan(p_value): return 0.0
        return p_value
    except Exception: return 0.0



import pandas as pd
import numpy as np
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
# Importy pro statistické testy
from scipy.stats import (
    chi2_contingency, # Pro Chi2 a očekávané frekvence
    fisher_exact,    # Pro Fisherův test
    shapiro,         # Test normality
    levene,          # Test homogenity rozptylů
    ttest_ind,       # Nepárový t-test (včetně Welchova)
    ttest_rel,       # Párový t-test
    mannwhitneyu,    # Mann-Whitney U (nepárový neparametrický)
    wilcoxon,        # Wilcoxon (párový neparametrický)
    f_oneway,        # ANOVA
    kruskal          # Kruskal-Wallis
)
import logging # Pro lepší logování

# --- Pydantic modely ---
class DependencyTestRequest(BaseModel):
    columns: List[str]
    method: Optional[str] = "auto"
    paired: Optional[bool] = False

# --- FastAPI App (předpokládáme existenci) ---
# app = FastAPI()

# --- Mock Data a Globální proměnné (předpokládáme existenci) ---
# stored_datasets = { ... }

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pomocné funkce (PŘEDPOKLÁDÁME JEJICH EXISTENCI NEBO DEFINICI V SOUBORU) ---
def run_shapiro(series):
    """Spustí Shapiro-Wilk test, vrací p-hodnotu nebo 0.0 při chybě/málo datech."""
    series = series.dropna()
    if len(series) < 3 or len(series.unique()) < 2:
        return 0.0 # Nelze testovat nebo nemá smysl
    try:
        # Ošetření konstantních dat, která mohou projít kontrolou unique, ale shapiro selže
        if series.std() < 1e-10: return 0.0
        stat, p_value = shapiro(series)
        if np.isnan(p_value): return 0.0
        return p_value
    except ValueError: return 0.0
    except Exception as e: logger.warning(f"Neočekávaná chyba v shapiro: {e}"); return 0.0

def run_levene(*groups):
    """Spustí Levene test, vrací p-hodnotu nebo 0.0 při chybě/málo datech."""
    valid_groups = [g.dropna() for g in groups if len(g.dropna()) >= 3]
    if len(valid_groups) < 2: return 0.0
    try:
        # Ošetření, pokud mají některé skupiny nulový rozptyl
        if any(g.std() < 1e-10 for g in valid_groups if len(g)>0):
            logger.warning("Levene test: Některá skupina má nulový rozptyl.")
            # V tomto případě nemůžeme předpokládat homogenitu
            return 0.0
        stat, p_value = levene(*valid_groups)
        if np.isnan(p_value): return 0.0
        return p_value
    except ValueError: return 0.0
    except Exception as e: logger.warning(f"Neočekávaná chyba v levene: {e}"); return 0.0

def robust_clean_nan_inf(value: Any) -> Any:
    """Rekurzivně čistí data pro JSON, nahrazuje NaN/Inf za None, převádí numpy typy."""
    if isinstance(value, dict):
        return {k: robust_clean_nan_inf(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [robust_clean_nan_inf(elem) for elem in value]
    elif isinstance(value, (np.ndarray, pd.Series)):
         # Zkontrolujeme dtype PŘED konverzí na list
         dtype_kind = value.dtype.kind
         cleaned_list = [robust_clean_nan_inf(elem) for elem in value.tolist()]
         # Pokud byl původní typ integer a všechny hodnoty jsou None, vrátíme None list,
         # jinak by se mohly prázdné listy interpretovat jako float
         if dtype_kind in 'iu' and all(x is None for x in cleaned_list):
             return cleaned_list
         # Zkusíme zachovat integery, pokud je to možné
         if dtype_kind in 'iu' and all(isinstance(x, int) or x is None for x in cleaned_list):
             return cleaned_list
         # Pokud obsahuje float nebo mix, převedeme None na NaN a pak na float list
         # s None tam, kde byl původně None/NaN/Inf
         # Toto je komplexní, možná je jednodušší nechat to být listem smíšených typů
         return cleaned_list # Vrátíme list se zachovanými None
    elif isinstance(value, (float, np.floating)):
        if pd.isna(value) or np.isinf(value): return None
        return float(value)
    elif isinstance(value, (int, np.integer)):
         if pd.isna(value): return None
         return int(value)
    elif isinstance(value, (bool, np.bool_)):
        return bool(value)
    elif isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    elif value is None or pd.isna(value):
        return None
    # Vrátíme hodnotu, pokud je to základní typ podporovaný JSONem
    elif isinstance(value, (str,)):
        return value
    # Pro ostatní nepodporované typy vrátíme jejich string reprezentaci nebo None
    try:
        json.dumps(value) # Zkusíme, jestli je JSON serializovatelný
        return value
    except TypeError:
        logger.warning(f"Typ {type(value)} není přímo JSON serializovatelný, vracím None.")
        return None
# --- Endpoint ---


@app.post("/api/dependency_test")
async def dependency_test(req: DependencyTestRequest):
    logger.info(f"Received dependency test request: Cols={req.columns}, Method='{req.method}', Paired={req.paired}")
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=404, detail="Data nejsou nahraná.")

    headers = stored_datasets["latest"].get("headers")
    data = stored_datasets["latest"].get("data")
    # Získání typů sloupců
    raw_types = stored_datasets["latest"].get("column_types", {})
    if not raw_types:
         logger.warning("Chybí info o typech sloupců, detekuji...")
         df_temp = pd.DataFrame(data, columns=headers)
         raw_types = {}
         for col in headers:
             try: pd.to_numeric(df_temp[col], errors='raise'); raw_types[col] = {"type": "Numeric"}
             except: raw_types[col] = {"type": "Categorical"}
         logger.info(f"Detekované typy: {raw_types}")

    types = {col: info.get("type", "Unknown").replace("Číselný","Numeric").replace("Kategorie","Categorical") for col, info in raw_types.items()}
    df = pd.DataFrame(data, columns=headers)

    selected_cols = req.columns
    method = (req.method or "auto").lower()
    paired = req.paired or False

    if len(selected_cols) < 2: raise HTTPException(status_code=400, detail="Je třeba vybrat alespoň dvě proměnné.")
    if not all(col in headers for col in selected_cols):
        missing = [col for col in selected_cols if col not in headers]; raise HTTPException(status_code=404, detail=f"Sloupce nenalezeny: {missing}")

    # --- Příprava dat a logika v jednom try bloku ---
    try:
        # Inicializace výstupního slovníku hned na začátku
        test_output: Dict[str, Any] = {
            "input_method": method, "input_paired": paired, "test_name": None,
            "reason": f"Metoda: {method}.", "columns": selected_cols, "results": [],
            "statistic": None, "p_value": None, "statistic_name": None,
            "degrees_freedom": None, "contingency_table": None, "warning_message": None
        }
        actual_method = method # Výchozí metoda

        # Příprava dat
        subdf = df[selected_cols].copy()
        n_selected = len(selected_cols)
        cat_cols_in_scope = [col for col in selected_cols if types.get(col) == "Categorical"]
        num_cols_in_scope = [col for col in selected_cols if types.get(col) == "Numeric"]
        n_cat = len(cat_cols_in_scope); n_num = len(num_cols_in_scope)

        if n_cat + n_num != n_selected:
            unknown_cols = [col for col in selected_cols if types.get(col, "Unknown") == "Unknown"]; logger.warning(f"Neznámý typ pro: {unknown_cols}.")

        logger.info(f"Selected: {n_selected} total, {n_cat} cat, {n_num} num")

        if n_num > 0: subdf[num_cols_in_scope] = subdf[num_cols_in_scope].apply(pd.to_numeric, errors='coerce')
        if n_cat > 0: subdf[cat_cols_in_scope] = subdf[cat_cols_in_scope].astype('category')

        initial_rows = len(subdf); subdf.dropna(inplace=True); final_rows = len(subdf)
        logger.info(f"Data prep: Removed {initial_rows - final_rows} NA rows. Final rows: {final_rows}")
        if final_rows < 5: raise ValueError(f"Nedostatek dat po odstranění NA (nalezeno: {final_rows}, min 5).")

        # --- Logika 'auto' výběru ---
        if method == "auto":
            logger.info("AUTO method selected. Determining test...")
            reason = "Auto výběr: "
            AUTO_CHI2, AUTO_FISHER = "chi2", "fisher"
            AUTO_TTEST_PAIRED, AUTO_TTEST_UNPAIRED_EQUAL, AUTO_TTEST_UNPAIRED_WELCH = "t.test.paired", "t.test.unpaired.equal", "t.test.unpaired.welch"
            AUTO_WILCOXON, AUTO_MANNWHITNEY = "wilcoxon", "mannwhitney"
            AUTO_ANOVA, AUTO_KRUSKAL = "anova", "kruskal"

            if n_cat == 2 and n_num == 0:
                cat1, cat2 = cat_cols_in_scope
                try:
                    cont_table = pd.crosstab(subdf[cat1], subdf[cat2])
                    if cont_table.size < 4 or cont_table.shape[0] < 2 or cont_table.shape[1] < 2: raise ValueError("Tabulka < 2x2.")
                    chi2, p, dof, expected = chi2_contingency(cont_table)
                    if np.any(expected < 5):
                        actual_method = AUTO_FISHER; reason += "2 Kat. (oček. < 5) -> Fisher."
                        if cont_table.shape != (2, 2): actual_method = AUTO_CHI2; reason = reason.replace("-> Fisher.", "-> Chi2 (tabulka není 2x2).")
                    else: actual_method = AUTO_CHI2; reason += "2 Kat. (oček. >= 5) -> χ²."
                except ValueError as e: raise ValueError(f"Nelze vytvořit kont. tabulku pro {cat1} vs {cat2}: {e}") from e
            elif n_cat > 2 and n_num == 0: actual_method = AUTO_CHI2; reason += f">2 Kat. -> χ² (mezi {cat_cols_in_scope[0]} vs {cat_cols_in_scope[1]})."
            elif n_cat == 1 and n_num == 1:
                cat_col, num_col = cat_cols_in_scope[0], num_cols_in_scope[0]
                num_levels = subdf[cat_col].nunique();
                if num_levels < 2: raise ValueError(f"Kat. '{cat_col}' < 2 úrovně.")
                groups = [group[num_col] for name, group in subdf.groupby(cat_col)]
                if num_levels == 2:
                    normality_p1 = run_shapiro(groups[0]); normality_p2 = run_shapiro(groups[1]); levene_p = run_levene(*groups)
                    normality_ok = normality_p1 > 0.05 and normality_p2 > 0.05; homogeneity_ok = levene_p > 0.05
                    reason += f"1Kat({num_levels})+1Num. Shapiro p=({normality_p1:.3f},{normality_p2:.3f}). Levene p={levene_p:.3f}. "
                    if normality_ok and homogeneity_ok: actual_method = AUTO_TTEST_UNPAIRED_EQUAL; reason += "-> t-test (nepár., shod. rozpt.)"
                    elif normality_ok and not homogeneity_ok: actual_method = AUTO_TTEST_UNPAIRED_WELCH; reason += "-> Welchův t-test (nepár., růz. rozpt.)"
                    else: actual_method = AUTO_MANNWHITNEY; reason += "-> Mann-Whitney U"
                else: # K > 2
                    normality_p_overall = run_shapiro(subdf[num_col]); levene_p = run_levene(*groups)
                    normality_ok = normality_p_overall > 0.05; homogeneity_ok = levene_p > 0.05
                    reason += f"1Kat({num_levels})+1Num. Shapiro(celk.) p={normality_p_overall:.3f}. Levene p={levene_p:.3f}. "
                    if normality_ok and homogeneity_ok: actual_method = AUTO_ANOVA; reason += "-> ANOVA"
                    else: actual_method = AUTO_KRUSKAL; reason += "-> Kruskal-Wallis"
            elif n_cat == 1 and n_num > 1: actual_method = AUTO_ANOVA; reason += f"1 Kat. + {n_num} Num. -> ANOVA (pro každý num. vs kat.)"
            elif n_cat == 0 and n_num == 2:
                num1_col, num2_col = num_cols_in_scope
                normality_p1 = run_shapiro(subdf[num1_col]); normality_p2 = run_shapiro(subdf[num2_col])
                normality_ok = normality_p1 > 0.05 and normality_p2 > 0.05
                reason += f"2 Num. Shapiro p=({normality_p1:.3f},{normality_p2:.3f}). Párová data={paired}. "
                if normality_ok:
                    actual_method = AUTO_TTEST_PAIRED if paired else "t.test.unpaired.auto" # Zkontroluje Levene později
                    reason += f"-> {('Párový' if paired else 'Nepárový')} t-test" + (" (Student/Welch dle Levene)" if not paired else "")
                else: actual_method = AUTO_WILCOXON if paired else AUTO_MANNWHITNEY; reason += f"-> {('Wilcoxonův párový' if paired else 'Mann-Whitney U')}"
            else: raise ValueError("Automatický výběr pro tuto kombinaci typů proměnných není podporován.")

            logger.info(f"AUTO selected method code: {actual_method}")
            test_output["reason"] = reason.strip()

        # --- Provedení vybraného testu ---
        test_output["test_name"] = actual_method # Skutečně použitý kód metody
        stat, p, dof = float('nan'), float('nan'), None # Inicializace pro testy vracející 1 výsledek

        if actual_method == "chi2":
            if n_cat < 2: raise ValueError("χ² test vyžaduje >= 2 kat.")
            cat1, cat2 = cat_cols_in_scope[0], cat_cols_in_scope[1]
            cont_table = pd.crosstab(subdf[cat1], subdf[cat2])
            if cont_table.size < 4 or cont_table.shape[0] < 2 or cont_table.shape[1] < 2: raise ValueError("Kontingenční tabulka < 2x2.")
            stat, p, dof, expected = chi2_contingency(cont_table)
            test_output["test_name"] = "Chi-squared (χ²)"; test_output["statistic"] = stat; test_output["p_value"] = p; test_output["degrees_freedom"] = dof; test_output["statistic_name"] = "χ²"; test_output["contingency_table"] = cont_table.to_dict()

        elif actual_method == "fisher":
             if n_cat != 2: raise ValueError("Fisher vyžaduje 2 kat.")
             cat1, cat2 = cat_cols_in_scope
             cont_table = pd.crosstab(subdf[cat1], subdf[cat2])
             if cont_table.shape != (2, 2):
                 logger.warning(f"Fisher byl zvolen pro ne-2x2 tabulku ({cat1} vs {cat2}). Používám Chi2.")
                 test_output["reason"] += " (Varování: Tabulka není 2x2, použit Chi2!)"
                 stat, p, dof, expected = chi2_contingency(cont_table)
                 test_output["test_name"] = "Chi-squared (χ² - fallback)"; test_output["statistic"] = stat; test_output["p_value"] = p; test_output["degrees_freedom"] = dof; test_output["statistic_name"] = "χ²"; test_output["contingency_table"] = cont_table.to_dict()
             else:
                 odds_ratio, p = fisher_exact(cont_table.values); test_output["test_name"] = "Fisherův přesný test"; test_output["statistic"] = odds_ratio; test_output["p_value"] = p; test_output["statistic_name"] = "Odds Ratio"; test_output["contingency_table"] = cont_table.to_dict()

        elif actual_method in ["anova", "kruskal"]:
             if n_cat < 1 or n_num < 1: raise ValueError(f"{actual_method.upper()} vyžaduje >=1 kat. a >=1 num.")
             results_list = []
             for cat_col in cat_cols_in_scope:
                 if subdf[cat_col].nunique() < 2: continue
                 groups_base = subdf.groupby(cat_col)
                 for num_col in num_cols_in_scope:
                     groups = [group[num_col].dropna() for name, group in groups_base if not group[num_col].dropna().empty]
                     if len(groups) < 2: continue
                     stat, p = float('nan'), float('nan')
                     try:
                         if actual_method == "anova": stat, p = f_oneway(*groups)
                         elif actual_method == "kruskal": stat, p = kruskal(*groups)
                     except ValueError as test_err: logger.warning(f"Test {actual_method} selhal pro {num_col} vs {cat_col}: {test_err}")
                     results_list.append({"cat_col": cat_col, "num_col": num_col, "statistic": stat if pd.notna(stat) else None, "p_value": p if pd.notna(p) else None})
             if not results_list: raise ValueError(f"{actual_method.upper()} nemohla být provedena.")
             test_output["test_name"] = "ANOVA" if actual_method == "anova" else "Kruskal-Wallis"; test_output["results"] = results_list

        elif actual_method.startswith("t.test"):
            if n_num != 2 or n_cat != 0: raise ValueError("t-test vyžaduje 2 num.")
            num1_col, num2_col = num_cols_in_scope
            num1 = subdf[num1_col]; num2 = subdf[num2_col]
            if actual_method == "t.test.paired":
                stat, p = ttest_rel(num1, num2, nan_policy='omit'); test_output["test_name"] = "Párový t-test"; test_output["statistic_name"] = "t"
            else: # Nepárové
                equal_var_flag = False # Default Welch
                if actual_method == "t.test.unpaired.equal": equal_var_flag = True
                elif actual_method == "t.test.unpaired.welch": equal_var_flag = False
                elif actual_method == "t.test.unpaired.auto": # Znovu Levene
                     levene_p = run_levene(num1, num2); equal_var_flag = levene_p > 0.05
                     test_output["reason"] += f" Levene p={levene_p:.3f}."
                elif method == "t.test": # Manuální volba 't.test'
                     levene_p = run_levene(num1, num2); equal_var_flag = levene_p > 0.05
                     test_output["reason"] += f" (Detekce rozptylů: Levene p={levene_p:.3f})."
                stat, p = ttest_ind(num1, num2, equal_var=equal_var_flag, nan_policy='omit')
                test_output["test_name"] = "Studentův t-test (nepárový)" if equal_var_flag else "Welchův t-test (nepárový)"; test_output["statistic_name"] = "t"
            test_output["statistic"] = stat; test_output["p_value"] = p

        elif actual_method == "wilcoxon":
            if n_num != 2 or n_cat != 0 or not paired: raise ValueError("Wilcoxon vyžaduje 2 num párové.")
            num1_col, num2_col = num_cols_in_scope; stat, p = wilcoxon(subdf[num1_col], subdf[num2_col], zero_method='zsplit', correction=True, mode='approx', nan_policy='omit')
            test_output["test_name"] = "Wilcoxonův párový test"; test_output["statistic"] = stat; test_output["p_value"] = p; test_output["statistic_name"] = "W"

        elif actual_method == "mannwhitney":
             if n_num == 2 and n_cat == 0 and not paired:
                 num1_col, num2_col = num_cols_in_scope; stat, p = mannwhitneyu(subdf[num1_col], subdf[num2_col], alternative='two-sided', nan_policy='omit')
                 test_output["test_name"] = "Mann-Whitney U test"
             elif n_num == 1 and n_cat == 1:
                  cat_col, num_col = cat_cols_in_scope[0], num_cols_in_scope[0];
                  if subdf[cat_col].nunique() != 2: raise ValueError("Mann-Whitney pro kat. vs num. vyžaduje 2 úrovně kat.")
                  groups = [group[num_col].dropna() for name, group in subdf.groupby(cat_col)]
                  if len(groups) != 2: raise ValueError("Nepodařilo se rozdělit do dvou skupin.")
                  stat, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided', nan_policy='omit')
                  test_output["test_name"] = "Mann-Whitney U test"
             else: raise ValueError("Neplatná kombinace pro Mann-Whitney U.")
             test_output["statistic"] = stat; test_output["p_value"] = p; test_output["statistic_name"] = "U"

        else:
             if method != "auto": raise HTTPException(status_code=400, detail=f"Metoda '{method}' není implementována.")
             else: raise NotImplementedError(f"Interní chyba: Metoda '{actual_method}' z 'auto' není implementována.")

        # --- Finální čištění NaN/Inf pro JSON ---
        final_output = robust_clean_nan_inf(test_output)
        logger.info("Dependency test finished successfully.")
        return final_output

    except (ValueError, KeyError) as user_err:
        logger.error(f"Error during dependency test: {user_err}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Chyba analýzy závislosti: {str(user_err)}")
    except HTTPException as http_err:
         logger.warning(f"HTTP Exception during dependency test: {http_err.detail}")
         raise http_err
    except Exception as e:
        logger.exception("--- UNEXPECTED ERROR during Dependency Test ---")
        raise HTTPException(status_code=500, detail=f"Neočekávaná interní chyba serveru při testu závislosti: {str(e)}")

# ... (zbytek souboru main.py) ...

class RegressionRequest(BaseModel):
    y: str
    x: List[str]
    method: Optional[str] = "auto"

# --- Helper funkce pro formátování (podobné R) ---
def format_p_value(p_val):
    if pd.isna(p_val):
        return '-'
    if p_val < 0.001:
        return "<0.001" # Jednodušší formátování než vědecká notace
    return f"{p_val:.3f}"

def format_coef(coef):
     if pd.isna(coef):
         return '-'
     # Zobrazí více míst pro malé koeficienty, méně pro velké
     if abs(coef) < 0.0001:
         return f"{coef:.4e}"
     elif abs(coef) < 1:
          return f"{coef:.4f}"
     else:
         return f"{coef:.3f}"


logging.basicConfig(level=logging.INFO) # Zobrazí INFO a vyšší (WARNING, ERROR, CRITICAL)
logger = logging.getLogger(__name__)
# Nastavení vyšší úrovně pro knihovny, aby nás nerušily, pokud nechceme
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("statsmodels").setLevel(logging.WARNING)

def robust_clean_nan_inf(value: Any) -> Any:
    """
    Rekurzivně čistí data pro JSON serializaci.
    Nahrazuje NaN/Inf hodnotami None.
    Převádí numpy číselné typy na standardní Python typy.
    Zachovává None.
    """
    if isinstance(value, dict):
        return {k: robust_clean_nan_inf(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [robust_clean_nan_inf(elem) for elem in value]
    elif isinstance(value, (np.ndarray, pd.Series)):
         cleaned_list = [robust_clean_nan_inf(elem) for elem in value.tolist()]
         return cleaned_list
    elif isinstance(value, (float, np.floating)):
        # Handles NaN and Inf specifically
        if pd.isna(value) or np.isinf(value): # Use pd.isna for broader check
            return None
        return float(value)
    elif isinstance(value, (int, np.integer)):
         # Handle potential pd.NA in integer columns converted by pandas > 1.0
         if pd.isna(value):
             return None
         return int(value)
    elif isinstance(value, (bool, np.bool_)):
        return bool(value)
    elif isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    elif value is None or pd.isna(value): # Catch None and pd.NA
        return None
    elif isinstance(value, str):
         return value
    # For other types supported by JSON (like standard Python numbers), return as is
    return value
    # --- Upravený Endpoint ---


@app.post("/api/regression_analysis")
async def regression_analysis(req: RegressionRequest):
    # --- Načtení dat a základní validace (stejné jako předtím) ---
    logger.info(f"Received regression request (statsmodels for OLS/Logit/MNLogit): Y='{req.y}', X={req.x}, Method='{req.method}'")
    global stored_datasets
    if "latest" not in stored_datasets: raise HTTPException(status_code=400, detail="Data nenahrána.")
    headers = stored_datasets["latest"].get("headers")
    data = stored_datasets["latest"].get("data")
    if not headers or not data: raise HTTPException(status_code=400, detail="Data nekompletní.")
    df = pd.DataFrame(data, columns=headers)
    logger.info(f"Loaded data. Shape: {df.shape}")
    y_col = req.y; x_cols = req.x; method = req.method.lower() if req.method else "auto"
    if y_col not in df.columns: raise HTTPException(status_code=400, detail=f"Y sloupec '{y_col}' nenalezen.")
    invalid_x = [col for col in x_cols if col not in df.columns];
    if invalid_x: raise HTTPException(status_code=400, detail=f"X sloupce {invalid_x} nenalezeny.")
    if y_col in x_cols: raise HTTPException(status_code=400, detail="Y nemůže být v X.")

    # --- Příprava dat (udržujeme X jako DataFrame) ---
    try:
        relevant_cols = [y_col] + x_cols
        df_subset = df[relevant_cols].replace('', np.nan).copy()
        for col in relevant_cols: df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
        initial_rows = len(df_subset); df_subset.dropna(inplace=True); final_rows = len(df_subset)
        logger.info(f"Data prep: Removed {initial_rows - final_rows} rows with NA. Final rows: {final_rows}")
        if final_rows < len(relevant_cols) + 1: raise ValueError(f"Nedostatek dat ({final_rows}).")

        y_pd = df_subset[y_col]
        X_pd = df_subset[x_cols]
        y_unique_prepared = y_pd.unique()
        if len(y_unique_prepared) < 2: raise ValueError("Y má méně než 2 unikátní hodnoty.")

        # Detekce typu Y a výběr metody 'auto' (stejná logika jako předtím)
        y_is_numeric = pd.api.types.is_numeric_dtype(y_pd.dtype)
        y_is_likely_binary = len(y_unique_prepared) == 2
        y_is_likely_multicat = (not y_is_numeric or pd.api.types.is_integer_dtype(y_pd.dtype)) and \
                               len(y_unique_prepared) > 2 and len(y_unique_prepared) <= 15 # Prah zůstává
        selected_method = method; reason = ""
        if method == "auto":
            if y_is_likely_binary: selected_method = "logistic"; reason = "Auto: Y má 2 úrovně -> Logistická (Logit)."
            elif y_is_likely_multicat: selected_method = "multinomial"; reason = "Auto: Y kat. (>2 úrovně) -> Multinomiální (MNLogit)."
            elif y_is_numeric:
                 n_samples, n_features = X_pd.shape; strong_corr = False
                 if n_features > 1:
                      try:
                          corr_matrix = X_pd.corr().abs()
                          upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                          strong_corr = (upper_tri > 0.7).any().any()
                      except Exception: pass
                 if n_features >= n_samples: selected_method = "ridge"; reason = "Auto: p >= n -> Ridge."
                 elif strong_corr: selected_method = "ridge"; reason = "Auto: Silná korelace X -> Ridge."
                 else: selected_method = "ols"; reason = "Auto: Standardní numerické Y -> OLS."
            else: raise ValueError(f"Nelze auto. určit metodu pro Y typu {y_pd.dtype}.")
        else: # Manuální volba + validace
             if method == "logistic" and not y_is_likely_binary: raise HTTPException(status_code=400, detail="Logistická vyžaduje Y se 2 úrovněmi.")
             if method == "multinomial" and not y_is_likely_multicat:
                 if not pd.api.types.is_string_dtype(y_pd.dtype) and not pd.api.types.is_object_dtype(y_pd.dtype):
                      raise HTTPException(status_code=400, detail="Multinomiální vyžaduje kategorické Y.")
             if method in ["ols", "ridge", "lasso", "elasticnet"] and not y_is_numeric: raise HTTPException(status_code=400, detail=f"Metoda '{method}' vyžaduje numerické Y.")
             selected_method = method
             reason = f"Metoda zvolena uživatelem: {method.upper()}"

        logger.info(f"Selected method: {selected_method}. Reason: {reason}")
        model_results = {"method": selected_method, "reason": reason}

    except (ValueError, KeyError) as prep_err:
         logger.error(f"Error during data preparation: {prep_err}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Chyba při přípravě dat: {str(prep_err)}")
    except Exception as e:
         logger.exception("--- UNEXPECTED ERROR DURING DATA PREPARATION ---")
         raise HTTPException(status_code=500, detail=f"Neočekávaná chyba serveru při přípravě dat: {str(e)}")

    # --- Fitování a Zpracování Výsledků ---
    try:
        # --- OLS pomocí STATSMODELS ---
        if selected_method == "ols":
            logger.info("Using statsmodels.OLS for analysis.")
            y_sm = y_pd.astype(float)
            X_sm_df = X_pd.astype(float)
            X_sm = sm.add_constant(X_sm_df, has_constant='raise')
            model = sm.OLS(y_sm, X_sm)
            results = model.fit()
            logger.info("Statsmodels OLS model fitted.")

            # Extrakce metrik a statistik (jako čísla)
            model_results["r2"] = getattr(results, 'rsquared', None)
            model_results["r2_adjusted"] = getattr(results, 'rsquared_adj', None)
            model_results["mse"] = getattr(results, 'mse_resid', None)
            model_results["rmse"] = np.sqrt(model_results["mse"]) if pd.notna(model_results["mse"]) and model_results["mse"] >= 0 else None
            model_results["f_statistic"] = getattr(results, 'fvalue', None)
            model_results["f_pvalue"] = getattr(results, 'f_pvalue', None) # p-hodnota F-testu
            model_results["n_observations"] = int(getattr(results, 'nobs', 0))
            model_results["intercept"] = results.params.get('const')
            model_results["accuracy"] = None # Není relevantní

            coeffs_list = []
            params = results.params.drop('const', errors='ignore')
            p_values = results.pvalues.drop('const', errors='ignore')
            try: conf_int_df = results.conf_int()
            except Exception: conf_int_df = pd.DataFrame(index=params.index, columns=[0, 1], data=np.nan)
            bse = results.bse.drop('const', errors='ignore') # Std. Error
            tvalues = results.tvalues.drop('const', errors='ignore') # t-value

            for name in params.index:
                coeffs_list.append({
                    "name": name,
                    "coef": params.get(name),
                    "stderr": bse.get(name),
                    "t_value": tvalues.get(name),
                    "p_value": p_values.get(name),
                    "ciLow": conf_int_df.loc[name, 0] if name in conf_int_df.index else None,
                    "ciHigh": conf_int_df.loc[name, 1] if name in conf_int_df.index else None,
                })
            model_results["coefficients"] = coeffs_list

            # Data pro grafy
            y_pred = getattr(results, 'fittedvalues', None)
            residuals_val = getattr(results, 'resid', None)
            X_aligned = X_pd # X už je DataFrame se správným indexem
            if X_aligned.shape[1] == 1 and y_pred is not None and not y_pred.empty:
                 model_results["scatter_data"] = {"x": X_aligned.iloc[:, 0].tolist(), "y_true": y_sm.tolist(), "y_pred": y_pred.tolist()}
            else: model_results["scatter_data"] = None
            if y_pred is not None and not y_pred.empty and residuals_val is not None and not residuals_val.empty:
                 model_results["residuals"] = {"predicted": y_pred.tolist(), "residuals": residuals_val.tolist()}
            else: model_results["residuals"] = None

            model_results["note"] = "Výsledky z OLS (statsmodels)."

        # --- LOGISTICKÁ REGRESE (binární) pomocí STATSMODELS ---
        elif selected_method == "logistic":
            logger.info("Using statsmodels.Logit for analysis.")
            # Y musí být numerické (0/1), pokud není, zkusíme převést
            if not pd.api.types.is_numeric_dtype(y_pd.dtype):
                 # Pokusí se převést např. True/False nebo 'Ano'/'Ne' na 0/1
                 y_pd, _ = pd.factorize(y_pd) # Vrací kódy a unikátní hodnoty
                 if len(_) != 2: raise ValueError("Logistická regrese vyžaduje přesně 2 kategorie v Y.")
                 y_sm = pd.Series(y_pd, index=X_pd.index).astype(int) # Zajistíme int a zachováme index
                 logger.info(f"Y převedeno na kódy 0/1 pro Logit.")
            else:
                 y_sm = y_pd.astype(int) # Zajistíme integer typ

            X_sm_df = X_pd.astype(float)
            X_sm = sm.add_constant(X_sm_df, has_constant='raise')

            try:
                model = Logit(y_sm, X_sm)
                results = model.fit(method='newton') # Běžný solver pro Logit
                logger.info("Statsmodels Logit model fitted.")
            except PerfectSeparationError:
                 logger.error("Perfect separation detected during Logit fit.")
                 raise HTTPException(status_code=400, detail="Chyba: Nastala perfektní separace dat. Logistická regrese nemůže být spolehlivě odhadnuta. Zkontrolujte, zda některá nezávislá proměnná dokonale nepredikuje výsledek.")
            except Exception as fit_err:
                 logger.error(f"Error fitting Logit model: {fit_err}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Chyba při fitování Logit modelu: {fit_err}")


            # Extrakce metrik a statistik (jako čísla)
            model_results["pseudo_r2"] = getattr(results, 'prsquared', None) # Pseudo R^2
            model_results["log_likelihood"] = getattr(results, 'llf', None)
            model_results["llr_p_value"] = getattr(results, 'llr_pvalue', None) # P-hodnota Likelihood Ratio testu
            model_results["n_observations"] = int(getattr(results, 'nobs', 0))
            model_results["intercept"] = results.params.get('const')
            model_results["r2"]=None; model_results["r2_adjusted"]=None; model_results["mse"]=None; model_results["rmse"]=None; model_results["f_statistic"]=None; model_results["f_pvalue"]=None;

            # Výpočet Accuracy (statsmodels ji přímo nedává)
            y_pred_prob = results.predict(X_sm)
            y_pred_class = (y_pred_prob > 0.5).astype(int) # Klasifikace podle prahu 0.5
            model_results["accuracy"] = accuracy_score(y_sm, y_pred_class)

            # Extrakce koeficientů a statistik (jako čísla)
            coeffs_list = []
            params = results.params.drop('const', errors='ignore')
            p_values = results.pvalues.drop('const', errors='ignore')
            try: conf_int_df = results.conf_int() # CI pro log-odds
            except Exception: conf_int_df = pd.DataFrame(index=params.index, columns=[0, 1], data=np.nan)
            bse = results.bse.drop('const', errors='ignore') # Std. Error
            zvalues = results.tvalues.drop('const', errors='ignore') # Z-values (pojmenováno tvalues)

            for name in params.index:
                coeffs_list.append({
                    "name": name,
                    "coef": params.get(name),    # Log-odds
                    "stderr": bse.get(name),
                    "z_value": zvalues.get(name), # Přejmenováno pro srozumitelnost
                    "p_value": p_values.get(name),
                    "ciLow": conf_int_df.loc[name, 0] if name in conf_int_df.index else None,
                    "ciHigh": conf_int_df.loc[name, 1] if name in conf_int_df.index else None,
                })
            model_results["coefficients"] = coeffs_list
            model_results["scatter_data"] = None # Scatter není vhodný
            model_results["residuals"] = None    # Rezidua nejsou standardní
            model_results["note"] = "Výsledky z Logistické regrese (statsmodels Logit). Koeficienty jsou log-odds."

        # --- MULTINOMIÁLNÍ REGRESE pomocí STATSMODELS ---
        elif selected_method == "multinomial":
            logger.info("Using statsmodels.MNLogit for analysis.")
            # Y musí být numerické kategorie (0, 1, 2...)
            y_codes, y_categories = pd.factorize(y_pd)
            y_mn = pd.Series(y_codes, index=X_pd.index).astype(int)
            logger.info(f"Y převedeno na kódy 0-{len(y_categories)-1} pro MNLogit. Kategorie: {list(y_categories)}")

            X_sm_df = X_pd.astype(float)
            X_sm = sm.add_constant(X_sm_df, has_constant='raise')

            try:
                model = MNLogit(y_mn, X_sm)
                # 'bfgs' nebo 'newton' jsou běžné, 'nm' může být pomalejší
                results = model.fit(method='bfgs', maxiter=300) # Zvýšení iterací
                logger.info("Statsmodels MNLogit model fitted.")
            except Exception as fit_err:
                 logger.error(f"Error fitting MNLogit model: {fit_err}", exc_info=True)
                 # Může selhat kvůli konvergenci, separaci atd.
                 raise HTTPException(status_code=500, detail=f"Chyba při fitování MNLogit modelu: {fit_err}")

            # Extrakce metrik a statistik
            model_results["pseudo_r2"] = getattr(results, 'prsquared', None)
            model_results["log_likelihood"] = getattr(results, 'llf', None)
            model_results["llr_p_value"] = getattr(results, 'llr_pvalue', None)
            model_results["n_observations"] = int(getattr(results, 'nobs', 0))
            # Intercept a koeficienty jsou složitější
            model_results["intercept"] = None # MNLogit nemá jeden intercept
            model_results["r2"]=None; model_results["r2_adjusted"]=None; model_results["mse"]=None; model_results["rmse"]=None; model_results["f_statistic"]=None; model_results["f_pvalue"]=None;

            # Výpočet Accuracy
            y_pred_prob = results.predict(X_sm) # Vrací pravděpodobnosti pro každou třídu
            y_pred_class_idx = np.argmax(y_pred_prob.values, axis=1) # Index třídy s nejvyšší P
            model_results["accuracy"] = accuracy_score(y_mn, y_pred_class_idx)

            # Extrakce koeficientů (params je DataFrame: index=features, columns=classes)
            # A statistik (pvalues, bse, tvalues/zvalues jsou také DataFrames)
            coeffs_list = []
            params_df = results.params # DataFrame (features+const x k-1 classes)
            pvalues_df = results.pvalues
            bse_df = results.bse
            zvalues_df = results.tvalues # Z-values

            # Referenční třída je první (index 0), ostatní jsou porovnány vůči ní
            # Sloupce v params_df jsou názvy ostatních tříd (kategorií)
            feature_names = X_sm.columns.drop('const', errors='ignore') # Názvy X proměnných

            for class_idx_str in params_df.columns: # Iterujeme přes cílové třídy (sloupce)
                try:
                    # Získáme původní název kategorie z indexu
                    class_idx = int(class_idx_str) # Sloupce jsou často čísla 1, 2,...
                    class_name = y_categories[class_idx] # Název kategorie
                except (ValueError, IndexError):
                     class_name = class_idx_str # Fallback na index jako string

                for feature_name in feature_names: # Iterujeme přes proměnné (řádky)
                    if feature_name in params_df.index: # Jistota
                         coeffs_list.append({
                             "name": f"{feature_name} (třída: {class_name})", # Název s třídou
                             "coef": params_df.loc[feature_name, class_idx_str],
                             "stderr": bse_df.loc[feature_name, class_idx_str] if feature_name in bse_df.index else None,
                             "z_value": zvalues_df.loc[feature_name, class_idx_str] if feature_name in zvalues_df.index else None,
                             "p_value": pvalues_df.loc[feature_name, class_idx_str] if feature_name in pvalues_df.index else None,
                             "ciLow": None, # CI pro MNLogit nejsou přímo v summary
                             "ciHigh": None,
                         })

            model_results["coefficients"] = coeffs_list
            model_results["scatter_data"] = None
            model_results["residuals"] = None
            model_results["note"] = f"Výsledky z Multinomiální regrese (statsmodels MNLogit). Koeficienty jsou log-odds pro danou třídu vs. referenční třídu '{y_categories[0]}'."


        # --- Ridge, Lasso, ElasticNet pomocí SKLEARN (zůstává stejné) ---
        elif selected_method in ["ridge", "lasso", "elasticnet"]:
            logger.info(f"Using sklearn {selected_method.upper()} for analysis.")
            scaler = StandardScaler()
            # ---> ZMĚNA: Používáme X_pd (DataFrame) pro select_dtypes <---
            X_numeric_df = X_pd.select_dtypes(include=np.number).astype(float)
            if X_numeric_df.shape[1] == 0: raise ValueError(f"Žádné numerické X pro {selected_method}.")
            if X_numeric_df.shape[1] < X_pd.shape[1]: logger.warning(f"Nenumerické sloupce vypuštěny pro {selected_method}.")

            X_scaled = scaler.fit_transform(X_numeric_df.values) # Fit na numpy array
            y_numeric = y_pd.astype(float).values # y jako numpy array

            alpha_val = 1.0
            if selected_method == "ridge": model = Ridge(alpha=alpha_val, random_state=42)
            elif selected_method == "lasso": model = Lasso(alpha=alpha_val, random_state=42)
            else: model = ElasticNet(alpha=alpha_val, random_state=42)

            model.fit(X_scaled, y_numeric)
            y_pred = model.predict(X_scaled)
            logger.info(f"Sklearn {selected_method.upper()} fitted.")

            # Extrakce výsledků (bez detailních statistik)
            model_results["intercept"] = float(model.intercept_)
            model_results["coefficients"] = [
                {"name": name, "coef": float(coef), "stderr": None, "t_value": None, "p_value": None, "ciLow": None, "ciHigh": None}
                for name, coef in zip(X_numeric_df.columns, model.coef_) # Použijeme sloupce z X_numeric_df
            ]
            model_results["r2"] = r2_score(y_numeric, y_pred)
            model_results["mse"] = mean_squared_error(y_numeric, y_pred)
            model_results["rmse"] = np.sqrt(model_results["mse"]) if model_results["mse"] >=0 else None
            model_results["n_observations"] = len(y_numeric)
            model_results["r2_adjusted"] = None; model_results["f_statistic"] = None; model_results["f_pvalue"]=None; model_results["accuracy"] = None

            # Data pro grafy
            residuals_val = y_numeric - y_pred
            if X_numeric_df.shape[1] == 1:
                 x_scatter = X_numeric_df.iloc[:, 0] # Původní X (DataFrame)
                 if len(x_scatter) == len(y_numeric) == len(y_pred):
                      model_results["scatter_data"] = {"x": x_scatter.tolist(), "y_true": y_numeric.tolist(), "y_pred": y_pred.tolist()}
                 else: model_results["scatter_data"] = None
            else: model_results["scatter_data"] = None
            if len(y_pred) == len(residuals_val):
                 model_results["residuals"] = {"predicted": y_pred.tolist(), "residuals": residuals_val.tolist()}
            else: model_results["residuals"] = None

            model_results["note"] = f"Výsledky z {selected_method.upper()} (sklearn). Koeficienty pro škálovaná X."

        # --- Neznámá metoda ---
        else:
            raise NotImplementedError(f"Metoda '{selected_method}' není implementována.")


        # --- Finální Čištění Výsledků pro JSON ---
        logger.info("Cleaning final results for JSON serialization...")
        final_results = robust_clean_nan_inf(model_results)

        logger.info("Regression analysis successful, returning results.")
        return final_results

    except (ValueError, NotImplementedError, MemoryError, HTTPException) as user_err:
        logger.error(f"Error during analysis ({selected_method}): {user_err}", exc_info=isinstance(user_err, ValueError))
        if isinstance(user_err, HTTPException): raise user_err
        raise HTTPException(status_code=400, detail=f"Chyba analýzy ({selected_method}): {str(user_err)}")
    except Exception as e:
        logger.exception(f"--- UNEXPECTED ERROR in Regression Analysis (Method: {selected_method}) ---")
        raise HTTPException(status_code=500, detail=f"Interní chyba serveru při výpočtu regrese ({selected_method}): {str(e)}")

# ... (importy jako dříve: FastAPI, HTTPException, pandas, atd.)
# Přidejte potřebné importy pro porovnání skupin
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, f_oneway, kruskal, levene # Přidán levene


# ... (definice app, stored_datasets, funkce check_normality jako dříve)
# ... (Endpoint /api/group_comparison zůstává)


# --- Nové Pydantic modely pro AI interpretaci porovnání skupin ---

class TestResultInterpretation(BaseModel):
    """Zjednodušená data o jednom testu pro AI"""
    numeric_variable: str
    test_name: str
    p_value: float
    is_significant: bool
    notes: Optional[str] = None # Důvod výběru testu

class GroupComparisonInterpretation(BaseModel):
    """Data o porovnání pro jednu skupinovou proměnnou"""
    group_variable: str
    tests_performed: List[TestResultInterpretation]

class GroupComparisonInterpretationRequest(BaseModel):
    analysis_type: str = "group_comparison"
    paired_analysis: bool
    comparisons: List[GroupComparisonInterpretation]


# --- Nový endpoint pro AI interpretaci porovnání skupin ---
@app.post("/api/interpret_group_comparison")
async def interpret_group_comparison(req: GroupComparisonInterpretationRequest):
    # --- System Prompt pro LLM ---
    system_prompt = (
        "Jsi AI asistent specializující se na analýzu dat. Uživatel provedl porovnání skupin a poskytne ti výsledky.\n\n"
        "Tvým úkolem je interpretovat tyto výsledky v češtině, jednoduchým a srozumitelným jazykem.\n\n"
        f"Jednalo se o {'párovou analýzu (porovnání dvou měření u stejných subjektů)' if req.paired_analysis else 'analýzu nezávislých skupin'}.\n\n"
        "Pro každý provedený test:\n"
        "- Uveď, která číselná proměnná byla porovnávána mezi skupinami definovanými kterou kategoriální proměnnou.\n"
        "- Zmiň použitý test (např. t-test, ANOVA, Mann-Whitney U, Wilcoxon, Kruskal-Wallis).\n"
        "- **Interpretuj p-hodnotu:**\n"
        "  - Pokud je p < 0.05, výsledek je **statisticky významný**. Popiš to jako 'Byl nalezen statisticky významný rozdíl v [číselná proměnná] mezi skupinami [kategoriální proměnná].' U párového testu: 'Byl nalezen statisticky významný rozdíl mezi [číselná proměnná 1] a [číselná proměnná 2].'\n"
        "  - Pokud je p >= 0.05, výsledek **není statisticky významný**. Popiš to jako 'Nebyl nalezen statisticky významný rozdíl...'.\n"
        "- Můžeš stručně zmínit důvod výběru testu, pokud je uveden v poznámce (např. kvůli normalitě dat, počtu skupin, párování).\n\n"
        "**Celkové shrnutí:**\n"
        "- Shrnout nejdůležitější (významné) rozdíly, které byly nalezeny.\n"
        "- Pokud nebyly nalezeny žádné významné rozdíly, konstatuj to.\n\n"
        "**Důležité upozornění:**\n"
        "- Připomeň, že statistická významnost (p < 0.05) neznamená automaticky velký nebo prakticky důležitý rozdíl (velikost efektu zde není hodnocena).\n"
        "- U nezávislých testů (ANOVA, Kruskal-Wallis) významný výsledek říká, že existuje rozdíl *někde* mezi skupinami, ale neříká *mezi kterými konkrétně* (k tomu by byly potřeba post-hoc testy).\n\n"
        "Pravidla:\n"
        "- Odpovídej v češtině.\n"
        "- Buď jasný a srozumitelný.\n"
        "- Zaměř se na interpretaci p-hodnoty v kontextu porovnání skupin.\n"
        "- Formátuj odpověď pro dobrou čitelnost (odstavce, body)."
    )

    # --- Sestavení User Promptu ---
    user_prompt_parts = [
        f"Provedl jsem analýzu typu '{req.analysis_type}' ({'párová' if req.paired_analysis else 'nezávislé skupiny'}).",
        "\nZde jsou výsledky jednotlivých porovnání:"
    ]

    for comp in req.comparisons:
        user_prompt_parts.append(f"\nPorovnání podle skupinové proměnné: **{comp.group_variable}**")
        for test in comp.tests_performed:
            significance = '(významné)' if test.is_significant else '(nevýznamné)'
            note_text = f" (Pozn.: {test.notes})" if test.notes else ""
            user_prompt_parts.append(
                f"- Proměnná '{test.numeric_variable}': Test='{test.test_name}', p-hodnota={test.p_value:.4f} {significance}{note_text}"
            )

    user_prompt_parts.append("\nProsím, interpretuj tyto výsledky.")
    full_user_prompt = "\n".join(user_prompt_parts)

    # --- Volání LLM API ---
    try:
        api_key = "OPENROUTER_API_KEY"
        if not api_key:
             raise HTTPException(status_code=500, detail="Chybí konfigurace API klíče pro LLM.")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free", # Nebo jiný model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                "max_tokens": 800 # Může být potřeba více pro více testů
            },
            timeout=90 # Delší timeout pro potenciálně více volání
        )
        response.raise_for_status()

        llm_data = response.json()
        interpretation_text = llm_data.get("choices", [{}])[0].get("message", {}).get("content")

        if not interpretation_text:
             raise HTTPException(status_code=500, detail="AI nevrátila platnou interpretaci.")

        return {"interpretation": interpretation_text}

    except requests.exceptions.RequestException as req_err:
        print(f"Chyba komunikace s LLM API: {req_err}")
        raise HTTPException(status_code=503, detail=f"Chyba při komunikaci s AI službou: {req_err}")
    except Exception as e:
        print(f"Neočekávaná chyba při interpretaci porovnání skupin: {e}")
        # print(f"User prompt was: {full_user_prompt}") # Pro debugging
        raise HTTPException(status_code=500, detail=f"Interní chyba serveru při generování interpretace: {str(e)}")

# ... (zbytek vaší FastAPI aplikace, včetně check_normality, pokud ji používáte)
# Ujistěte se, že check_normality je definovaná a dostupná

async def check_normality():
    # Implementace vaší funkce check_normality zde...
    # Měla by vrátit strukturu podobnou:
    # return {"results": [{"column": "col_name", "isNormal": True/False}, ...]}
    # Pokud ji nemáte, budete muset upravit logiku v /api/group_comparison
    # nebo ji implementovat. Prozatím vrátím prázdný placeholder.
    print("VAROVÁNÍ: Funkce check_normality není plně implementována v tomto příkladu.")
    # Načtěte data, proveďte Shapiro-Wilk test pro každý číselný sloupec
    # a vraťte výsledky. Příklad:
    if "latest" not in stored_datasets: return {"results": []}
    df = pd.DataFrame(stored_datasets["latest"]["data"], columns=stored_datasets["latest"]["headers"])
    num_cols = [c["name"] for c in stored_datasets["latest"].get("column_types",{}).values() if c["type"] == "Číselný"]
    normality_results = []
    for col in num_cols:
        try:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) >= 3:
                stat, p_value = shapiro(series)
                normality_results.append({"column": col, "isNormal": bool(p_value > 0.05)})
            else:
                normality_results.append({"column": col, "isNormal": False}) # Málo dat pro test
        except Exception:
                normality_results.append({"column": col, "isNormal": False}) # Chyba při zpracování
    return {"results": normality_results}




class CorrelationPairInterpretation(BaseModel):
    """Zjednodušená data o páru pro AI interpretaci"""
    var1: str
    var2: str
    correlation: float
    pValue: float
    significant: bool
    strength: str
class CorrelationInterpretationRequest(BaseModel):
    analysis_type: str = "correlation"
    method: str
    variables: List[str]
    correlation_pairs: List[CorrelationPairInterpretation]
    visualization_type: str # "matrix" nebo "scatterplot"




@app.post("/api/interpret_correlation")
async def interpret_correlation(req: CorrelationInterpretationRequest):
    # --- System Prompt pro LLM (interpretace korelace) ---
    system_prompt = (
        "Jsi AI asistent specializující se na analýzu dat. Uživatel provedl korelační analýzu a poskytne ti její výsledky.\n\n"
        "Tvým úkolem je interpretovat tyto výsledky v češtině, jednoduchým a srozumitelným jazykem pro někoho, kdo nemusí být expert na statistiku.\n\n"
        f"Použitá metoda byla '{req.method}'. Stručně zmiň, pro jaký typ vztahu je tato metoda vhodná (Pearson = lineární, Spearman/Kendall = monotónní/pořadový).\n\n"
        "Pro každý významný pár (significant = true):\n"
        "- Uveď proměnné.\n"
        "- Popiš sílu a směr vztahu (použij hodnotu 'correlation' a 'strength'). Např. 'Silná pozitivní korelace (r=0.8) naznačuje, že když roste X, roste i Y.' nebo 'Střední negativní korelace (r=-0.4) naznačuje, že když roste X, Y má tendenci klesat.'\n"
        "- Zmiň, že vztah je statisticky významný (p < 0.05), což znamená, že je nepravděpodobné, že by šlo o náhodu.\n\n"
        "Pro nevýznamné páry (significant = false):\n"
        "- Můžeš je zmínit souhrnně nebo vynechat, pokud jich je mnoho. Uveď, že mezi nimi nebyl nalezen statisticky významný vztah.\n\n"
        "Celkové shrnutí:\n"
        "- Pokud bylo analyzováno více než 2 proměnné (visualization_type = 'matrix'), shrň nejdůležitější (nejsilnější významné) nalezené vztahy.\n"
        "- Pokud byl jen jeden pár (visualization_type = 'scatterplot'), shrň výsledek pro tento pár.\n"
        "- **Důležité:** Zdůrazni, že **korelace neznamená kauzalitu** (to, že dvě věci spolu souvisí, neznamená, že jedna způsobuje druhou).\n\n"
        "Pravidla:\n"
        "- Odpovídej v češtině.\n"
        "- Buď stručný a věcný, ale srozumitelný.\n"
        "- Nepoužívej příliš technický žargon, pokud to není nutné (vysvětli p-hodnotu jednoduše).\n"
        "- Neuváděj vzorce ani kód.\n"
        "- Formátuj odpověď do odstavců nebo bodů pro lepší čitelnost.\n"
        "- Zaměř se na interpretaci, ne na opakování číselných hodnot (kromě např.  pro ilustraci)."
    )

    # --- Sestavení User Promptu z dat od frontendu ---
    user_prompt_parts = [
        f"Provedl jsem korelační analýzu ('{req.analysis_type}') metodou '{req.method}' pro proměnné: {', '.join(req.variables)}.",
        f"Vizualizace byla typu: {'maticová (více proměnných)' if req.visualization_type == 'matrix' else 'bodový graf (dvě proměnné)'}.",
        "\nZde jsou výsledky pro jednotlivé páry:"
    ]

    significant_pairs = [p for p in req.correlation_pairs if p.significant]
    non_significant_pairs = [p for p in req.correlation_pairs if not p.significant]

    if significant_pairs:
        user_prompt_parts.append("\nStatisticky významné vztahy (p < 0.05):")
        for pair in significant_pairs:
            user_prompt_parts.append(
                f"- {pair.var1} a {pair.var2}: Korelace r={pair.correlation:.3f} (Síla: {pair.strength}), p-hodnota={pair.pValue:.4f}"
            )
    else:
        user_prompt_parts.append("\nNebyly nalezeny žádné statisticky významné vztahy (p < 0.05).")

    if non_significant_pairs and len(significant_pairs) < len(req.correlation_pairs): # Zmíníme jen pokud existují a nejsou všechny významné
         user_prompt_parts.append(f"\nMezi {len(non_significant_pairs)} dalšími páry nebyl nalezen statisticky významný vztah (p >= 0.05).")


    user_prompt_parts.append("\nProsím, interpretuj tyto výsledky.")
    full_user_prompt = "\n".join(user_prompt_parts)

    # --- Volání LLM API (stejné jako u regrese) ---
    try:
        api_key = "OPENROUTER_API_KEY"
        if not api_key:
             raise HTTPException(status_code=500, detail="Chybí konfigurace API klíče pro LLM.")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free", # Nebo jiný model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                "max_tokens": 600 # Možná potřeba více pro více párů
            },
            timeout=60
        )
        response.raise_for_status() # Chyba pro 4xx/5xx

        llm_data = response.json()
        interpretation_text = llm_data.get("choices", [{}])[0].get("message", {}).get("content")

        if not interpretation_text:
             raise HTTPException(status_code=500, detail="AI nevrátila platnou interpretaci.")

        return {"interpretation": interpretation_text}

    except requests.exceptions.RequestException as req_err:
        print(f"Chyba komunikace s LLM API: {req_err}")
        raise HTTPException(status_code=503, detail=f"Chyba při komunikaci s AI službou: {req_err}")
    except Exception as e:
        print(f"Neočekávaná chyba při interpretaci korelace: {e}")
        # print(f"User prompt was: {full_user_prompt}") # Pro debugging
        raise HTTPException(status_code=500, detail=f"Interní chyba serveru při generování interpretace: {str(e)}")

class CorrelationRequest(BaseModel):
    columns: List[str]
    method: str  # 'auto', 'pearson', 'spearman', 'kendall'

@app.post("/api/correlation_analysis")
async def correlation_analysis(req: CorrelationRequest):
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    headers = stored_datasets["latest"]["headers"]
    data = stored_datasets["latest"]["data"]
    df = pd.DataFrame(data, columns=headers)

    selected_cols = req.columns
    method = req.method.lower()

    if len(selected_cols) < 2:
        raise HTTPException(status_code=400, detail="Je třeba vybrat alespoň dvě proměnné.")

    try:
        subdf = df[selected_cols].apply(pd.to_numeric, errors='coerce').dropna()
        sample_size = len(subdf)

        used_method = method
        reason = ""

        if method == "auto":
            # Pokud nejsou informace o normalitě, dopočítej je
            if "normality" not in stored_datasets["latest"]:
                normality = {}
                for col in selected_cols:
                    series = subdf[col].dropna()
                    if len(series) >= 3:
                        stat, p_value = shapiro(series)
                        normality[col] = bool(p_value > 0.05)
                    else:
                        normality[col] = False
                stored_datasets["latest"]["normality"] = normality
            else:
                normality = stored_datasets["latest"]["normality"]

            use_pearson = True
            for col in selected_cols:
                if not normality.get(col, False):
                    use_pearson = False
                    break

            reason = "Test 'auto': zvolena metoda na základě normality." + (
                " Všechny proměnné jsou normální → Pearson." if use_pearson else " Některé proměnné nejsou normální → Spearman.")
            used_method = "pearson" if use_pearson else "spearman"

        if used_method not in ["pearson", "spearman", "kendall"]:
            raise HTTPException(status_code=400, detail="Neplatná metoda korelace.")

        n = len(selected_cols)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        p_values = [[0.0 for _ in range(n)] for _ in range(n)]
        results = []

        def interpret_strength(r):
            abs_r = abs(r)
            if abs_r < 0.1:
                return "Žádná/slabá"
            elif abs_r < 0.3:
                return "Slabá"
            elif abs_r < 0.5:
                return "Střední"
            elif abs_r < 0.7:
                return "Silná"
            else:
                return "Velmi silná"

        def fisher_confidence_interval(r, n, alpha=0.05):
            if n < 4 or abs(r) >= 1.0:
                return (None, None)
            z = 0.5 * math.log((1 + r) / (1 - r))
            se = 1 / math.sqrt(n - 3)
            z_crit = norm.ppf(1 - alpha / 2)
            z_low = z - z_crit * se
            z_high = z + z_crit * se
            r_low = (math.exp(2 * z_low) - 1) / (math.exp(2 * z_low) + 1)
            r_high = (math.exp(2 * z_high) - 1) / (math.exp(2 * z_high) + 1)
            return (r_low, r_high)

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                    p_values[i][j] = 0.0
                elif j < i:
                    matrix[i][j] = matrix[j][i]
                    p_values[i][j] = p_values[j][i]
                else:
                    x = subdf[selected_cols[i]]
                    y = subdf[selected_cols[j]]

                    if used_method == "pearson":
                        corr, p = pearsonr(x, y)
                    elif used_method == "spearman":
                        corr, p = spearmanr(x, y)
                    elif used_method == "kendall":
                        corr, p = kendalltau(x, y)

                    matrix[i][j] = corr
                    p_values[i][j] = p

                    r2 = corr ** 2
                    ci_low, ci_high = fisher_confidence_interval(corr, sample_size)

                    results.append({
                        "var1": selected_cols[i],
                        "var2": selected_cols[j],
                        "correlation": float(corr),
                        "pValue": float(p),
                        "rSquared": float(r2),
                        "ciLow": ci_low,
                        "ciHigh": ci_high,
                        "strength": interpret_strength(corr),
                        "significant": bool(p < 0.05)
                    })

        scatter_data = None
        if len(selected_cols) == 2:
            scatter_data = {
                "x": subdf[selected_cols[0]].tolist(),
                "y": subdf[selected_cols[1]].tolist(),
                "xLabel": selected_cols[0],
                "yLabel": selected_cols[1]
            }

        return {
            "columns": selected_cols,
            "matrix": matrix,
            "pValues": p_values,
            "method": used_method,
            "reason": reason,
            "results": results,
            "scatterData": scatter_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při výpočtu korelace: {str(e)}")


@app.options("/api/analyze")
async def options_analyze():
    return {"message": "CORS preflight OK"}

class DataRequest(BaseModel):
    headers: List[str]
    data: List[List[Optional[Union[str, float, int]]]]
# 🔹 OPRAVA: Třída `DataRequest` byla zdvojena, upraveno na správnou verzi

# Globální cache pro uložená data
stored_datasets = {}


@app.post("/api/store_data")
async def store_data(request: Request):
    try:
        body = await request.json()
        headers = body.get("headers")
        data = body.get("data")

        if not headers or not data:
            raise HTTPException(status_code=400, detail="Missing headers or data")

        stored_datasets["latest"] = {
            "headers": headers,
            "data": data
        }

        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PromptRequest(BaseModel):
    prompt: str




class ClusterRequest(BaseModel):
    columns: List[str]
    algorithm: Literal["auto", "kmeans", "dbscan", "hierarchical"] = "auto"
    distance: Literal["auto", "euclidean", "manhattan", "cosine"] = "auto"
    num_clusters: Optional[int] = None
    standardize: bool = True

@app.post("/api/cluster_analysis")
async def cluster_analysis(req: ClusterRequest):
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    headers = stored_datasets["latest"]["headers"]
    data = stored_datasets["latest"]["data"]
    df = pd.DataFrame(data, columns=headers)

    if len(req.columns) < 2:
        raise HTTPException(status_code=400, detail="Je třeba vybrat alespoň dvě proměnné.")

    df_subset = df[req.columns].apply(pd.to_numeric, errors="coerce").dropna()
    if df_subset.empty:
        raise HTTPException(status_code=400, detail="Vybrané proměnné neobsahují žádná validní data.")

    if req.standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_subset)
    else:
        data_scaled = df_subset.values

    n_samples, n_features = data_scaled.shape
    used_algorithm = req.algorithm
    used_distance = req.distance if req.distance != "auto" else "euclidean"
    used_clusters = req.num_clusters
    reason = ""

    # Automatická detekce algoritmu
    if req.algorithm == "auto":
        if n_samples < 500:
            used_algorithm = "hierarchical"
            reason = "Dataset je malý → použita hierarchická shluková analýza."
        elif n_features > 20:
            used_algorithm = "kmeans"
            reason = "Dataset má mnoho proměnných → použita metoda KMeans s možností PCA."
        else:
            used_algorithm = "kmeans"
            reason = "Standardní podmínky → použita metoda KMeans."

    # Shlukování
    if used_algorithm == "kmeans":
        if req.num_clusters is None:
            sil_scores = []
            best_k = 2
            for k in range(2, min(10, n_samples)):
                model = KMeans(n_clusters=k, random_state=42)
                labels = model.fit_predict(data_scaled)
                score = silhouette_score(data_scaled, labels)
                sil_scores.append((k, score))
            best_k = max(sil_scores, key=lambda x: x[1])[0]
            used_clusters = best_k
            reason += f" Automaticky určeno {best_k} shluků pomocí silhouette skóre."
        model = KMeans(n_clusters=used_clusters, random_state=42)

    elif used_algorithm == "hierarchical":
        used_clusters = req.num_clusters or 3
        if used_distance != "euclidean":
            model = AgglomerativeClustering(n_clusters=used_clusters, affinity=used_distance, linkage="average")
            reason += f" Použita metrika '{used_distance}' s linkage='average'."
        else:
            model = AgglomerativeClustering(n_clusters=used_clusters, linkage="ward")
            reason += " Použita metrika 'euclidean' s linkage='ward'."

    elif used_algorithm == "dbscan":
        model = DBSCAN(metric=used_distance)
    else:
        raise HTTPException(status_code=400, detail="Neznámý algoritmus.")

    labels = model.fit_predict(data_scaled)

    silhouette = None
    if used_algorithm in ["kmeans", "hierarchical"] and len(set(labels)) > 1:
        silhouette = silhouette_score(data_scaled, labels)

    df_result = df_subset.copy()
    df_result["Cluster"] = labels

    summary_df = df_result.groupby("Cluster").agg(['mean', 'std', 'min', 'max']).round(3)
    summary_dict = summary_df.to_dict(orient="index")

    summary = {}
    for cluster, stats_per_cluster in summary_dict.items():
        flat_stats = {}
        for key, stat_value in stats_per_cluster.items():
            if isinstance(key, tuple) and len(key) == 2:
                col, stat = key
                flat_stats[f"{col} ({stat})"] = float(stat_value)
            else:
                flat_stats[str(key)] = float(stat_value)
        summary[int(cluster)] = flat_stats

    # PCA pro vizualizaci
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data_scaled)

    # 2D projekce
    pca_2d = {
        "x": pca_components[:, 0].tolist(),
        "y": pca_components[:, 1].tolist(),
        "labels": labels.tolist()
    }

    # Biplot - směry proměnných
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pca_vectors = []
    for name, vec in zip(req.columns, loadings):
        pca_vectors.append({
            "name": name,
            "x": float(vec[0]),
            "y": float(vec[1])
        })

    # Variance tabulka (volitelně pro další výstupy)
    pca_variance_table = []
    cumulative = 0
    for i, (eig, ratio) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_)):
        cumulative += ratio
        pca_variance_table.append({
            "Component": f"PC{i + 1}",
            "Eigenvalue": round(eig, 4),
            "% Variance": round(ratio * 100, 2),
            "Cumulative %": round(cumulative * 100, 2)
        })

    return {
        "method": used_algorithm,
        "reason": reason,
        "clusters": used_clusters,
        "distance_metric": used_distance,
        "silhouette_score": silhouette,
        "summary": summary,
        "labels": labels.tolist(),
        "columns": req.columns,
        "pca_2d": pca_2d,
        "pca_components": pca_vectors,
        "pca_variance_table": pca_variance_table
    }

class FactorAnalysisInterpretationRequest(BaseModel):
    analysis_type: str = "factor_analysis" # Keep consistent pattern
    columns_used: List[str]
    n_factors_extracted: int
    rotation_used: str
    standardized: bool
    # dropped_rows: int # Less critical for interpretation itself
    kmo_model: Optional[float] = None # Use Optional for safety
    bartlett_p_value: Optional[float] = None
    total_variance_explained_pct: Optional[float] = None

# --- Helper Functions ---
# ... (Keep safe_float function) ...

# --- API Endpoints ---
# ... (Keep ALL your existing endpoints like /api/classification_analysis, /api/factor_analysis, etc.) ...


# --- NEW Endpoint for AI Interpretation of Factor Analysis ---
@app.post("/api/interpret_factor_analysis")
async def interpret_factor_analysis(req: FactorAnalysisInterpretationRequest):
    logger.info(f"Received factor analysis interpretation request for columns: {req.columns_used}")

    # --- System Prompt for LLM ---
    system_prompt = (
        "Jsi AI asistent specializující se na analýzu dat. Uživatel provedl faktorovou analýzu (FA) a poskytne ti její klíčové výsledky.\n\n"
        "Tvým úkolem je interpretovat tyto výsledky v češtině, jednoduchým a srozumitelným jazykem pro někoho bez hlubokých znalostí statistiky.\n\n"
        "**Co je Faktorová Analýza (stručně):**\n"
        "FA se snaží najít skryté (latentní) 'faktory', které vysvětlují vzájemné korelace mezi původními pozorovanými proměnnými. Cílem je zjednodušit data a identifikovat základní struktury.\n\n"
        f"**Výsledky této analýzy:**\n"
        f"- Bylo extrahováno **{req.n_factors_extracted}** faktorů z proměnných: {', '.join(req.columns_used)}.\n"
        f"- Použitá metoda rotace byla: **{req.rotation_used}**. (Rotace pomáhá lépe interpretovat faktory; '{req.rotation_used}' je běžná volba).\n"
        f"- Data byla před analýzou {'standardizována (převedena na stejné měřítko)' if req.standardized else 'použita bez standardizace'}.\n\n"
        "**Hodnocení vhodnosti dat a modelu:**\n"
        f"- **KMO test (Kaiser-Meyer-Olkin):** {f'Hodnota {req.kmo_model:.3f}.' if req.kmo_model is not None else 'N/A.'} "
        f"{interpret_kmo(req.kmo_model) if req.kmo_model is not None else ''} (Hodnoty nad 0.6 jsou obecně považovány za přijatelné, vyšší jsou lepší. Ukazuje, zda data mají dostatek společné variance pro FA.)\n"
        f"- **Bartlettův test sférickosti:** {f'p-hodnota = {req.bartlett_p_value:.4g}.' if req.bartlett_p_value is not None else 'N/A.'} "
        f"{'Tento výsledek je statisticky významný (p < 0.05), což naznačuje, že mezi proměnnými existují korelace a data JSOU vhodná pro FA.' if req.bartlett_p_value is not None and req.bartlett_p_value < 0.05 else ('Tento výsledek NENÍ statisticky významný (p >= 0.05), což znamená, že data NEMUSÍ být vhodná pro FA (proměnné spolu dostatečně nekorelují).' if req.bartlett_p_value is not None else '')}\n"
        f"- **Celková vysvětlená variance:** {f'Nalezené faktory společně vysvětlují **{req.total_variance_explained_pct:.1f}%** celkové variability původních proměnných.' if req.total_variance_explained_pct is not None else 'Informace o celkové vysvětlené varianci není k dispozici.'} (Vyšší procento znamená, že faktory lépe zachycují informace z původních dat. Často se hledá hodnota alespoň 50-60 %.)\n\n"
        "**Další informace (z tabulek, které uživatel vidí):**\n"
        "- Tabulka **Faktorové zátěže (Loadings)** ukazuje, jak silně každá původní proměnná koreluje s každým extrahovaným faktorem. Vysoké zátěže (obvykle > 0.4 nebo 0.5) pomáhají pochopit, co daný faktor reprezentuje.\n"
        "- **Komunality** ukazují, jaký podíl variance *každé jednotlivé* proměnné je vysvětlen všemi nalezenými faktory dohromady.\n\n"
        "**Celkové shrnutí:**\n"
        "- Zhodnoť, zda se na základě KMO, Bartlettova testu a vysvětlené variance zdá analýza smysluplná a model užitečný.\n"
        f"- Zdůrazni, že pojmenování a interpretace významu jednotlivých {req.n_factors_extracted} faktorů vyžaduje podívat se na faktorové zátěže (které proměnné mají u daného faktoru vysokou hodnotu) a zapojit znalost oboru.\n\n"
        "Pravidla:\n"
        "- Odpovídej v češtině.\n"
        "- Buď jasný, stručný a srozumitelný pro laika.\n"
        "- Vysvětli význam klíčových čísel jednoduše.\n"
        "- Neuváděj vzorce ani kód.\n"
        "- Formátuj odpověď pro dobrou čitelnost (odstavce, tučné písmo)."
    )

    # --- Sestavení User Promptu ---
    # V tomto případě je většina informací už v system promptu,
    # user prompt může být jednoduchý nebo zopakovat klíčové metriky pro kontext.
    user_prompt_parts = [
        f"Provedl jsem faktorovou analýzu s následujícími výsledky:",
        f"- Počet extrahovaných faktorů: {req.n_factors_extracted}",
        f"- Použitá rotace: {req.rotation_used}",
        f"- KMO: {req.kmo_model:.3f}" if req.kmo_model is not None else "- KMO: N/A",
        f"- Bartlett p-value: {req.bartlett_p_value:.4g}" if req.bartlett_p_value is not None else "- Bartlett p-value: N/A",
        f"- Celková vysvětlená variance: {req.total_variance_explained_pct:.1f}%" if req.total_variance_explained_pct is not None else "- Celková vysvětlená variance: N/A",
        "\nProsím, interpretuj tyto výsledky."
    ]
    full_user_prompt = "\n".join(user_prompt_parts)

    # --- Volání LLM API (použijte vaši existující logiku/klíč) ---
    try:

        api_key = "OPENROUTER_API_KEY"
        if not api_key:
             raise HTTPException(status_code=500, detail="Chybí konfigurace API klíče pro LLM.")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", # Váš LLM endpoint
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free", # Váš model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                "max_tokens": 700 # Může být potřeba více pro FA
            },
            timeout=90 # Delší timeout pro komplexnější interpretaci
        )
        response.raise_for_status()

        llm_data = response.json()
        interpretation_text = llm_data.get("choices", [{}])[0].get("message", {}).get("content")

        if not interpretation_text:
             raise HTTPException(status_code=500, detail="AI nevrátila platnou interpretaci.")

        logger.info("AI interpretace pro FA úspěšně vygenerována.")
        return {"interpretation": interpretation_text}

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Chyba komunikace s LLM API pro FA interpretaci: {req_err}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Chyba při komunikaci s AI službou: {req_err}")
    except Exception as e:
        logger.error(f"Neočekávaná chyba při interpretaci FA: {e}", exc_info=True)
        # print(f"System prompt was: {system_prompt}") # Pro debugging
        # print(f"User prompt was: {full_user_prompt}") # Pro debugging
        raise HTTPException(status_code=500, detail=f"Interní chyba serveru při generování interpretace FA: {str(e)}")


# --- Helper function for KMO interpretation (add this somewhere accessible) ---
def interpret_kmo(kmo_value: Optional[float]) -> str:
    if kmo_value is None:
        return ""
    if kmo_value < 0.5: return "Tato hodnota je nepřijatelná."
    if kmo_value < 0.6: return "Tato hodnota je mizerná."
    if kmo_value < 0.7: return "Tato hodnota je slabá."
    if kmo_value < 0.8: return "Tato hodnota je střední."
    if kmo_value < 0.9: return "Tato hodnota je dobrá."
    return "Tato hodnota je vynikající."
class ClusterInterpretationRequest(BaseModel):
    analysis_type: str = "clustering"
    algorithm_used: str
    distance_metric: str
    number_of_clusters_found: Optional[int] = None # Může být null pro DBSCAN
    silhouette_score: Optional[float] = None # Může být null
    columns_used: List[str]
    # Můžeme přidat další info, např. zda byla použita standardizace

# --- Nový endpoint pro AI interpretaci shlukování ---
@app.post("/api/interpret_clustering")
async def interpret_clustering(req: ClusterInterpretationRequest):
    # --- System Prompt pro LLM ---
    system_prompt = (
        "Jsi AI asistent specializující se na analýzu dat. Uživatel provedl shlukovou analýzu a poskytne ti její klíčové výsledky.\n\n"
        "Tvým úkolem je interpretovat tyto výsledky v češtině, jednoduchým a srozumitelným jazykem.\n\n"
        f"Použitý algoritmus byl: **{req.algorithm_used}** s metrikou vzdálenosti '{req.distance_metric}'. Analýza byla provedena na proměnných: {', '.join(req.columns_used)}.\n\n"
        "**Interpretace výsledků:**\n"
        f"- **Počet nalezených shluků:** {'Bylo nalezeno ' + str(req.number_of_clusters_found) + ' shluků.' if req.number_of_clusters_found is not None else 'Algoritmus DBSCAN nalezl shluky automaticky (počet není pevně daný).'} Pokud DBSCAN našel body označené jako šum (-1), zmiň to - jsou to body, které nezapadají do žádného hustého shluku.\n"
        f"- **Silhouette Score:** {'Silhouette skóre bylo ' + f'{req.silhouette_score:.3f}' + '.' if req.silhouette_score is not None else 'Silhouette skóre nebylo pro tento algoritmus (DBSCAN) relevantní nebo nebylo možné spočítat (např. jen 1 shluk).'} "
        "Toto skóre měří, jak dobře jsou body odděleny mezi shluky a jak jsou si podobné body uvnitř jednoho shluku. Hodnoty blízko +1 znamenají dobře definované shluky. Hodnoty kolem 0 znamenají překrývající se shluky. Hodnoty blízko -1 znamenají, že body mohly být přiřazeny do špatných shluků.\n\n"
        "**Celkové zhodnocení:**\n"
        "- Na základě počtu shluků a Silhouette skóre (pokud je k dispozici) zhodnoť kvalitu shlukování. Např. 'Model našel X dobře oddělených shluků (vysoké Silhouette skóre).' nebo 'Nalezené shluky se zdají být překrývající (nízké/nulové Silhouette skóre).' nebo 'DBSCAN identifikoval několik hustých oblastí a možný šum.'\n"
        "- Zmínit, že vizualizace pomocí PCA (pokud byla zobrazena) pomáhá vidět strukturu shluků ve 2D, i když původní data měla více dimenzí.\n"
        "- Připomeň, že interpretace *významu* jednotlivých shluků (co charakterizuje shluk 1 vs. shluk 2) vyžaduje další analýzu průměrů/mediánů proměnných v jednotlivých shlucích (což uživatel vidí v souhrnné tabulce).\n\n"
        "Pravidla:\n"
        "- Odpovídej v češtině.\n"
        "- Buď srozumitelný.\n"
        "- Vysvětli význam Silhouette skóre jednoduše.\n"
        "- Neuváděj vzorce ani kód.\n"
        "- Formátuj odpověď pro dobrou čitelnost."
    )

    # --- Sestavení User Promptu ---
    user_prompt_parts = [
        f"Provedl jsem shlukovou analýzu ('{req.analysis_type}') algoritmem '{req.algorithm_used}' s metrikou '{req.distance_metric}'.",
        f"Analýza proběhla na sloupcích: {', '.join(req.columns_used)}.",
    ]
    if req.number_of_clusters_found is not None:
        user_prompt_parts.append(f"Bylo nalezeno {req.number_of_clusters_found} shluků.")
    else:
         user_prompt_parts.append("Počet shluků byl určen automaticky (DBSCAN).")

    if req.silhouette_score is not None:
         user_prompt_parts.append(f"Silhouette skóre: {req.silhouette_score:.4f}")
    else:
         user_prompt_parts.append("Silhouette skóre nebylo relevantní nebo spočítáno.")

    user_prompt_parts.append("\nProsím, interpretuj tyto výsledky.")
    full_user_prompt = "\n".join(user_prompt_parts)

    # --- Volání LLM API ---
    try:
        api_key ="OPENROUTER_API_KEY"
        if not api_key:
             raise HTTPException(status_code=500, detail="Chybí konfigurace API klíče pro LLM.")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                "max_tokens": 600
            },
            timeout=60
        )
        response.raise_for_status()

        llm_data = response.json()
        interpretation_text = llm_data.get("choices", [{}])[0].get("message", {}).get("content")

        if not interpretation_text:
             raise HTTPException(status_code=500, detail="AI nevrátila platnou interpretaci.")

        return {"interpretation": interpretation_text}

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Chyba komunikace s LLM API: {req_err}")
        raise HTTPException(status_code=503, detail=f"Chyba při komunikaci s AI službou: {req_err}")
    except Exception as e:
        logger.error(f"Neočekávaná chyba při interpretaci shlukování: {e}", exc_info=True)
        # print(f"User prompt was: {full_user_prompt}") # Pro debugging
        raise HTTPException(status_code=500, detail=f"Interní chyba serveru při generování interpretace: {str(e)}")


@app.post("/api/ai_suggest_analysis")
async def ai_suggest_analysis(req: PromptRequest):
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="No stored data available.")

    headers = stored_datasets["latest"]["headers"]
    columns_text = ", ".join(headers)
    full_prompt = (
        f"I have a dataset with the following columns: {columns_text}.\n\n"
        f"My question is: {req.prompt}"
    )

    system_prompt = (
        "You are a helpful assistant for data analysis. The user will describe what they want to find out from their dataset.\n\n"
        "Your task is to classify the request into one of the following types of analysis:\n\n"
        "1. Porovnání skupin\n"
        "2. Vztah mezi proměnnými\n"
        "3. Klasifikace\n"
        "4. Shluková analýza\n"
        "5. Faktorová analýza\n\n"
        "Rules:\n"
        "The response will be in Czech.\n"
        "- Respond with **only one** of the above categories.\n"
        "- Include a **short explanation** (1–3 sentences) why this analysis is appropriate.\n"
        "- If you are not sure what type of analysis to recommend, ask the user to clarify their problem.\n\n"
        "Forbidden:\n"
        "- Do not mention statistical tests.\n"
        "- Do not include code or formulas.\n"
        "- Do not guess multiple types — pick only one or ask for clarification."
    )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                "max_tokens": 300
            },
            timeout=60
        )

        return response.json()

    except Exception as e:
        return {"error": str(e)}

class CoefficientInterpretation(BaseModel):
    name: str
    value: Optional[float] = None # Posíláme jen hodnotu pro jednoduchost

class ResultsSummaryInterpretation(BaseModel):
    r2: Optional[float] = None
    r2_adjusted: Optional[float] = None
    rmse: Optional[float] = None
    accuracy: Optional[float] = None
    overfitting_detected: Optional[bool] = None

class InterpretationRequest(BaseModel):
    analysis_type: str = "regression"
    method: str
    dependent_variable: str
    independent_variables: List[str]
    results_summary: ResultsSummaryInterpretation
    coefficients: List[CoefficientInterpretation]
    intercept: Optional[float] = None

# --- Nový endpoint pro AI interpretaci ---
@app.post("/api/interpret_regression")
async def interpret_regression(req: InterpretationRequest):
    # --- System Prompt pro LLM ---
    system_prompt = (
        "Jsi AI asistent specializující se na analýzu dat. Uživatel provedl regresní analýzu a poskytne ti její výsledky.\n\n"
        "Tvým úkolem je interpretovat tyto výsledky v češtině, jednoduchým a srozumitelným jazykem pro někoho, kdo nemusí být expert na statistiku.\n\n"
        "Zaměř se na:\n"
        "1.  **Celkovou kvalitu modelu:** Vysvětli, co znamenají klíčové metriky (R-squared pro regresi, Accuracy pro klasifikaci) a jak dobře model vysvětluje data nebo predikuje výsledek.\n"
        "2.  **Vliv nezávislých proměnných:** Popiš, které proměnné se zdají být nejdůležitější (na základě koeficientů) a jaký mají vliv (pozitivní/negativní) na závislou proměnnou. Zmiň i intercept (průsečík), pokud má smysluplnou interpretaci v kontextu.\n"
        "3.  **Potenciální problémy:** Pokud byla detekována známka overfittingu, upozorni na to a stručně vysvětli, co to znamená.\n"
        "4.  **Použitou metodu:** Stručně zmiň použitou metodu a proč byla pravděpodobně vhodná (např. lineární regrese pro číselný výstup, logistická pro binární).\n\n"
        "Pravidla:\n"
        "- Odpovídej v češtině.\n"
        "- Buď stručný a věcný, ale srozumitelný.\n"
        "- Nepoužívej příliš technický žargon, pokud to není nutné.\n"
        "- Neuváděj vzorce ani kód.\n"
        "- Formátuj odpověď do odstavců pro lepší čitelnost.\n"
        "- Pokud některá metrika chybí (např. R-squared u klasifikace), nekomentuj její absenci, soustřeď se na dostupné informace.\n"
        "- Pokud jsou koeficienty pro multinomiální regresi, vysvětli, že reprezentují vliv na jednu z kategorií oproti referenční."
    )

    # --- Sestavení User Promptu z dat od frontendu ---
    user_prompt_parts = [
        f"Provedl jsem analýzu typu '{req.analysis_type}' s použitím metody '{req.method}'.",
        f"Závislá proměnná (Y): {req.dependent_variable}",
        f"Nezávislé proměnné (X): {', '.join(req.independent_variables)}",
        "\nKlíčové výsledky modelu:"
    ]
    if req.results_summary.r2 is not None:
        user_prompt_parts.append(f"- R-squared (R²): {req.results_summary.r2:.3f}")
    if req.results_summary.r2_adjusted is not None:
        user_prompt_parts.append(f"- Adjusted R²: {req.results_summary.r2_adjusted:.3f}")
    if req.results_summary.rmse is not None:
        user_prompt_parts.append(f"- RMSE (Root Mean Squared Error): {req.results_summary.rmse:.3f}")
    if req.results_summary.accuracy is not None:
        user_prompt_parts.append(f"- Přesnost (Accuracy): {req.results_summary.accuracy * 100:.1f}%")
    if req.results_summary.overfitting_detected is not None:
        user_prompt_parts.append(f"- Detekován možný overfitting: {'Ano' if req.results_summary.overfitting_detected else 'Ne'}")

    user_prompt_parts.append("\nOdhadnuté koeficienty:")
    if req.intercept is not None:
         user_prompt_parts.append(f"- Intercept (průsečík): {req.intercept:.3f}")
    for coef in req.coefficients:
         user_prompt_parts.append(f"- {coef.name}: {coef.value:.3f}" if coef.value is not None else f"- {coef.name}: N/A")

    # Přidání poznámky pro multinomiální regresi, pokud je to relevantní
    if req.method == "multinomial":
         user_prompt_parts.append("\n(Poznámka: Koeficienty u multinomiální regrese ukazují vliv proměnné na pravděpodobnost jedné z kategorií vůči referenční kategorii.)")

    user_prompt_parts.append("\nProsím, interpretuj tyto výsledky.")
    full_user_prompt = "\n".join(user_prompt_parts)

    # --- Volání LLM API (podobně jako v ai_suggest_analysis) ---
    try:
        # POZOR: Zabezpečte svůj API klíč! Neukládejte ho přímo v kódu.
        # Použijte např. environment variables.
        api_key ="OPENROUTER_API_KEY"
        if not api_key:
             raise HTTPException(status_code=500, detail="Chybí konfigurace API klíče pro LLM.")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free", # Nebo jiný model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                "max_tokens": 500 # Můžete upravit podle potřeby
            },
            timeout=60
        )
        response.raise_for_status() # Vyvolá chybu pro status kódy 4xx/5xx

        llm_data = response.json()

        # Extrakce textu odpovědi
        interpretation_text = llm_data.get("choices", [{}])[0].get("message", {}).get("content")

        if not interpretation_text:
             raise HTTPException(status_code=500, detail="AI nevrátila platnou interpretaci.")

        # Vracíme výsledek ve formátu očekávaném frontendem
        return {"interpretation": interpretation_text}

    except requests.exceptions.RequestException as req_err:
        print(f"Chyba komunikace s LLM API: {req_err}")
        raise HTTPException(status_code=503, detail=f"Chyba při komunikaci s AI službou: {req_err}")
    except Exception as e:
        print(f"Neočekávaná chyba při interpretaci: {e}")
        # Můžete logovat 'full_user_prompt' pro debugging
        raise HTTPException(status_code=500, detail=f"Interní chyba serveru při generování interpretace: {str(e)}")



class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/ai_suggest_relationship")
async def ai_suggest_relationship(req: PromptRequest):
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="No stored data available.")

    headers = stored_datasets["latest"]["headers"]
    columns_text = ", ".join(headers)
    full_prompt = (
        f"Mám dataset s následujícími sloupci: {columns_text}.\n\n"
        f"Můj dotaz je: {req.prompt}"
    )

    system_prompt = (
        "Jsi nápomocný asistent pro analýzu vztahů mezi proměnnými v datové sadě.\n"
        "Uživatel popíše, co chce z dat zjistit.\n\n"
        "Tvoje úloha je doporučit jeden z následujících typů analýzy:\n\n"
        "1. Korelace\n"
        "2. Regrese\n"
        "3. Test závislosti\n\n"
        "Pravidla:\n"
        "- Odpověz pouze jedním z těchto názvů.\n"
        "- Přidej krátké vysvětlení (1–3 věty), proč je daný typ vhodný.\n"
        "- Pokud není jasné, co uživatel chce, požádej o upřesnění.\n"
        "- Odpověď piš v češtině.\n\n"
        "Zakázáno:\n"
        "- Nezmiňuj konkrétní statistické testy.\n"
        "- Nepiš kód nebo rovnice.\n"
        "- Nehádej více možností – vyber pouze jednu nebo požádej o upřesnění."
    )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                "max_tokens": 300
            },
            timeout=60
        )

        return response.json()

    except Exception as e:
        return {"error": str(e)}



class TransformRequest(BaseModel):
    column: str
    method: str  # 'log', 'sqrt', 'boxcox'

@app.post("/api/transform_column")
async def transform_column(req: TransformRequest):
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    headers = stored_datasets["latest"]["headers"]
    data = stored_datasets["latest"]["data"]
    df = pd.DataFrame(data, columns=headers)

    if req.column not in df.columns:
        raise HTTPException(status_code=404, detail="Sloupec nebyl nalezen.")

    try:
        # převedeme na čísla
        series = pd.to_numeric(df[req.column], errors="coerce")

        if req.method == "log":
            if (series <= 0).any():
                raise ValueError("Logaritmická transformace vyžaduje pouze kladné hodnoty.")
            transformed = np.log(series)
        elif req.method == "sqrt":
            if (series < 0).any():
                raise ValueError("Odmocnina není definovaná pro záporné hodnoty.")
            transformed = np.sqrt(series)
        elif req.method == "boxcox":
            # odstraníme nuly a zápory
            if (series <= 0).any():
                raise ValueError("Box-Cox transformace vyžaduje pouze kladné hodnoty.")
            # odstranit NaN pro boxcox
            non_na = series.dropna()
            transformed_non_na, _ = boxcox(non_na)
            # znovu vložíme NaN zpět na původní místa
            transformed = pd.Series(data=np.nan, index=series.index)
            transformed[non_na.index] = transformed_non_na
        else:
            raise ValueError("Neznámá transformační metoda.")

        # nahradíme původní sloupec transformovanými daty
        df[req.column] = transformed

        # přepíšeme zpět do uložených dat
        stored_datasets["latest"]["headers"] = list(df.columns)
        stored_datasets["latest"]["data"] = df.values.tolist()

        return {"message": f"Transformace sloupce '{req.column}' metodou '{req.method}' proběhla úspěšně."}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Chyba při transformaci dat.")


@app.get("/api/check_normality")
async def check_normality(preferred_test: Optional[str] = Query(None)):
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    headers = stored_datasets["latest"]["headers"]
    data = stored_datasets["latest"]["data"]
    df = pd.DataFrame(data, columns=headers)

    results = []

    for col in df.columns:
        try:
            original_series = pd.to_numeric(df[col], errors="coerce")
            series = original_series.dropna()

            if len(series) < 3:
                continue  # Nedostatek dat

            total_count = len(original_series)
            missing_count = original_series.isna().sum()
            has_missing = missing_count > 0

            # Detekce extrémních hodnot
            z_scores = zscore(series)
            outlier_ratio = np.mean(np.abs(z_scores) > 3)
            has_many_outliers = outlier_ratio > 0.05

            test_used = ""
            p = None
            reason = ""

            if preferred_test == "shapiro" or (preferred_test is None and len(series) < 50):
                stat, p = shapiro(series)
                test_used = "Shapiro-Wilk"
                reason = "Test vybrán, protože počet hodnot je menší než 50."
            elif preferred_test == "ks" or preferred_test is None:
                standardized = (series - series.mean()) / series.std()
                stat, p = kstest(standardized, "norm")
                test_used = "Kolmogorov–Smirnov"
                reason = "Test vybrán, protože počet hodnot je 50 nebo více."

            note_parts = [reason]
            if has_many_outliers:
                note_parts.append("Pozor: hodně outlierů, může ovlivnit výsledek.")
            if has_missing:
                note_parts.append(f"Sloupec obsahuje {missing_count} chybějících hodnot z {total_count}.")

            results.append({
                "column": col,
                "test": test_used,
                "pValue": float(f"{p:.6g}"),
                "isNormal": bool(p > 0.05),
                "warning": " ".join(note_parts) if note_parts else "-",
                "hasMissing": bool(has_missing)  # <- Tady je oprava
            })

        except Exception as e:
            print(f"Chyba ve sloupci {col}: {e}")
            continue
    stored_datasets["latest"]["normality"] = {
        result["column"]: result["isNormal"] for result in results
    }

    return {"results": results}

@app.get("/api/get_column_types")
async def get_column_types():
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    analyze_result = stored_datasets["latest"].get("column_types")
    if not analyze_result:
        raise HTTPException(status_code=400, detail="Data nebyla ještě analyzována.")

    response = []
    for col, info in analyze_result.items():
        response.append({
            "name": col,
            "type": info["type"]
        })

    return response

class GroupComparisonRequest(BaseModel):
    numerical: List[str]
    categorical: List[str]
    paired: bool


@app.post("/api/group_comparison")
async def group_comparison(req: GroupComparisonRequest):
    numerical = req.numerical
    categorical = req.categorical
    paired = req.paired

    if "latest" not in stored_datasets:
        raise HTTPException(status_code=404, detail="Data nejsou nahraná.") # 404 pro nenalezeno

    # Zkontroluj, jestli už máme výpočet normality. Pokud ne, zavolej ho.
    if "normality" not in stored_datasets["latest"]:
        print("Informace o normalitě nenalezena, počítám...")
        try:
            # Spustíme funkci pro výpočet normality
            normality_result_full = await check_normality() # Získání kompletní odpovědi
            # Uložíme výsledek (slovník {sloupec: bool}) do cache
            stored_datasets["latest"]["normality"] = {
                r["column"]: r["isNormal"] for r in normality_result_full.get("results", [])
            }
            print("Informace o normalitě vypočítána a uložena.")
        except Exception as e:
             print(f"Chyba při výpočtu normality: {e}")
             # Pokud výpočet selže, pokračujeme s prázdným dict, což povede k neparametrickým testům
             stored_datasets["latest"]["normality"] = {}


    if not numerical or not categorical:
        raise HTTPException(status_code=400, detail="Musíte vybrat numerické i kategoriální proměnné.")

    df = pd.DataFrame(stored_datasets["latest"]["data"], columns=stored_datasets["latest"]["headers"])
    # Získání normality info bezpečně, s default {} pokud neexistuje
    normality_info = stored_datasets["latest"].get("normality", {})

    results_by_group = []

    # --- Logika pro párový test ---
    if paired:
        if len(numerical) != 2 or len(categorical) != 1:
            raise HTTPException(status_code=400, detail="Párová analýza vyžaduje výběr přesně 2 číselných a 1 kategoriální proměnné.")

        col1, col2 = numerical
        group_col = categorical[0] # Předpokládáme, že pro párový test má smysl jen 1 skupina pro definici párů (ID subjektu apod.) - zvážit, zda je to váš případ užití

        # Pokus o konverzi sloupců na numerické, chyby budou NaN
        try:
            df[col1] = pd.to_numeric(df[col1], errors='coerce')
            df[col2] = pd.to_numeric(df[col2], errors='coerce')
        except KeyError as e:
             raise HTTPException(status_code=404, detail=f"Sloupec '{e}' nebyl nalezen v datech.")
        except Exception as e:
             # Obecná chyba při konverzi
             raise HTTPException(status_code=400, detail=f"Chyba při konverzi číselných sloupců: {e}")

        group_results = []

        # Iterujeme přes skupiny definované kategoriální proměnnou
        # POZNÁMKA: U párového testu se typicky testuje rozdíl napříč podmínkami *pro stejné subjekty*.
        # Group_col by zde měl identifikovat spíše subjekty nebo páry, pokud testujete efekt *uvnitř* subjektů.
        # Pokud group_col definuje nezávislé skupiny a vy chcete párový test *v rámci každé skupiny*, logika je zde správná.
        # Zvažte, zda váš scénář odpovídá tomuto předpokladu.
        if group_col not in df.columns:
             raise HTTPException(status_code=404, detail=f"Kategoriální sloupec '{group_col}' nebyl nalezen v datech.")

        for group_value, group_df in df.groupby(group_col):
            # Vytvoření párů a odstranění řádků s jakýmkoli NA v páru
            paired_data = group_df[[col1, col2]].dropna()

            if len(paired_data) < 3: # Potřebujeme alespoň 3 páry pro smysluplný test
                print(f"Přeskakuji skupinu '{group_value}' pro párový test - méně než 3 platné páry ({len(paired_data)}).")
                continue

            x_vals = paired_data[col1]
            y_vals = paired_data[col2]

            # ***** OPRAVA ZDE: Testujeme normalitu ROZDÍLŮ *****
            differences = x_vals - y_vals
            is_diff_normal = False # Výchozí stav: nenormální
            note = f"Párový test, skupina: {group_value}"
            test = "N/A"
            p = float('nan')
            stat = float('nan')

            try:
                # Potřebujeme alespoň 3 rozdíly pro Shapiro test
                if len(differences) >= 3:
                    # Ošetření případu, kdy jsou všechny rozdíly stejné (Shapiro selže)
                    if len(differences.unique()) == 1:
                        is_diff_normal = False # Nelze testovat normalitu, předpokládáme neparametrický
                        note += ", všechny rozdíly stejné"
                    else:
                        stat_norm, p_norm = shapiro(differences)
                        is_diff_normal = p_norm > 0.05
                        print(f"  Normality test rozdílů (skupina {group_value}): p = {p_norm:.4f}, isNormal = {is_diff_normal}")
                else:
                    # Pokud máme méně než 3 rozdíly (nemělo by nastat kvůli kontrole výše, ale pro jistotu)
                     is_diff_normal = False
                     note += ", málo dat pro test normality rozdílů"


                # Výběr testu na základě normality rozdílů
                if is_diff_normal:
                    # Rozdíly jsou normálně rozložené -> Párový t-test
                    stat, p = ttest_rel(x_vals, y_vals)
                    test = "Párový t-test"
                    note += ", normální rozložení rozdílů"
                else:
                    # Rozdíly nejsou normálně rozložené (nebo test selhal) -> Wilcoxonův test
                    # Wilcoxon vyžaduje N > ~5-10 pro spolehlivé p-hodnoty, pro malé N může dát varování
                    if len(differences) > 0: # Zajistí, že nevoláme s prázdnými daty
                         # Ošetření pro případ, kdy jsou všechny rozdíly nulové (Wilcoxon může selhat)
                         if (differences == 0).all():
                             test = "Wilcoxonův test"
                             stat, p = float('nan'), 1.0 # Není žádný rozdíl
                             note += ", všechny rozdíly nulové"
                         else:
                             # Použití korekce a metody pro zpracování nul a vazeb
                             stat, p = wilcoxon(x_vals, y_vals, zero_method='zsplit', correction=True, mode='approx')
                             test = "Wilcoxonův test"
                             note += ", nenormální rozložení rozdílů (nebo nelze testovat)"
                    else:
                         # Nemělo by nastat, ale pro jistotu
                         test = "Wilcoxonův test (neběžel)"
                         note += ", žádná data pro Wilcoxon test"

            except ValueError as ve:
                 print(f"Chyba při statistickém testu pro skupinu {group_value}: {ve}")
                 test = f"Chyba testu ({'t-test' if is_diff_normal else 'Wilcoxon'})"
                 note += f", chyba: {ve}"
                 stat, p = float('nan'), float('nan')
            except Exception as e:
                 print(f"Neočekávaná chyba při statistickém testu pro skupinu {group_value}: {e}")
                 test = "Neočekávaná chyba testu"
                 note += f", chyba: {e}"
                 stat, p = float('nan'), float('nan')


            group_results.append({
                "numericColumn": f"{col1} vs {col2}", # Jasnější označení
                "test": test,
                "statistic": float(f"{stat:.6g}") if not np.isnan(stat) else None, # Přidání statistiky
                "pValue": float(f"{p:.6g}") if not np.isnan(p) else None, # Zajistí None místo NaN pro JSON
                "isSignificant": bool(not np.isnan(p) and p < 0.05),
                "note": note
            })

        # Struktura výsledků pro párový test (agregováno přes skupiny)
        results_by_group.append({
            "groupColumn": group_col, # Název sloupce, který definoval "skupiny" pro test
            "type": "paired",
            "results": group_results
        })

    # --- Logika pro nezávislé skupiny (zůstává stejná) ---
    else:
        for group_col in categorical:
            if group_col not in df.columns:
                print(f"Varování: Kategoriální sloupec '{group_col}' nebyl nalezen, přeskakuji.")
                continue

            group_results = []

            for num_col in numerical:
                if num_col not in df.columns:
                     print(f"Varování: Numerický sloupec '{num_col}' nebyl nalezen, přeskakuji pro skupinu '{group_col}'.")
                     continue

                test = "N/A"
                p = float('nan')
                stat = float('nan')
                reason = ""
                is_significant = False

                try:
                    # Pokus o konverzi numerického sloupce ZDE, abychom nezměnili původní df pro další iterace
                    numeric_data_series = pd.to_numeric(df[num_col], errors='coerce')
                    # Vytvoření subsetu s aktuální kategoriální a numerickou proměnnou
                    subset = pd.concat([df[group_col], numeric_data_series], axis=1).dropna()

                    if subset.empty or subset[group_col].nunique() < 2:
                        print(f"Přeskakuji {num_col} vs {group_col} - nedostatek dat nebo méně než 2 skupiny po odstranění NA.")
                        continue

                    unique_groups = subset[group_col].unique()
                    # Vytvoření seznamu Series pro každou skupinu
                    values = [subset[subset[group_col] == g][num_col] for g in unique_groups]

                    # Získání informace o normalitě z cache (s default False)
                    is_col_normal = normality_info.get(num_col, False)
                    equal_var = True # Výchozí předpoklad

                    if len(unique_groups) == 2:
                        # Levene test pro 2 skupiny - jen pokud máme dostatek dat v obou skupinách
                        if all(len(v) >= 3 for v in values): # Levene potřebuje alespoň pár bodů
                           try:
                               stat_levene, p_levene = levene(*values)
                               equal_var = p_levene > 0.05
                               reason += f"Levene p={p_levene:.3f}. "
                           except ValueError as ve:
                               print(f"Levene test selhal pro {num_col} vs {group_col}: {ve}. Předpokládám nerovnost rozptylů.")
                               equal_var = False # Pokud test selže, bezpečnější předpokládat nerovnost
                               reason += "Levene test selhal. "
                        else:
                           equal_var = False # Málo dat pro Levene, předpokládáme nerovnost
                           reason += "Málo dat pro Levene test. "


                        # Výběr testu pro 2 skupiny
                        if is_col_normal and equal_var:
                            stat, p = ttest_ind(*values, equal_var=True)
                            test = "t-test (nezávislý)"
                            reason += "2 skupiny, normální data*, homogenní rozptyly"
                        elif is_col_normal: # equal_var je False
                            stat, p = ttest_ind(*values, equal_var=False)
                            test = "Welchův t-test"
                            reason += "2 skupiny, normální data*, rozdílné rozptyly"
                        else: # Nenormální data
                            stat, p = mannwhitneyu(*values, alternative='two-sided') # Explicitně two-sided
                            test = "Mann–Whitney U"
                            reason += "2 skupiny, nenormální data*"
                    elif len(unique_groups) > 2:
                        # Levene test pro >2 skupiny
                        if all(len(v) >= 3 for v in values):
                            try:
                                stat_levene, p_levene = levene(*values)
                                equal_var = p_levene > 0.05
                                reason += f"Levene p={p_levene:.3f}. "
                            except ValueError as ve:
                                print(f"Levene test selhal pro {num_col} vs {group_col}: {ve}. Předpokládám nerovnost rozptylů.")
                                equal_var = False
                                reason += "Levene test selhal. "
                        else:
                            equal_var = False
                            reason += "Málo dat pro Levene test. "

                        # Výběr testu pro >2 skupiny
                        if is_col_normal and equal_var:
                            stat, p = f_oneway(*values)
                            test = "ANOVA"
                            reason += "více skupin, normální data*, homogenní rozptyly"
                        else: # Nenormální data NEBO rozdílné rozptyly
                            stat, p = kruskal(*values)
                            test = "Kruskal–Wallis"
                            reason += "více skupin, nenormální data* nebo rozdílné rozptyly"
                    else: # Méně než 2 skupiny (nemělo by nastat kvůli kontrole výše)
                        continue

                    is_significant = bool(not np.isnan(p) and p < 0.05)

                except ValueError as ve:
                     print(f"Chyba při statistickém testu pro {num_col} vs {group_col}: {ve}")
                     test = f"Chyba testu"
                     reason += f", chyba: {ve}"
                     stat, p, is_significant = float('nan'), float('nan'), False
                except Exception as e:
                     print(f"Neočekávaná chyba při statistickém testu pro {num_col} vs {group_col}: {e}")
                     test = "Neočekávaná chyba testu"
                     reason += f", chyba: {e}"
                     stat, p, is_significant = float('nan'), float('nan'), False


                group_results.append({
                    "numericColumn": num_col,
                    "test": test,
                    "statistic": float(f"{stat:.6g}") if not np.isnan(stat) else None,
                    "pValue": float(f"{p:.6g}") if not np.isnan(p) else None,
                    "isSignificant": is_significant,
                    "note": reason.strip() + " (*dle předpočítané normality)" # Dodatek k původu normality
                })

            # Přidání výsledků pro aktuální kategoriální proměnnou
            if group_results: # Přidáme jen pokud máme nějaké výsledky
                results_by_group.append({
                    "groupColumn": group_col,
                    "type": "independent",
                    "results": group_results
                })

    if not results_by_group:
         # Pokud se neprovedl žádný test (např. kvůli nedostatku dat)
         raise HTTPException(status_code=400, detail="Nepodařilo se provést žádné porovnání skupin. Zkontrolujte data a výběr proměnných.")


    return {"results": results_by_group}

@app.post("/api/update_column_type")
async def update_column_type(req: Request):
    body = await req.json()
    column = body.get("column")
    new_type = body.get("newType")

    if not column or new_type not in ["Kategorie", "Číselný"]:
        raise HTTPException(status_code=400, detail="Neplatný vstup")

    if "latest" not in stored_datasets or "column_types" not in stored_datasets["latest"]:
        raise HTTPException(status_code=400, detail="Data nejsou připravena")

    if column not in stored_datasets["latest"]["column_types"]:
        raise HTTPException(status_code=404, detail="Sloupec nenalezen")

    stored_datasets["latest"]["column_types"][column]["type"] = new_type
    return {"status": "updated"}

@app.get("/api/get_outliers")
async def get_outliers():
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    headers = stored_datasets["latest"]["headers"]
    data = stored_datasets["latest"]["data"]

    df = pd.DataFrame(data, columns=headers)
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    outliers_summary = []

    for col in df_numeric.columns:
        series = df_numeric[col].dropna()
        if series.empty:
            continue

        mean = series.mean()
        std = series.std()
        if std == 0:
            continue

        z_scores = (series - mean) / std
        outliers = series[abs(z_scores) > 3]

        outliers_summary.append({
            "column": col,
            "count": int(len(outliers)),
            "percent": float(len(outliers)) / len(series) * 100,
            "mean": float(mean),
            "std": float(std),
            "values": series.tolist(),
            "outliers": outliers.tolist()
        })

    return outliers_summary

# @app.post("/api/analyze_with_llm")
# async def analyze_with_llm(request: DataRequest):
#     df = pd.DataFrame(request.data, columns=request.headers)
#
#     # 🚀 Odešleme celý dataset (ale max. 500 řádků, pokud je velký)
#     if len(df) > 500:
#         df = df.sample(500, random_state=42)
#
#     # ✅ Převod na CSV formát (lepší pro LLM než JSON)
#     csv_data = df.to_csv(index=False)
#
#     def stream_llm_response():
#         try:
#             print("🟡 Odesílám dotaz na LLM...")
#
#             response = requests.post(
#                 "http://127.0.0.1:1234/v1/chat/completions",
#                 json={
#                     #"model": "deepseek-r1-distill-llama-8b",
#                     "model": "hermes-3-llama-3.1-8b",
#                     "messages": [{"role": "user", "content": f"{csv_data}"}],
#                     "max_tokens": 200,
#                     "stream": True
#                 },
#                 timeout=60,
#                 stream=True
#             )
#
#             print("🟢 LLM odpověď začíná streamovat...")
#             full_response = ""
#             last_char = ""
#
#             for line in response.iter_lines():
#                 if line:
#                     decoded_line = line.decode("utf-8").strip()
#                     print(f"🔹 Přijatý řádek: {decoded_line}")
#
#                     if decoded_line.startswith("data:"):
#                         decoded_line = decoded_line.replace("data: ", "")
#
#                     try:
#                         json_data = json.loads(decoded_line)
#
#                         if "choices" in json_data and json_data["choices"]:
#                             content_chunk = json_data["choices"][0]["delta"].get("content", "")
#
#                             if content_chunk:
#                                 # ✅ Pokud poslední znak není mezera, ale přichází další text, přidáme mezeru
#                                 if last_char not in ["", " ", "\n"] and content_chunk[0] not in [".", ",", "!", "?",
#                                                                                                  ";", ":"]:
#                                     content_chunk = " " + content_chunk
#
#                                 full_response += content_chunk
#                                 last_char = content_chunk[-1] if content_chunk else last_char
#
#                                 print(f"📝 Obsah: {content_chunk}")
#                                 yield content_chunk
#
#                     except (KeyError, json.JSONDecodeError):
#                         print("⚠️ Chyba při parsování JSON, pokračuji...")
#                         continue
#
#             print("✅ Streamování dokončeno. Celá odpověď:")
#             print(full_response)
#
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Chyba při komunikaci s LLM: {str(e)}")
#             yield f"❌ Chyba při komunikaci s LLM: {str(e)}\n"
#
#     return StreamingResponse(stream_llm_response(), media_type="text/event-stream")
#
@app.post("/api/analyze_with_llm")
async def analyze_with_llm(request: DataRequest):
    df = pd.DataFrame(request.data, columns=request.headers)
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    csv_data = df.to_csv(index=False)
    def stream_llm_response():
        try:
            print("Odesílám dotaz na LLM...")
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek/deepseek-chat-v3-0324:free",
                             "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Classify each column of the input table as either numerical or categorical.\n\n"
                                "Respond with a single line of letters:\n\n"
                                "Use \"c\" for numerical columns.\n\n"
                                "Use \"k\" for categorical columns.\n\n"
                                "Separate each letter with a semicolon (;).\n\n"
                                "There HAS TO BE a semicolon after each letter\n\n"
                                "The total number of letters must match the number of columns in the dataset.\n"
                                "Do not include column names, explanations, or any other text.\n\n"
                                "Forbidden: column names, descriptions, reasoning\n"
                                "Allowed: c;k;"
                            )
                        },
                        {
                            "role": "user",
                            "content": csv_data
                        }
                    ],
                    "max_tokens": 200,
                    "stream": True
                },
                timeout=60,
                stream=True
            )

            print("🟢 LLM odpověď začíná streamovat...")
            full_response = ""
            last_char = ""

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8").strip()
                    print(f"🔹 Přijatý řádek: {decoded_line}")

                    if decoded_line.startswith("data: "):
                        decoded_line = decoded_line.replace("data: ", "")

                    try:
                        json_data = json.loads(decoded_line)

                        if "choices" in json_data and json_data["choices"]:
                            content_chunk = json_data["choices"][0]["delta"].get("content", "")

                            if content_chunk:
                                if last_char not in ["", " ", "\n"] and content_chunk[0] not in [".", ",", "!", "?", ";", ":"]:
                                    content_chunk = " " + content_chunk

                                full_response += content_chunk
                                last_char = content_chunk[-1] if content_chunk else last_char

                                print(f"📝 Obsah: {content_chunk}")
                                yield content_chunk

                    except (KeyError, json.JSONDecodeError):
                        print("⚠️ Chyba při parsování JSON, pokračuji...")
                        continue

            print("✅ Streamování dokončeno. Celá odpověď:")
            print(full_response)

        except requests.exceptions.RequestException as e:
            print(f"❌ Chyba při komunikaci s LLM: {str(e)}")
            yield f"❌ Chyba při komunikaci s LLM: {str(e)}\n"

    return StreamingResponse(stream_llm_response(), media_type="text/event-stream")

class UpdateTypeRequest(BaseModel):
    column: str
    newType: str # "Kat
    

@app.post("/api/validate_and_update_column_type") # Nový název endpointu je lepší
async def validate_and_update_column_type(req: UpdateTypeRequest):
    column = req.column
    new_type = req.newType
    logger.info(f"Received request to validate and update type for column '{column}' to '{new_type}'") # Log

    # --- Základní validace vstupu ---
    if new_type not in ["Kategorie", "Číselný"]:
        logger.error(f"Invalid newType received: {new_type}")
        raise HTTPException(status_code=400, detail="Neplatný cílový typ sloupce.")

    if "latest" not in stored_datasets:
        logger.error("Attempted to update type but no data is stored.")
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    headers = stored_datasets["latest"].get("headers")
    data = stored_datasets["latest"].get("data")
    column_types_info = stored_datasets["latest"].get("column_types", {})

    if not headers or data is None: # Zkontrolujte i data
        logger.error("Stored data is incomplete (missing headers or data).")
        raise HTTPException(status_code=400, detail="Uložená data jsou nekompletní.")

    if column not in headers:
        logger.error(f"Column '{column}' not found in headers.")
        raise HTTPException(status_code=404, detail=f"Sloupec '{column}' nebyl v datech nalezen.")

    if column not in column_types_info:
         # Pokud typy ještě nebyly analyzovány, nemůžeme měnit
         logger.error(f"Column types not analyzed yet, cannot update type for '{column}'.")
         raise HTTPException(status_code=400, detail="Typy sloupců ještě nebyly analyzovány.")


    # --- Validace konverze dat (pokud se mění na Číselný) ---
    if new_type == "Číselný":
        try:
            # Získání indexu sloupce
            col_index = headers.index(column)
            # Extrakce hodnot sloupce (efektivnější než tvořit celý DataFrame)
            column_values = [row[col_index] for row in data if row is not None and len(row) > col_index]

            # Vytvoření Pandas Series pro snadnější validaci
            series = pd.Series(column_values)
            # Nahrazení prázdných stringů a None konzistentně NaN
            series = series.replace('', np.nan).astype(object) #astype(object) for mixed types initially

            # Pokus o konverzi na číslo, chyby budou NaN
            numeric_series = pd.to_numeric(series, errors='coerce')

            # Zjištění, zda PŮVODNÍ ne-NaN hodnoty selhaly při konverzi
            original_non_na_mask = series.notna()
            conversion_failed_mask = numeric_series.isna() & original_non_na_mask

            if conversion_failed_mask.any():
                # Najdi pár příkladů hodnot, které selhaly
                failed_examples = series[conversion_failed_mask].unique()
                examples_str = ", ".join(map(str, failed_examples[:5])) # Ukaž max 5 příkladů
                error_msg = f"Sloupec '{column}' nelze převést na číselný typ, protože obsahuje nečíselné hodnoty (např.: {examples_str})."
                logger.warning(f"Validation failed for column '{column}' to 'Číselný': {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg) # Vrátíme chybu 400

            logger.info(f"Validation successful for column '{column}' to 'Číselný'.")

        except HTTPException as http_exc:
             raise http_exc # Propagujeme HTTP chybu dál
        except Exception as e:
            logger.error(f"Unexpected error during validation for column '{column}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Nastala chyba při validaci dat sloupce '{column}': {str(e)}")

    # --- Aktualizace typu v `stored_datasets`, pokud validace prošla (nebo se měnilo na Kategorie) ---
    stored_datasets["latest"]["column_types"][column]["type"] = new_type
    logger.info(f"Successfully updated type for column '{column}' to '{new_type}'.")

    # Vrátíme úspěšnou odpověď
    return {"status": "updated", "column": column, "newType": new_type}
@app.post("/api/handle_outliers")
async def handle_outliers(request: Request):
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=404, detail="No data uploaded.")

    body = await request.json()
    column = body.get("column")
    method = body.get("method")
    custom_value = body.get("custom_value")

    headers = stored_datasets["latest"]["headers"]
    raw_data = stored_datasets["latest"]["data"]
    df = pd.DataFrame(raw_data, columns=headers)

    # Pomocná funkce pro zpracování jednoho sloupce
    def handle_column(col):
        values = pd.to_numeric(df[col], errors='coerce')
        mean = values.mean()
        std = values.std()
        z_scores = (values - mean) / std
        outlier_mask = z_scores.abs() > 2

        if method == "remove":
            return ~outlier_mask  # Vrací masku pro ponechané řádky

        elif method == "replace_mean":
            df.loc[outlier_mask, col] = mean
        elif method == "replace_median":
            df.loc[outlier_mask, col] = values.median()
        elif method == "replace_custom":
            try:
                replacement = float(custom_value)
                df.loc[outlier_mask, col] = replacement
            except:
                raise HTTPException(status_code=400, detail="Invalid custom value")
        elif method == "clip":
            lower_limit = mean - 2 * std
            upper_limit = mean + 2 * std
            df.loc[values < lower_limit, col] = lower_limit
            df.loc[values > upper_limit, col] = upper_limit

        return None

    if column == "ALL":
        if method == "remove":
            # vytvoř kombinovanou masku pro všechny sloupce
            keep_mask = pd.Series([True] * len(df))
            for col in df.columns:
                col_mask = handle_column(col)
                if col_mask is not None:
                    keep_mask &= col_mask
            df = df[keep_mask]
        else:
            for col in df.columns:
                handle_column(col)
    else:
        if column not in df.columns:
            raise HTTPException(status_code=400, detail="Column not found.")
        if method == "remove":
            keep_mask = handle_column(column)
            if keep_mask is not None:
                df = df[keep_mask]
        else:
            handle_column(column)

    stored_datasets["latest"] = {
        "headers": df.columns.tolist(),
        "data": df.astype(str).where(pd.notna(df), "").values.tolist()
    }

    return {"status": "outliers handled"}

class FactorAnalysisRequest(BaseModel):
    columns: List[str] = Field(..., min_length=3)
    n_factors: Optional[int] = Field(None, gt=0)
    rotation: Optional[str] = "varimax"
    standardize: bool = True

class DataAdequacy(BaseModel):
    kmo_model: Optional[float] = None
    bartlett_chi_square: Optional[float] = None
    bartlett_p_value: Optional[float] = None

class FactorVarianceItem(BaseModel):
    factor: str
    ssl: float
    variance_pct: float
    cumulative_variance_pct: float

class FactorAnalysisResult(BaseModel):
    columns_used: List[str]
    n_factors_requested: Optional[int]
    n_factors_extracted: int
    eigenvalue_criterion_used: bool
    eigenvalues: Optional[List[Optional[float]]] = None # Allow None within the list too
    rotation_used: str
    standardized: bool
    dropped_rows: int
    data_adequacy: DataAdequacy
    factor_loadings: Dict[str, Dict[str, Optional[float]]] # Allow None for loadings
    factor_variance: List[FactorVarianceItem]
    total_variance_explained_pct: Optional[float] = None
    communalities: Dict[str, Optional[float]] # Allow None for communalities

# --- Helper Function for NaN/Inf (Add this if you don't have a similar one) ---
def safe_float(value: Any) -> Optional[float]:
    """Converts value to float, returning None if NaN, Inf, or conversion fails."""
    try:
        # Check if value is already None or an empty string which pandas might produce
        if value is None or value == '':
            return None
        f_val = float(value)
        if math.isnan(f_val) or math.isinf(f_val):
            return None
        return f_val
    except (ValueError, TypeError):
        return None


@app.post("/api/factor_analysis", response_model=FactorAnalysisResult)
async def run_factor_analysis(request: FactorAnalysisRequest):
    """
    Performs Factor Analysis on selected numeric columns of the stored dataset.
    """
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    # Get data from your global store
    headers = stored_datasets["latest"].get("headers")
    data = stored_datasets["latest"].get("data")
    column_types_info = stored_datasets["latest"].get("column_types", {}) # Get type info if available

    if not headers or not data:
        raise HTTPException(status_code=400, detail="Chybí hlavičky nebo data v uloženém datasetu.")

    df_full = pd.DataFrame(data, columns=headers)
    if df_full.empty:
        raise HTTPException(status_code=404, detail="Dataset je prázdný.")

    # --- Input Validation ---
    # Frontend already checks for min 3, but good practice to double-check
    if len(request.columns) < 3:
         raise HTTPException(status_code=400, detail="Pro faktorovou analýzu vyberte alespoň 3 proměnné.")

    missing_cols = [col for col in request.columns if col not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=404, detail=f"Sloupce nenalezeny v datasetu: {', '.join(missing_cols)}")

    # Check if selected columns are considered numeric based on stored types (optional but good)
    non_numeric_selected = []
    if column_types_info:
        for col in request.columns:
            col_info = column_types_info.get(col)
            if col_info and col_info.get("type") != "Číselný":
                non_numeric_selected.append(col)
        if non_numeric_selected:
             logger.warning(f"Pokoušíte se provést FA na sloupcích, které nebyly detekovány jako číselné: {', '.join(non_numeric_selected)}. Výsledky mohou být nesprávné.")
             # You could raise an HTTPException here if you want to be strict
             # raise HTTPException(status_code=400, detail=f"Vybrané sloupce nejsou číselné: {', '.join(non_numeric_selected)}")


    # Select data and ensure numeric types
    try:
        df_selected = df_full[request.columns].copy()
        # Attempt to convert columns to numeric, coercing errors will result in NaN
        for col in request.columns:
             # Replace empty strings with NaN before converting
             df_selected[col] = df_selected[col].replace('', np.nan)
             df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

        # Check if any column is entirely NaN after coercion
        if df_selected.isnull().all().any():
             bad_cols = df_selected.columns[df_selected.isnull().all()].tolist()
             raise ValueError(f"Vybrané sloupce obsahují pouze nečíselné hodnoty nebo jsou prázdné: {', '.join(bad_cols)}")

    except ValueError as e:
         raise HTTPException(status_code=400, detail=f"Chyba při výběru nebo konverzi sloupců na numerický typ: {e}")
    except Exception as e:
        logger.error(f"Neočekávaná chyba při přípravě FA dat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chyba při přípravě dat pro FA: {e}")

    # --- Data Preprocessing ---
    initial_rows = len(df_selected)
    df_clean = df_selected.dropna()
    dropped_rows = initial_rows - len(df_clean)

    if len(df_clean) < len(request.columns) + 1:
         raise HTTPException(status_code=400, detail=f"Nedostatek validních dat po odstranění {dropped_rows} řádků s chybějícími hodnotami ({len(df_clean)} řádků). Potřeba alespoň {len(request.columns) + 1}.")
    if df_clean.empty:
         raise HTTPException(status_code=400, detail="Po odstranění chybějících hodnot nezůstala žádná data.")

    # Standardize if requested
    data_to_analyze = df_clean # Keep as DataFrame for easier handling later
    if request.standardize:
        try:
            scaler = StandardScaler()
            # Fit and transform, result is numpy array
            scaled_values = scaler.fit_transform(df_clean.values)
            # Convert back to DataFrame with original columns and index
            data_to_analyze = pd.DataFrame(scaled_values, columns=df_clean.columns, index=df_clean.index)
            logger.info("Data pro FA byla standardizována.")
        except Exception as e:
             logger.error(f"Chyba při standardizaci dat pro FA: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Chyba při standardizaci dat: {e}")

    # --- Adequacy Tests ---
    kmo_all, kmo_model = None, None
    bartlett_chi_square, bartlett_p_value = None, None
    try:
        # Check for zero variance
        if (data_to_analyze.var() < 1e-10).any(): # Check for near-zero variance
             zero_var_cols = data_to_analyze.columns[data_to_analyze.var() < 1e-10].tolist()
             logger.warning(f"Sloupce s nulovou nebo téměř nulovou variancí nalezeny: {zero_var_cols}. FA může selhat nebo být nespolehlivá.")
             # Don't calculate tests if variance is zero, they will likely fail

        else:
            # KMO Test - requires DataFrame
            try:
                kmo_per_variable, kmo_model = calculate_kmo(data_to_analyze)
                kmo_model = safe_float(kmo_model)
            except np.linalg.LinAlgError:
                 logger.warning("KMO test selhal - pravděpodobně singulární matice.")
                 kmo_model = None
            except ValueError as e:
                 logger.warning(f"KMO test selhal: {e}")
                 kmo_model = None


            # Bartlett's Test - requires DataFrame
            try:
                 chi_square, p_value = calculate_bartlett_sphericity(data_to_analyze)
                 bartlett_chi_square = safe_float(chi_square)
                 bartlett_p_value = safe_float(p_value)
            except ValueError as e:
                 logger.warning(f"Bartlettův test selhal: {e}")
                 bartlett_chi_square = None
                 bartlett_p_value = None


    except Exception as e:
        logger.error(f"Neočekávaná chyba během adequacy testů: {e}", exc_info=True)
        # Continue analysis, but tests will be None

    adequacy_results = DataAdequacy(
        kmo_model=kmo_model,
        bartlett_chi_square=bartlett_chi_square,
        bartlett_p_value=bartlett_p_value
    )

    # --- Determine Number of Factors ---
    n_factors_extracted = 0
    eigenvalues_list: Optional[List[Optional[float]]] = None
    eigenvalue_criterion_used = False

    if request.n_factors:
        n_factors_extracted = request.n_factors
        if n_factors_extracted >= len(request.columns):
             raise HTTPException(
                 status_code=400,
                 detail=f"Počet požadovaných faktorů ({n_factors_extracted}) nemůže být roven nebo větší než počet proměnných ({len(request.columns)})."
             )
        logger.info(f"Použije se {n_factors_extracted} faktorů dle požadavku uživatele.")
    else:
        eigenvalue_criterion_used = True
        logger.info("Určení počtu faktorů pomocí Kaiserova kritéria (eigenvalue > 1).")
        try:
            # Calculate eigenvalues from the CORRELATION matrix of the processed data
            correlation_matrix = data_to_analyze.corr().values
            eigenvalues, _ = np.linalg.eigh(correlation_matrix) # Use eigh for symmetric matrices
            eigenvalues_sorted = sorted(eigenvalues.tolist(), reverse=True) # Convert to list
            eigenvalues_list = [safe_float(ev) for ev in eigenvalues_sorted]

            # Count eigenvalues > 1 (handle None from safe_float)
            n_factors_extracted = sum(ev > 1 for ev in eigenvalues_list if ev is not None)

            if n_factors_extracted == 0:
                logger.warning("Žádné eigenvalue > 1. Extrahuji 1 faktor.")
                n_factors_extracted = 1
            elif n_factors_extracted >= len(request.columns):
                logger.warning(f"Kaiserovo kritérium navrhlo {n_factors_extracted} faktorů (>= počtu proměnných). Redukuji na {max(1, len(request.columns) - 1)}.")
                n_factors_extracted = max(1, len(request.columns) - 1)
            else:
                 logger.info(f"Kaiserovo kritérium navrhlo {n_factors_extracted} faktorů.")

        except np.linalg.LinAlgError:
             logger.error("Chyba při výpočtu eigenvalues (singulární matice?)", exc_info=True)
             raise HTTPException(status_code=500, detail="Chyba při výpočtu eigenvalues (pravděpodobně singulární korelační matice). Zkuste jiné proměnné.")
        except Exception as e:
             logger.error(f"Neočekávaná chyba při určování počtu faktorů: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Neočekávaná chyba při určování počtu faktorů: {e}")

    # --- Fit Factor Analysis Model ---
    # Map 'none' rotation from frontend to None for the library
    rotation_param = None if request.rotation == 'none' else request.rotation

    fa = FactorAnalyzer(
        n_factors=n_factors_extracted,
        rotation=rotation_param,
        method='minres', # Principal Axis Factoring (minimum residual) is common
        use_smc=True
    )

    try:
        logger.info(f"Spouštím FactorAnalyzer s {n_factors_extracted} faktory a rotací '{rotation_param}'...")
        # Fit requires numpy array or DataFrame
        fa.fit(data_to_analyze)
        logger.info("FactorAnalyzer fit dokončen.")
    except ValueError as e:
         logger.error(f"Chyba při fitování modelu Factor Analysis: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Chyba při fitování modelu Factor Analysis: {str(e)}. Zkuste méně faktorů, jiné proměnné, jinou rotaci nebo zkontrolujte data (např. nulová variance).")
    except np.linalg.LinAlgError as e:
         logger.error(f"Lineární algebra chyba při fitování FA: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Chyba lineární algebry při fitování FA: {str(e)}. Časté u singulárních matic (např. perfektní korelace).")
    except Exception as e:
         logger.error(f"Neočekávaná chyba při fitování modelu Factor Analysis: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Neočekávaná chyba při fitování modelu Factor Analysis: {e}")


    # --- Extract Results ---
    try:
        loadings = fa.loadings_ # NumPy array (n_variables, n_factors)
        variance_info = fa.get_factor_variance() # Tuple: (SSL, Percent Var, Cumulative Var)
        communalities_array = fa.get_communalities() # NumPy array

    except Exception as e:
        logger.error(f"Chyba při extrakci výsledků z FA modelu: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chyba při získávání výsledků z analyzátoru faktorů.")

    # --- Format Results ---
    factor_names = [f"Factor{i+1}" for i in range(n_factors_extracted)] # Changed name to match frontend

    # Factor Loadings
    loadings_dict: Dict[str, Dict[str, Optional[float]]] = {}
    variable_names = data_to_analyze.columns.tolist() # Use columns from the analyzed data
    for i, var_name in enumerate(variable_names):
        loadings_dict[var_name] = {}
        for j, factor_name in enumerate(factor_names):
             # Ensure loadings array dimensions match expected factors/variables
             if i < loadings.shape[0] and j < loadings.shape[1]:
                 loadings_dict[var_name][factor_name] = safe_float(loadings[i, j])
             else:
                 loadings_dict[var_name][factor_name] = None # Or handle as error
                 logger.warning(f"Nesoulad dimenzí v loadings pro {var_name}, {factor_name}")


    # Factor Variance
    variance_list: List[FactorVarianceItem] = []
    total_variance_explained = None
    if variance_info and len(variance_info) == 3 and len(variance_info[0]) == n_factors_extracted:
        ssl_values, var_pct_values, cum_var_pct_values = variance_info
        for j, factor_name in enumerate(factor_names):
            ssl = safe_float(ssl_values[j])
            var_pct = safe_float(var_pct_values[j] * 100)
            cum_var_pct = safe_float(cum_var_pct_values[j] * 100)

            variance_list.append(FactorVarianceItem(
                factor=factor_name,
                ssl=ssl if ssl is not None else 0.0,
                variance_pct=var_pct if var_pct is not None else 0.0,
                cumulative_variance_pct=cum_var_pct if cum_var_pct is not None else 0.0,
            ))
        if variance_list:
             total_variance_explained = variance_list[-1].cumulative_variance_pct # Get from last item
    else:
         logger.warning("Nepodařilo se získat validní informace o varianci faktorů.")


    # Communalities
    communalities_dict: Dict[str, Optional[float]] = {}
    if communalities_array is not None and len(communalities_array) == len(variable_names):
        for i, var_name in enumerate(variable_names):
            communalities_dict[var_name] = safe_float(communalities_array[i])
    else:
         logger.warning("Nepodařilo se získat validní informace o komunalitách nebo nesedí délka.")
         for var_name in variable_names:
             communalities_dict[var_name] = None # Fill with None if extraction failed


    # --- Construct Final Result Object ---
    result = FactorAnalysisResult(
        columns_used=variable_names, # Use the actual columns used after cleaning/preprocessing
        n_factors_requested=request.n_factors,
        n_factors_extracted=n_factors_extracted,
        eigenvalue_criterion_used=eigenvalue_criterion_used,
        eigenvalues=eigenvalues_list if eigenvalue_criterion_used else None,
        # Use 'Bez rotace' if rotation was 'none'/'None', else use the requested rotation name
        rotation_used="Bez rotace" if rotation_param is None else request.rotation,
        standardized=request.standardize,
        dropped_rows=dropped_rows,
        data_adequacy=adequacy_results,
        factor_loadings=loadings_dict,
        factor_variance=variance_list,
        total_variance_explained_pct=safe_float(total_variance_explained), # Ensure it's safe float
        communalities=communalities_dict,
    )

    logger.info("Faktorová analýza úspěšně dokončena.")
    return result

@app.get("/api/get_stored_data")
async def get_stored_data():
    if "latest" not in stored_datasets:
        return {"headers": [], "data": []}
    return stored_datasets["latest"]


@app.post("/api/fill_missing")
async def fill_missing(request: Request):
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=400, detail="Data nejsou nahraná.")

    body = await request.json()
    strategies: dict = body.get("strategies", {})

    headers = stored_datasets["latest"]["headers"]
    raw_data = stored_datasets["latest"]["data"]
    prev_mask = stored_datasets["latest"].get("filled_mask")

    df = pd.DataFrame(raw_data, columns=headers)

    # 🧼 Zrušení předchozích doplněných hodnot
    if prev_mask:
        prev_mask_np = np.array(prev_mask)
        for row_idx in range(len(df)):
            for col_idx in range(len(headers)):
                if prev_mask_np[row_idx][col_idx]:
                    df.iat[row_idx, col_idx] = None

    filled_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col, method in strategies.items():
        if col not in df.columns:
            continue

        # 📌 Vždy resetuj chybějící hodnoty na None před výpočtem
        mask = df[col].isna() | (df[col] == "") | (df[col].astype(str).str.lower() == "nan")
        df.loc[mask, col] = None
        original = df[col].copy()

        if method == "mean":
            value = pd.to_numeric(df[col], errors='coerce').mean()
            df.loc[mask, col] = value

        elif method == "median":
            value = pd.to_numeric(df[col], errors='coerce').median()
            df.loc[mask, col] = value

        elif method == "mode":
            mode_series = df[col].mode()
            if not mode_series.empty:
                df.loc[mask, col] = mode_series[0]

        elif method == "zero":
            df.loc[mask, col] = 0

        elif method == "drop":
            df = df[~mask]
            filled_mask = filled_mask.loc[df.index]
            continue

        elif method == "ffill":
            df[col] = df[col].replace("", pd.NA).fillna(method='ffill').fillna(method='bfill')

        elif method == "bfill":
            df[col] = df[col].replace("", pd.NA).fillna(method='bfill').fillna(method='ffill')

        elif method == "interpolate":
            df[col] = pd.to_numeric(df[col], errors='coerce').interpolate(method='linear', limit_direction='both')

        # Označ změny v masce
        filled_mask[col] = mask & (df[col] != original)

    # 📦 Ulož výsledek
    stored_datasets["latest"] = {
        "headers": df.columns.tolist(),
        "data": df.astype(str).where(pd.notna(df), "").values.tolist(),
        "filled_mask": filled_mask.values.tolist()
    }

    return {
        "status": "filled",
        "filled_cells": int(filled_mask.values.sum()),
        "headers": df.columns.tolist(),
        "data": df.astype(str).where(pd.notna(df), "").values.tolist(),
        "filled_mask": filled_mask.values.tolist()
    }


@app.post("/api/analyze")
async def analyze_data(request: DataRequest):
    logger.info("Received request for /api/analyze")
    column_types_result = {}

    try:
        # --- MINIMALISTICKÁ ZMĚNA: Zajistit existenci 'latest' ---
        if "latest" not in stored_datasets:
            logger.info("Creating 'latest' key in stored_datasets.")
            stored_datasets["latest"] = {}
        # --- KONEC MINIMALISTICKÉ ZMĚNY ---

        # Validace vstupu
        if not request.headers:
             logger.warning("/api/analyze: Received empty headers.")
             stored_datasets["latest"]["column_types"] = {} # Uložit prázdný, aby klíč existoval
             return {"column_types": {}}

        df = pd.DataFrame(request.data or [], columns=request.headers)
        logger.info(f"/api/analyze: DataFrame shape {df.shape}")

        if df.empty and request.headers:
             logger.warning("/api/analyze: DataFrame is empty but headers exist.")
             # Vytvoříme záznamy s neznámým typem pro hlavičky
             for col in request.headers:
                 column_types_result[col] = {"type": "Neznámý", "missing": "100.00%"}
             stored_datasets["latest"]["column_types"] = column_types_result
             return {"column_types": column_types_result}
        elif df.empty:
             logger.warning("/api/analyze: DataFrame is empty and no headers.")
             stored_datasets["latest"]["column_types"] = {}
             return {"column_types": {}}


        df = df.replace(['', None], np.nan)

        missing_data = df.isna().mean() * 100
        total_rows = len(df)

        for column in request.headers:
            if column not in df.columns:
                 column_types_result[column] = {"type": "Neznámý", "missing": "100.00%"}
                 continue

            series = df[column].dropna()
            unique_values = series.nunique()

            # Vaše logika detekce prahu
            if total_rows < 200: category_threshold = 8
            elif total_rows < 1000: category_threshold = 10
            else: category_threshold = 15

            # Vaše logika detekce typu
            is_numeric = pd.to_numeric(series, errors='coerce').notna().all() if not series.empty else False # Přidána kontrola na prázdnou sérii
            detected_type = "Kategorie"
            if is_numeric:
                if unique_values > category_threshold:
                    detected_type = "Číselný"

            column_types_result[column] = {
                "type": detected_type,
                "missing": f"{missing_data.get(column, 0):.2f}%"
            }

    except Exception as e:
        logger.error(f"Error during /api/analyze: {e}", exc_info=True)
        # I při chybě zajistíme existenci klíče
        if "latest" not in stored_datasets:
            stored_datasets["latest"] = {}
        stored_datasets["latest"]["column_types"] = {} # Uložit prázdný
        raise HTTPException(status_code=500, detail=f"Chyba při analýze dat: {str(e)}")

    # --- Uložení výsledku ---
    # Klíč 'latest' by už měl existovat z kontroly na začátku
    stored_datasets["latest"]["column_types"] = column_types_result
    logger.info(f"/api/analyze completed. Stored/Updated 'column_types' for keys: {list(column_types_result.keys())}")

    return {"column_types": column_types_result}

class RecalculateSingleNormalityRequest(BaseModel):
    column: str      # Název sloupce k přepočtu
    test_method: str # Požadovaná metoda ('shapiro' nebo 'ks')

class NormalityResultModel(BaseModel):
    column: str
    test: str
    pValue: float
    isNormal: bool
    warning: str
    hasMissing: bool

@app.post("/api/recalculate_single_normality", response_model=NormalityResultModel)
async def recalculate_single_column_normality(request: RecalculateSingleNormalityRequest):
    """
    Zcela samostatně přepočítá normalitu pro JEDEN specifický sloupec
    s použitím explicitně zadané testovací metody.
    Načítá si data znovu a neovlivňuje ostatní endpointy.
    """
    column_name = request.column
    test_method = request.test_method

    # 1. Získání dat (stejně jako by to dělal jiný endpoint)
    if "latest" not in stored_datasets:
        raise HTTPException(status_code=404, detail="Data nejsou nahraná v paměti serveru.")

    dataset_info = stored_datasets["latest"]
    headers = dataset_info.get("headers")
    data = dataset_info.get("data")

    if not headers or not data:
         raise HTTPException(status_code=400, detail="V uložených datech chybí hlavičky nebo data.")

    # 2. Validace vstupů specifických pro tento požadavek
    if column_name not in headers:
         raise HTTPException(status_code=404, detail=f"Požadovaný sloupec '{column_name}' nebyl v datech nalezen.")
    if test_method not in ["shapiro", "ks"]:
        raise HTTPException(status_code=400, detail="Neznámá testovací metoda. Povolené hodnoty jsou 'shapiro' nebo 'ks'.")

    # 3. Zpracování dat pro daný sloupec (izolovaně)
    try:
        df = pd.DataFrame(data, columns=headers)
        if column_name not in df.columns:
             # Dvojitá kontrola pro jistotu, i když už máme kontrolu v headers
             raise HTTPException(status_code=404, detail=f"Sloupec '{column_name}' se nepodařilo najít v DataFrame.")

        original_series = df[column_name] # Získáme původní sérii
        # Pokus o konverzi na numerický typ, nevalidní hodnoty budou NaN
        numeric_series_coerced = pd.to_numeric(original_series, errors='coerce')
        # Odstraníme NaN hodnoty pro samotný test
        series_cleaned = numeric_series_coerced.dropna()

        # 4. Základní kontroly na vyčištěných datech
        if not pd.api.types.is_numeric_dtype(series_cleaned):
             # Pokud ani po dropna není numerický (např. původně jen text), nemůžeme testovat
             raise HTTPException(status_code=400, detail=f"Sloupec '{column_name}' neobsahuje číselná data nebo jen chybějící hodnoty.")
        if series_cleaned.empty:
             raise HTTPException(status_code=400, detail=f"Sloupec '{column_name}' neobsahuje žádné platné číselné hodnoty po odstranění chybějících.")
        if len(series_cleaned) < 3:
            # Oba testy vyžadují alespoň 3 body
            raise HTTPException(status_code=400, detail=f"Nedostatek platných hodnot (< 3) ve sloupci '{column_name}' pro provedení testu '{test_method}'. Nalezeno: {len(series_cleaned)}.")

        # 5. Výpočet pomocných informací (chybějící hodnoty, outliery)
        total_count = len(original_series)
        missing_count = numeric_series_coerced.isna().sum() # Počet NaN po pokusu o konverzi
        has_missing = missing_count > 0

        outlier_note = ""
        if len(series_cleaned) > 1: # Potřebujeme alespoň 2 body pro std dev
            std_dev = series_cleaned.std(ddof=1)
            if std_dev is not None and not np.isnan(std_dev) and std_dev > 0:
                z_scores = zscore(series_cleaned, ddof=1)
                outlier_ratio = np.mean(np.abs(z_scores) > 3)
                if outlier_ratio > 0.05:
                    outlier_note = "Pozor: detekováno >5% potenciálních outlierů (Z-skóre > 3), může ovlivnit výsledek."
            elif std_dev == 0:
                 outlier_note = "Pozn: Všechny platné hodnoty jsou stejné."
            # else: std_dev je None nebo NaN - nemůžeme počítat z-scores

        # 6. Provedení vyžádaného testu normality
        test_used = ""
        p_value_float = float('nan') # Defaultní hodnota pro případ chyby
        reason = ""
        stat = None

        try:
            if test_method == "shapiro":
                if len(series_cleaned) > 5000:
                    # Fallback pro Shapiro-Wilk nad 5000 hodnot
                    standardized = (series_cleaned - series_cleaned.mean()) / series_cleaned.std(ddof=1)
                    stat, p = kstest(standardized, "norm")
                    p_value_float = float(f"{p:.6g}")
                    test_used = "Kolmogorov–Smirnov (fallback z Shapiro >5000)"
                    reason = f"Explicitně vyžádán Shapiro-Wilk, ale počet hodnot (N={len(series_cleaned)}) > 5000. Použit K-S."
                else:
                    # Standardní Shapiro-Wilk
                    stat, p = shapiro(series_cleaned)
                    p_value_float = float(f"{p:.6g}")
                    test_used = "Shapiro-Wilk"
                    reason = f"Explicitně vyžádán test {test_used} (N={len(series_cleaned)})."

            elif test_method == "ks":
                # Kolmogorov-Smirnov test
                standardized = (series_cleaned - series_cleaned.mean()) / series_cleaned.std(ddof=1)
                stat, p = kstest(standardized, "norm")
                p_value_float = float(f"{p:.6g}")
                test_used = "Kolmogorov–Smirnov"
                reason = f"Explicitně vyžádán test {test_used} (N={len(series_cleaned)})."

        except ValueError as ve:
            # Chyba přímo v testovací funkci (např. teoreticky N<3 i když jsme kontrolovali)
             raise HTTPException(status_code=400, detail=f"Chyba při provádění testu '{test_method}' pro sloupec '{column_name}': {ve}")

        # 7. Sestavení finální poznámky (warning)
        note_parts = [reason] # Začneme důvodem výběru testu
        if outlier_note:
             note_parts.append(outlier_note)
        if has_missing:
            note_parts.append(f"Původní sloupec obsahuje {missing_count} chybějících/nečíselných hodnot z {total_count}.")
        final_warning = " ".join(note_parts) if note_parts else "-"

        # 8. Sestavení a vrácení výsledku
        is_normal = bool(not np.isnan(p_value_float) and p_value_float > 0.05)

        return NormalityResultModel(
            column=column_name,
            test=test_used,
            pValue=p_value_float,
            isNormal=is_normal,
            warning=final_warning,
            hasMissing=has_missing
        )

    except HTTPException:
        # Pokud byla vyhozena HTTPException už dříve (validace vstupů, dat),
        # necháme ji projít dál.
        raise
    except Exception as e:
        # Záchyt jakékoli jiné neočekávané chyby během zpracování
        print(f"Neočekávaná chyba při přepočtu normality pro sloupec {column_name} metodou {test_method}: {e}")
        # Můžeme zde logovat celé traceback: import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Interní serverová chyba při přepočtu normality pro sloupec '{column_name}'. Detail: {e}")