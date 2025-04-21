import React, { useEffect, useState, useCallback } from 'react';
import Plot from 'react-plotly.js';

// --- Rozhraní (Interfaces) ---
interface ColumnType {
    name: string;
    type: string;
}

// Rozšířený interface pro koeficienty
interface Coefficient {
    name: string;
    coef: number | number[]; // Číslo nebo pole pro multinomiální
    stderr?: number | null;   // Standard Error
    t_value?: number | null;  // t-statistic (pro OLS)
    z_value?: number | null;  // z-statistic (pro Logit/MNLogit) - přidáno
    p_value?: number | null;  // p-value
    ciLow?: number | null;    // Dolní mez CI
    ciHigh?: number | null;   // Horní mez CI
}

interface OverfittingInfo {
    r2_train: number;
    r2_test: number;
    difference: number;
    is_overfitting: boolean;
}

interface ScatterData {
    x: number[];
    y_true: number[];
    y_pred: number[];
}

interface Residuals {
    predicted: number[];
    residuals: number[];
}

// Rozšířený interface pro výsledky regrese
interface RegressionResult {
    coefficients: Coefficient[];
    intercept?: number | number[]; // Může být číslo nebo pole (pro MNLogit z sklearn)
    r2?: number | null;
    r2_adjusted?: number | null;
    mse?: number | null;
    rmse?: number | null;
    f_statistic?: number | null;
    f_pvalue?: number | null; // p-hodnota F-testu pro OLS
    pseudo_r2?: number | null; // Pseudo R^2 pro Logit/MNLogit
    log_likelihood?: number | null; // Log-likelihood pro Logit/MNLogit
    llr_p_value?: number | null; // p-hodnota LR testu pro Logit/MNLogit
    n_observations?: number | null; // Počet pozorování
    accuracy?: number | null; // Pro výsledky klasifikace
    overfitting?: OverfittingInfo | null; // Ponecháno, i když se teď nepoužívá
    scatter_data?: ScatterData | null;
    residuals?: Residuals | null;
    method: string;
    reason: string;
    note?: string | null; // Poznámka z backendu
    warnings?: string | string[] | null; // Varování z backendu
}

// --- Pomocné funkce ---
const formatNum = (num: number | undefined | null, digits = 3): string => {
    if (typeof num !== 'number' || num === null || isNaN(num)) {
        return '–'; // Použijeme pomlčku pro neplatné/chybějící hodnoty
    }
    // Zvážit formátování pro velmi malé p-hodnoty, ale to dělá getSignificanceStars
    return num.toFixed(digits);
};

const formatPValue = (pValue: number | undefined | null, digits = 4): string => {
    if (typeof pValue !== 'number' || pValue === null || isNaN(pValue)) {
        return '–';
    }
    if (pValue < 0.0001) { // Pokud je velmi malé, použij notaci
        return '<0.0001';
    }
    return pValue.toFixed(digits);
};


const getSignificanceStars = (pValue: number | undefined | null): string => {
    if (typeof pValue !== 'number' || pValue === null || isNaN(pValue)) {
        return '';
    }
    if (pValue < 0.001) return '***';
    if (pValue < 0.01) return '**';
    if (pValue < 0.05) return '*';
    // if (pValue < 0.1) return '.'; // Alternativa
    return '';
};


// --- Hlavní Komponenta ---
export default function RegressionAnalysis() {
    // --- Stavy pro výběr sloupců a metody ---
    const [allColumns, setAllColumns] = useState<ColumnType[]>([]);
    const [yVar, setYVar] = useState<string>('');
    const [xVars, setXVars] = useState<string[]>([]);
    const [method, setMethod] = useState<string>('auto');

    // --- Stavy pro výsledky analýzy a chyby ---
    const [result, setResult] = useState<RegressionResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    // --- Stavy pro načítání ---
    const [isLoadingColumns, setIsLoadingColumns] = useState<boolean>(true);
    const [isLoadingAnalysis, setIsLoadingAnalysis] = useState<boolean>(false);

    // --- Stavy pro AI interpretaci ---
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    const [isInterpreting, setIsInterpreting] = useState<boolean>(false);
    const [aiError, setAiError] = useState<string | null>(null);

    // --- Funkce pro načtení sloupců ---
    const fetchColumns = useCallback(async () => {
        setIsLoadingColumns(true);
        setError(null);
        setResult(null);
        setAiInterpretation(null);
        setAiError(null);
        try {
            // TODO: Nahradit URL správnou cestou k vašemu API
            const res = await fetch("http://localhost:8000/api/get_column_types");
            if (!res.ok) throw new Error(`Nepodařilo se načíst sloupce: ${res.statusText} (status: ${res.status})`);
            // Ošetření prázdné odpovědi nebo ne-JSON odpovědi
            const text = await res.text();
            if (!text) {
                console.warn("API vrátilo prázdnou odpověď pro get_column_types");
                setAllColumns([]);
                return; // Skončíme, pokud není co parsovat
            }
            const data: ColumnType[] = JSON.parse(text); // Parsovat text explicitně
            setAllColumns(data || []); // || [] pro případ null/undefined z JSON
        } catch (err: any) {
            console.error("Column Fetch Error:", err);
            setError(`Chyba načítání sloupců: ${err.message}`);
            setAllColumns([]); // Reset na prázdné pole při chybě
        } finally {
            setIsLoadingColumns(false);
        }
    }, []); // Prázdné pole závislostí, fetchColumns se nemění

    useEffect(() => {
        fetchColumns();
    }, [fetchColumns]); // fetchColumns je nyní v useCallback

    // --- Funkce pro resetování ---
    const resetResultsAndInterpretation = useCallback(() => {
        setResult(null);
        setError(null);
        setAiInterpretation(null);
        setAiError(null);
    }, []);

    // --- Handlery pro změnu vstupů ---
    const handleYChange = useCallback((selectedY: string) => {
        setYVar(selectedY);
        // Odstraníme Y z X, pokud tam bylo
        setXVars(prevX => prevX.filter(col => col !== selectedY));
        resetResultsAndInterpretation();
    }, [resetResultsAndInterpretation]);

    const handleXToggle = useCallback((colName: string, isChecked: boolean) => {
        setXVars(prevX => {
            if (isChecked && colName === yVar) return prevX; // Nelze vybrat Y jako X
            const newX = isChecked ? [...prevX, colName] : prevX.filter(c => c !== colName);
            resetResultsAndInterpretation();
            return newX;
        });
    }, [yVar, resetResultsAndInterpretation]); // Přidána závislost yVar

    const handleMethodChange = useCallback((newMethod: string) => {
        setMethod(newMethod);
        resetResultsAndInterpretation();
    }, [resetResultsAndInterpretation]);

    // --- Handler pro spuštění analýzy ---
    const handleRun = useCallback(async () => {
        if (!yVar || xVars.length === 0) {
            setError("Vyberte prosím závislou a alespoň jednu nezávislou proměnnou.");
            setResult(null);
            setAiInterpretation(null);
            setAiError(null);
            return;
        }

        resetResultsAndInterpretation();
        setIsLoadingAnalysis(true);

        try {
            // TODO: Nahradit URL správnou cestou k vašemu API
            const response = await fetch("http://localhost:8000/api/regression_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ y: yVar, x: xVars, method }),
            });

            // Zpracování odpovědi (včetně chybových stavů)
            const responseData = await response.json();

            if (!response.ok) {
                // Preferujeme 'detail' z FastAPI HTTPException, jinak obecná zpráva
                const errorMessage = responseData.detail || `Chyba ${response.status}: ${response.statusText}`;
                console.error("Backend Error Response:", responseData); // Logování celé odpovědi
                setError(errorMessage);
                setResult(null);
            } else {
                console.log("Received Regression Result:", responseData); // Log úspěšných dat
                setResult(responseData as RegressionResult);
                setError(null); // Vyčistit chybu po úspěchu
            }
        } catch (err: any) {
            console.error("Regression Analysis Fetch/Parse Error:", err);
            // Zobrazit obecnější chybu, pokud selže komunikace nebo parsování
            setError(`Chyba komunikace nebo zpracování odpovědi: ${err.message}`);
            setResult(null);
        } finally {
            setIsLoadingAnalysis(false);
        }
    }, [yVar, xVars, method, resetResultsAndInterpretation]); // Závislosti

    // --- Handler pro AI interpretaci ---
    const handleInterpret = useCallback(async () => {
        if (!result) return; // Bez výsledků není co interpretovat

        setIsInterpreting(true);
        setAiInterpretation(null);
        setAiError(null);

        // Připravení payloadu pro AI
        const interpretationPayload = {
            analysis_type: "regression",
            method: result.method,
            dependent_variable: yVar,
            independent_variables: xVars,
            results_summary: {
                r2: result.r2,
                r2_adjusted: result.r2_adjusted,
                rmse: result.rmse,
                accuracy: result.accuracy,
                pseudo_r2: result.pseudo_r2, // Přidáno
                // overfitting_detected: result.overfitting?.is_overfitting, // Overfitting se teď nepoužívá
            },
            // Posíláme jen hodnoty koeficientů pro jednoduchost interpretace
            coefficients: result.coefficients.map(c => ({
                name: c.name,
                value: Array.isArray(c.coef) ? (c.coef.length > 0 ? c.coef[0] : null) : c.coef,
                significant: c.p_value != null && c.p_value < 0.05 // Přidáme info o signifikanci
            })),
            intercept: Array.isArray(result.intercept) ? (result.intercept.length > 0 ? result.intercept[0] : null) : result.intercept,
            n_observations: result.n_observations, // Přidáno
        };

        try {
            // TODO: Nahradit URL správnou cestou k vašemu API
            const response = await fetch("http://localhost:8000/api/interpret_regression", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(interpretationPayload),
            });
            const data = await response.json();

            if (!response.ok) {
                const errorMessage = data.detail || data.error || `Chyba AI interpretace (${response.status})`;
                throw new Error(errorMessage);
            }

            const interpretationText = data.interpretation || data.choices?.[0]?.message?.content;
            if (interpretationText) {
                setAiInterpretation(interpretationText);
            } else {
                console.error("Neočekávaný formát odpovědi od AI:", data);
                throw new Error("Nepodařilo se získat text interpretace.");
            }
        } catch (err: any) {
            console.error("AI Interpretation Error:", err);
            setAiError(`Chyba AI interpretace: ${err.message}`);
        } finally {
            setIsInterpreting(false);
        }
    }, [result, yVar, xVars]); // Závislosti

    // --- Pomocné proměnné pro podmíněné zobrazení ---
    // Zda jde o klasifikační metodu (podle backendu)
    const isClassification = result?.method === "logistic" || result?.method === "multinomial";
    // Zda máme statistiky (p-hodnoty atd.), což indikuje použití statsmodels
    const hasAdvancedStats = result?.coefficients?.some(c => c.p_value != null);

    // Podmínky pro zobrazení grafů
    const showScatterPlot = !isClassification && // Ne pro klasifikaci
        xVars.length === 1 && // Jen pro jednoduchou regresi
        result?.scatter_data?.x != null &&
        Array.isArray(result.scatter_data.x) &&
        Array.isArray(result.scatter_data.y_true) &&
        Array.isArray(result.scatter_data.y_pred) &&
        result.scatter_data.x.length > 0;

    const showResidualPlot = !isClassification && // Ne pro klasifikaci
        result?.residuals?.predicted != null &&
        Array.isArray(result.residuals.predicted) &&
        Array.isArray(result.residuals.residuals) &&
        result.residuals.predicted.length > 0;

    // --- JSX Struktura Komponenty ---
    return (
        <div className="space-y-6 p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
            {/* Sekce Vstupů */}
            <fieldset disabled={isLoadingAnalysis || isLoadingColumns} className="space-y-4">
                {/* Výběr Y */}
                <div>
                    <label htmlFor="y-select" className="block text-sm font-medium text-gray-800 dark:text-gray-200 mb-1">1. Závislá proměnná (Y):</label>
                    {isLoadingColumns ? (
                        <p className="text-xs text-gray-500 dark:text-gray-400">Načítám sloupce...</p>
                    ) : (
                        <select
                            id="y-select"
                            value={yVar}
                            onChange={e => handleYChange(e.target.value)}
                            className="mt-1 border border-gray-300 rounded px-3 py-2 text-sm w-full max-w-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-60"
                            aria-label="Vyberte závislou proměnnou"
                        >
                            <option value="">-- Vyberte Y --</option>
                            {allColumns.map(col => (
                                <option key={col.name} value={col.name} disabled={xVars.includes(col.name)}>
                                    {col.name} ({col.type})
                                </option>
                            ))}
                        </select>
                    )}
                </div>

                {/* Výběr X */}
                <div>
                    <p className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2">2. Nezávislé proměnné (X):</p>
                    {isLoadingColumns ? (
                        <p className="text-xs text-gray-500 dark:text-gray-400">Načítám sloupce...</p>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-x-4 gap-y-2 max-h-48 overflow-y-auto p-2 border rounded dark:border-gray-600">
                            {allColumns.map(col => (
                                <label key={col.name} htmlFor={`x-col-${col.name}`} className={`flex items-center gap-2 text-sm p-1 rounded ${col.name === yVar ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer'}`}>
                                    <input
                                        id={`x-col-${col.name}`}
                                        type="checkbox"
                                        checked={xVars.includes(col.name)}
                                        onChange={e => handleXToggle(col.name, e.target.checked)}
                                        disabled={col.name === yVar}
                                        className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 disabled:opacity-50"
                                    />
                                    <span className={col.name === yVar ? 'line-through text-gray-400 dark:text-gray-500' : 'text-gray-700 dark:text-gray-300'}>{col.name} ({col.type})</span>
                                </label>
                            ))}
                        </div>
                    )}
                </div>

                {/* Výběr Metody */}
                <div>
                    <label htmlFor="reg-method" className="block text-sm font-medium text-gray-800 dark:text-gray-200 mb-1">3. Metoda regrese:</label>
                    <select
                        id="reg-method"
                        value={method}
                        onChange={(e) => handleMethodChange(e.target.value)}
                        className="mt-1 border border-gray-300 rounded px-3 py-2 text-sm w-full max-w-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-60"
                        aria-label="Vyberte metodu regrese"
                    >
                        <option value="auto">Automaticky (doporučeno)</option>
                        <option value="ols">Lineární regrese (OLS)</option>
                        <option value="ridge">Ridge regrese</option>
                        <option value="lasso">Lasso regrese</option>
                        <option value="elasticnet">ElasticNet</option>
                        <option value="logistic">Logistická regrese (binární Y)</option>
                        <option value="multinomial">Multinomiální regrese (kategorické Y)</option>
                    </select>
                </div>
            </fieldset>

            {/* Tlačítko Spustit */}
            <button
                onClick={handleRun}
                disabled={!yVar || xVars.length === 0 || isLoadingAnalysis || isLoadingColumns}
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out"
            >
                {isLoadingAnalysis ? (
                    <>
                        <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                        Analyzuji...
                    </>
                ) : "Spustit regresní analýzu"}
            </button>

            {/* Zobrazení Chyby Analýzy */}
            {error && (
                <div role="alert" className="p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                    ⚠️ Chyba analýzy: {error}
                </div>
            )}

            {/* Sekce Výsledků Analýzy */}
            {result && !isLoadingAnalysis && (
                <div className="space-y-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                    {/* --- Shrnutí Metrik --- */}
                    <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded border border-gray-200 dark:border-gray-600 text-sm">
                        <p className="mb-2">
                            <strong className="font-medium text-gray-800 dark:text-gray-200">Použitá metoda:</strong>
                            <span className="ml-2 font-semibold capitalize text-indigo-700 dark:text-indigo-400">{result.method}</span>
                        </p>
                        {result.reason && <p className="text-gray-600 dark:text-gray-400 text-xs mb-3">{result.reason}</p>}
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-1 text-gray-700 dark:text-gray-300">
                            {/* Klasifikační metriky */}
                            {isClassification && (
                                <>
                                    <p><strong>Přesnost (Accuracy):</strong> {formatNum((result.accuracy ?? 0) * 100, 1)} %</p>
                                    {/* Další klasifikační metriky ze statsmodels */}
                                    {result.pseudo_r2 != null && <p><strong>Pseudo R²:</strong> {formatNum(result.pseudo_r2)}</p>}
                                    {result.log_likelihood != null && <p><strong>Log-Likelihood:</strong> {formatNum(result.log_likelihood, 1)}</p>}
                                    {result.llr_p_value != null && <p><strong>LLR p-hodnota:</strong> {formatPValue(result.llr_p_value)}</p>}
                                </>
                            )}
                            {/* Regresní metriky */}
                            {!isClassification && (
                                <>
                                    {result.r2 != null && <p><strong>R²:</strong> {formatNum(result.r2)}</p>}
                                    {result.r2_adjusted != null && <p><strong>Adjusted R²:</strong> {formatNum(result.r2_adjusted)}</p>}
                                    {result.mse != null && <p><strong>MSE:</strong> {formatNum(result.mse)}</p>}
                                    {result.rmse != null && <p><strong>RMSE:</strong> {formatNum(result.rmse)}</p>}
                                    {result.f_statistic != null && <p><strong>F-statistik:</strong> {formatNum(result.f_statistic, 2)}</p>}
                                    {result.f_pvalue != null && <p><strong>F-test p-hodnota:</strong> {formatPValue(result.f_pvalue)}</p>}
                                </>
                            )}
                            {/* Společná metrika */}
                            {result.n_observations != null && <p><strong>Počet pozorování:</strong> {result.n_observations}</p>}
                        </div>
                        {/* Poznámka z backendu */}
                        {result.note && <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">{result.note}</p>}
                        {/* Varování z backendu */}
                        {result.warnings && (
                            <div className="mt-2 p-2 bg-yellow-100 border border-yellow-300 text-yellow-800 rounded text-xs dark:bg-yellow-900/30 dark:text-yellow-200 dark:border-yellow-700">
                                <strong>Varování modelu:</strong>
                                <pre className="whitespace-pre-wrap font-mono text-xs mt-1">{Array.isArray(result.warnings) ? result.warnings.join('\n') : result.warnings}</pre>
                            </div>
                        )}
                    </div>

                    {/* --- Tabulka Koeficientů --- */}
                    <div>
                        <h4 className="text-base font-semibold mb-2 text-gray-800 dark:text-gray-100">Odhadnuté koeficienty</h4>
                        {/* Div pro scrollbar a max výšku */}
                        <div className="overflow-x-auto max-h-96 overflow-y-auto border border-gray-200 dark:border-gray-600 rounded-md shadow-sm">
                            <table className="min-w-full text-sm divide-y divide-gray-200 dark:divide-gray-700">
                                <thead className="bg-gray-100 dark:bg-gray-700 sticky top-0 z-10">
                                <tr>
                                    <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Proměnná</th>
                                    <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Odhad</th>
                                    {/* Hlavičky pro statistiky (pokud existují) */}
                                    {hasAdvancedStats && (
                                        <>
                                            <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Std. Chyba</th>
                                            <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">t/z-hodnota</th>
                                            <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">p-hodnota</th>
                                            <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">95% CI</th>
                                        </>
                                    )}
                                    {/* Hlavička pro CI u sklearn regresí (pokud nemají statistiky) */}
                                    {!hasAdvancedStats && !isClassification && (
                                        <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">95% CI</th>
                                    )}
                                </tr>
                                </thead>
                                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                                {/* Zobrazení interceptu (pouze pokud není klasifikace) */}
                                {!isClassification && result?.intercept != null && (
                                    <tr className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                        <td className="px-3 py-2 font-medium text-gray-900 dark:text-gray-100">(Intercept)</td>
                                        <td className="px-3 py-2 text-center font-mono text-gray-700 dark:text-gray-300">
                                            {formatNum(result.intercept as number)}
                                        </td>
                                        {/* Prázdné buňky pro statistiky interceptu */}
                                        {hasAdvancedStats && (
                                            <>
                                                <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400"></td>
                                                <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400"></td>
                                                <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400"></td>
                                                <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400">N/A</td>
                                            </>
                                        )}
                                        {!hasAdvancedStats && !isClassification && (
                                            <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400">N/A</td>
                                        )}
                                    </tr>
                                )}

                                {/* Mapování koeficientů */}
                                {result?.coefficients?.map((c: any, idx) => { // Použití 'any' pro flexibilitu s t/z
                                    const hasStats = c.p_value != null;
                                    const t_or_z_value = c.t_value ?? c.z_value;

                                    return (
                                        <tr key={c.name || idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                            {/* Název proměnné */}
                                            <td className="px-3 py-2 text-gray-800 dark:text-gray-200 whitespace-nowrap">{c.name}</td>
                                            {/* Odhad (Koeficient) */}
                                            <td className="px-3 py-2 text-center font-mono text-gray-700 dark:text-gray-300">
                                                {Array.isArray(c.coef) ? c.coef.map(coef => formatNum(coef)).join(' | ') : formatNum(c.coef)}
                                                {/* Hvězdičky signifikance */}
                                                {hasStats && <span className="ml-1 text-red-600 dark:text-red-400 font-bold">{getSignificanceStars(c.p_value)}</span>}
                                            </td>
                                            {/* Statistiky (pokud existují) */}
                                            {hasStats && (
                                                <>
                                                    <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400">{formatNum(c.stderr)}</td>
                                                    <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400">{formatNum(t_or_z_value)}</td>
                                                    <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400">{formatPValue(c.p_value)}</td>
                                                    <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400">
                                                        {(c.ciLow != null && c.ciHigh != null) ? `[${formatNum(c.ciLow)}, ${formatNum(c.ciHigh)}]` : "–"}
                                                    </td>
                                                </>
                                            )}
                                            {/* CI pro sklearn regrese (pokud nemají statistiky) */}
                                            {!hasStats && !isClassification && (
                                                <td className="px-3 py-2 text-center font-mono text-xs text-gray-600 dark:text-gray-400">
                                                    {(c.ciLow != null && c.ciHigh != null) ? `[${formatNum(c.ciLow)}, ${formatNum(c.ciHigh)}]` : "–"}
                                                </td>
                                            )}
                                        </tr>
                                    );
                                })}
                                </tbody>
                            </table>
                        </div>
                        {/* Legenda a poznámky pod tabulkou */}
                        {hasAdvancedStats && (
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                Signif. kódy: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
                            </p>
                        )}
                        {/* Poznámka pro multinomiální (zůstává) */}
                        {isClassification && result.coefficients?.some((c: any) => c.name?.includes('(třída:')) && (
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Pozn.: Pro multinomiální regresi je zobrazen koeficient pro každou třídu (vs. referenční).</p>
                        )}
                    </div>

                    {/* --- Informace o Overfittingu (skryto, protože se nepoužívá) --- */}
                    {/* {!isClassification && result.overfitting && ( ... )} */}

                    {/* --- Grafy --- */}
                    <div className='grid grid-cols-1 lg:grid-cols-2 gap-6'>
                        {/* Scatter Plot */}
                        {showScatterPlot && (
                            <div className="border rounded-lg p-2 bg-white dark:bg-gray-800 shadow-sm">
                                <h4 className="text-base font-semibold mb-1 text-gray-800 dark:text-gray-100 px-1">Vztah Y vs X</h4>
                                <Plot
                                    // data a layout zůstávají stejné
                                    data={[
                                        { x: result!.scatter_data!.x, y: result!.scatter_data!.y_true, mode: 'markers', name: 'Skutečné', type: 'scatter', marker: { color: '#a0aec0', size: 5, opacity: 0.7 } },
                                        // Seřazení dat pro linii predikce
                                        {
                                            x: [...result!.scatter_data!.x].sort((a, b) => a - b),
                                            // Seřadíme y_pred podle seřazeného x
                                            y: result!.scatter_data!.x
                                                .map((val, i) => ({ x: val, y: result!.scatter_data!.y_pred[i] })) // Spojíme x a y_pred
                                                .sort((a, b) => a.x - b.x) // Seřadíme podle x
                                                .map(pair => pair.y), // Vezmeme seřazené y_pred
                                            mode: 'lines', name: 'Predikce', type: 'scatter', line: { color: '#e53e3e', width: 2 }
                                        }
                                    ]}
                                    layout={{
                                        autosize: true,
                                        xaxis: { title: xVars[0], zeroline: false, color: 'rgb(55 65 81)', gridcolor: 'rgba(203, 213, 225, 0.5)' }, // Barvy os a mřížky
                                        yaxis: { title: yVar, zeroline: false, color: 'rgb(55 65 81)', gridcolor: 'rgba(203, 213, 225, 0.5)' },
                                        margin: { l: 50, r: 20, t: 30, b: 50 },
                                        hovermode: 'closest',
                                        showlegend: true,
                                        legend: { y: 1.1, orientation: 'h' },
                                        paper_bgcolor: 'rgba(0,0,0,0)', // Průhledné pozadí
                                        plot_bgcolor: 'rgba(0,0,0,0)',  // Průhledné pozadí plotu
                                        font: { color: 'rgb(55 65 81)' } // Barva písma (pro světlý i tmavý mód)
                                    }}
                                    useResizeHandler={true}
                                    className="w-full h-[450px]"
                                    config={{ responsive: true, displayModeBar: false }} // Skrytí nástrojové lišty Plotly
                                />
                            </div>
                        )}

                        {/* Residual Plot */}
                        {showResidualPlot && (
                            <div className={`border rounded-lg p-2 bg-white dark:bg-gray-800 shadow-sm ${!showScatterPlot ? 'lg:col-span-2' : ''}`}>
                                <h4 className="text-base font-semibold mb-1 text-gray-800 dark:text-gray-100 px-1">Analýza reziduí</h4>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2 px-1">Graf reziduí (chyb) vůči predikovaným hodnotám. Body by měly být náhodně rozptýleny kolem nuly.</p>
                                <Plot
                                    // data a layout zůstávají stejné
                                    data={[
                                        { x: result!.residuals!.predicted, y: result!.residuals!.residuals, mode: 'markers', type: 'scatter', name: 'Rezidua', marker: { color: '#4a5568', size: 5, opacity: 0.7 } }
                                    ]}
                                    layout={{
                                        autosize: true,
                                        xaxis: { title: 'Predikovaná hodnota (Y pred)', zeroline: false, color: 'rgb(55 65 81)', gridcolor: 'rgba(203, 213, 225, 0.5)' },
                                        yaxis: { title: 'Reziduum (Y true - Y pred)', zeroline: true, zerolinecolor: '#cbd5e0', color: 'rgb(55 65 81)', gridcolor: 'rgba(203, 213, 225, 0.5)' },
                                        margin: { l: 50, r: 20, t: 30, b: 50 },
                                        hovermode: 'closest',
                                        showlegend: false,
                                        paper_bgcolor: 'rgba(0,0,0,0)',
                                        plot_bgcolor: 'rgba(0,0,0,0)',
                                        font: { color: 'rgb(55 65 81)' }
                                    }}
                                    useResizeHandler={true}
                                    className="w-full h-[450px]"
                                    config={{ responsive: true, displayModeBar: false }}
                                />
                            </div>
                        )}
                    </div>

                    {/* --- Sekce pro AI Interpretaci --- */}
                    {result && ( // Zobrazíme jen pokud máme výsledek
                        <div className="pt-6 border-t border-dashed border-gray-300 dark:border-gray-600">
                            <h4 className="text-base font-semibold mb-3 text-gray-800 dark:text-gray-100">Interpretace pomocí AI</h4>
                            {/* Tlačítko pro spuštění interpretace */}
                            {!aiInterpretation && !isInterpreting && !aiError && (
                                <button
                                    onClick={handleInterpret}
                                    disabled={isInterpreting}
                                    className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out"
                                >
                                    💡 Interpretovat výsledek pomocí AI
                                </button>
                            )}
                            {/* Indikátor načítání */}
                            {isInterpreting && (
                                <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 p-2 rounded bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600">
                                    <svg className="animate-spin h-4 w-4 text-indigo-600 dark:text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                                    AI generuje interpretaci...
                                </div>
                            )}
                            {/* Zobrazení chyby AI */}
                            {aiError && (
                                <div role="alert" className="mt-3 p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                                    <p>⚠️ Chyba interpretace: {aiError}</p>
                                    <button onClick={handleInterpret} className="mt-1 text-xs font-medium text-red-800 dark:text-red-300 underline hover:text-red-900 dark:hover:text-red-200">
                                        Zkusit znovu
                                    </button>
                                </div>
                            )}
                            {/* Zobrazení úspěšné interpretace */}
                            {aiInterpretation && !isInterpreting && (
                                <div className="mt-3 p-4 bg-gray-100 dark:bg-gray-700/60 rounded border border-gray-200 dark:border-gray-600">
                                    <p className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">{aiInterpretation}</p>
                                    <button
                                        onClick={() => { setAiInterpretation(null); setAiError(null); }}
                                        className="mt-3 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline"
                                    >
                                        Skrýt / Generovat novou
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
            {/* --- Konec sekce Výsledků --- */}
        </div>
    );
}