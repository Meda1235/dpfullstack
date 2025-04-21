import React, { useEffect, useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
// --- ADDED INTERFACE DEFINITIONS --- (Beze změny)
interface ColumnType { name: string; type: string; }
interface CorrelationPairResult { var1: string; var2: string; correlation: number; pValue: number; rSquared?: number; ciLow?: number; ciHigh?: number; strength: string; significant: boolean; }
interface ScatterData { x: number[]; y: number[]; xLabel: string; yLabel: string; }
interface CorrelationResult { matrix: number[][]; columns: string[]; pValues: number[][]; method: string; reason: string; results: CorrelationPairResult[]; scatterData?: ScatterData; }
// --- END OF ADDED INTERFACE DEFINITIONS ---

// --- Pomocné funkce (Beze změny) ---
const formatPValue = (pValue: number | undefined | null): string => {
    if (typeof pValue !== 'number' || isNaN(pValue)) return '-'; // Vrací string '-'
    if (pValue < 0.001) return pValue.toExponential(2); // Vrací string (exponenciální notace)
    return pValue.toFixed(3); // Vrací string (zaokrouhlené číslo)
};

const formatCorr = (corr: number | undefined | null): string => {
    if (typeof corr !== 'number' || isNaN(corr)) return '-'; // Vrací string '-'
    return corr.toFixed(3); // Vrací string (zaokrouhlené číslo)
}


export default function CorrelationAnalysis() {
    // --- Stávající stavy ---
    const [allColumns, setAllColumns] = useState<ColumnType[]>([]);
    const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
    const [method, setMethod] = useState<string>('auto');
    const [results, setResults] = useState<CorrelationResult | null>(null);
    const [isLoadingColumns, setIsLoadingColumns] = useState<boolean>(true);
    const [isLoadingAnalysis, setIsLoadingAnalysis] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    // --- Nové stavy pro AI interpretaci ---
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    const [isInterpreting, setIsInterpreting] = useState<boolean>(false);
    const [aiError, setAiError] = useState<string | null>(null);
    // --- Konec nových stavů ---


    // --- Fetch sloupců (Beze změny, jen přidáme reset AI) ---
    const fetchColumns = useCallback(async () => {
        setIsLoadingColumns(true);
        setError(null);
        setResults(null);
        setAiInterpretation(null); // Reset AI
        setAiError(null);        // Reset AI chyby
        try {
            // ... (zbytek funkce fetchColumns beze změny)
            const res = await fetch("http://localhost:8000/api/get_column_types");
            if (!res.ok) throw new Error(`Nepodařilo se načíst sloupce: ${res.statusText}`);
            const data: ColumnType[] = await res.json();
            const numericCols = data.filter(col => col.type === "Číselný");
            setAllColumns(numericCols);
        } catch (err: any) {
            console.error("Column Fetch Error:", err);
            setError(`Chyba načítání sloupců: ${err.message}`);
            setAllColumns([]);
        } finally {
            setIsLoadingColumns(false);
        }
    }, []);

    useEffect(() => {
        fetchColumns();
    }, [fetchColumns]);

    // --- Funkce pro resetování výsledků a interpretace ---
    const resetResultsAndInterpretation = useCallback(() => {
        setResults(null);
        setError(null); // Reset chyby analýzy
        setAiInterpretation(null);
        setAiError(null);
        // Nereseujeme isLoadingAnalysis zde
    }, []);


    // --- Handle změny checkboxů ---
    const handleCheckboxChange = (columnName: string, isChecked: boolean) => {
        setSelectedColumns(prev => {
            const newSelection = isChecked
                ? [...prev, columnName]
                : prev.filter(c => c !== columnName);
            // Reset při změně výběru
            resetResultsAndInterpretation();
            return newSelection;
        });
    };

    // --- Handle změny metody ---
    const handleMethodChange = (newMethod: string) => {
        setMethod(newMethod);
        resetResultsAndInterpretation(); // Reset při změně metody
    }

    // --- Spuštění analýzy (přidán reset AI) ---
    const runAnalysis = useCallback(async () => {
        if (selectedColumns.length < 2) {
            setError("Vyberte prosím alespoň dvě číselné proměnné.");
            setResults(null);
            setAiInterpretation(null);
            setAiError(null);
            return;
        }

        // Reset před spuštěním
        resetResultsAndInterpretation();
        setIsLoadingAnalysis(true); // Načítání až po resetu

        try {
            const response = await fetch("http://localhost:8000/api/correlation_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    columns: selectedColumns,
                    method: method,
                }),
            });
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `Chyba ${response.status}: ${response.statusText}`);
            }
            setResults(data as CorrelationResult);
            setError(null); // Vyčistit chybu, pokud ok
        } catch (err: any) {
            console.error("Correlation Analysis Error:", err);
            setError(`Chyba analýzy: ${err.message}`);
            setResults(null);
        } finally {
            setIsLoadingAnalysis(false);
        }
    }, [selectedColumns, method, resetResultsAndInterpretation]); // Přidána závislost

    // --- Nová funkce pro volání AI interpretace ---
    const handleInterpret = useCallback(async () => {
        if (!results) return;

        setIsInterpreting(true);
        setAiInterpretation(null);
        setAiError(null);

        // Připravíme data pro AI - posíláme jen klíčové informace
        const interpretationPayload = {
            analysis_type: "correlation",
            method: results.method,
            variables: results.columns,
            // Posíláme zjednodušený seznam párů
            correlation_pairs: results.results.map(pair => ({
                var1: pair.var1,
                var2: pair.var2,
                correlation: pair.correlation,
                pValue: pair.pValue,
                significant: pair.significant,
                strength: pair.strength // Síla může být pro AI užitečná
            })),
            // Můžeme přidat info, zda se zobrazuje matice nebo scatter
            visualization_type: results.columns.length > 2 ? "matrix" : "scatterplot"
        };

        try {
            const response = await fetch("http://localhost:8000/api/interpret_correlation", { // Cíl: Nový endpoint
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
                throw new Error("Nepodařilo se získat text interpretace z odpovědi AI.");
            }

        } catch (err: any) {
            console.error("AI Interpretation Error:", err);
            setAiError(`Chyba při získávání AI interpretace: ${err.message}`);
        } finally {
            setIsInterpreting(false);
        }
    }, [results]); // Závislost na výsledcích
    // --- Konec nové funkce ---

    const numericColumns = allColumns; // Už je filtrováno ve fetchColumns

    // --- JSX ---
    return (
        <div className="space-y-6 p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
            {/* Výběr sloupců (beze změny) */}
            <fieldset disabled={isLoadingAnalysis || isLoadingColumns} className="space-y-2">
                <legend className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2">1. Vyberte číselné proměnné (min. 2):</legend>
                {isLoadingColumns && <p className="text-sm text-gray-500 dark:text-gray-400">Načítám sloupce...</p>}
                {!isLoadingColumns && numericColumns.length === 0 && (
                    <p className="text-sm text-red-600 dark:text-red-400">Nebyly nalezeny žádné číselné sloupce.</p>
                )}
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-x-4 gap-y-2 max-h-48 overflow-y-auto p-2 border rounded dark:border-gray-600">
                    {numericColumns.map((col) => (
                        <label key={col.name} htmlFor={`corr-col-${col.name}`} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 p-1 rounded cursor-pointer">
                            <input
                                id={`corr-col-${col.name}`}
                                type="checkbox"
                                checked={selectedColumns.includes(col.name)}
                                onChange={(e) => handleCheckboxChange(col.name, e.target.checked)}
                                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800"
                                disabled={isLoadingAnalysis || isLoadingColumns} // Disable inputs during loading
                            />
                            <span>{col.name}</span>
                        </label>
                    ))}
                </div>
            </fieldset>

            {/* Výběr metody (Použijeme handleMethodChange) */}
            <div>
                <label htmlFor="corr-method" className="block text-sm font-medium text-gray-800 dark:text-gray-200 mb-1">2. Metoda korelace:</label>
                <select
                    id="corr-method"
                    value={method}
                    onChange={(e) => handleMethodChange(e.target.value)} // Použít novou funkci
                    className="mt-1 border border-gray-300 rounded px-3 py-2 text-sm w-full max-w-xs dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                    aria-label="Vyberte metodu korelace"
                    disabled={isLoadingAnalysis || isLoadingColumns} // Disable inputs during loading
                >
                    <option value="auto">Automaticky (doporučeno)</option>
                    <option value="pearson">Pearson (lineární vztah, normální data)</option>
                    <option value="spearman">Spearman (monotónní vztah, pořadová data)</option>
                    <option value="kendall">Kendall Tau (pořadová data, menší vzorky)</option>
                </select>
            </div>

            {/* Tlačítko Spustit (beze změny) */}
            <button
                onClick={runAnalysis}
                disabled={selectedColumns.length < 2 || isLoadingAnalysis || isLoadingColumns}
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out"
            >
                {/* ... SVG a text ... */}
                {isLoadingAnalysis ? (
                    <>
                        <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                        Analyzuji...
                    </>
                ) : "Spustit korelační analýzu"}
            </button>

            {/* Zobrazení chyby analýzy */}
            {error && (
                <div role="alert" className="p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                    ⚠️ Chyba analýzy: {error}
                </div>
            )}


            {/* Zobrazení výsledků */}
            {results && !isLoadingAnalysis && (
                <div className="space-y-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                    {/* Info o metodě (beze změny) */}
                    <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded border border-gray-200 dark:border-gray-600 text-sm">
                        <p><strong className="font-medium text-gray-800 dark:text-gray-200">Použitá metoda:</strong> <span className="font-semibold capitalize text-indigo-700 dark:text-indigo-400">{results.method}</span></p>
                        {results.reason && <p className="text-gray-600 dark:text-gray-400 mt-1">{results.reason}</p>}
                    </div>

                    {/* Tabulka výsledků (beze změny) */}
                    <div>
                        {/* ... kód tabulky ... */}
                        <h4 className="text-base font-semibold mb-2 text-gray-800 dark:text-gray-100">Detailní výsledky párů:</h4>
                        <div className="max-h-[500px] overflow-y-auto border border-gray-200 dark:border-gray-600 rounded-md shadow-sm">
                            <table className="min-w-full text-sm divide-y divide-gray-200 dark:divide-gray-700">
                                <thead className="bg-gray-100 dark:bg-gray-700 sticky top-0">
                                {/* ... hlavička tabulky ... */}
                                <tr>
                                    <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Proměnná A</th>
                                    <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Proměnná B</th>
                                    <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Korelace (r)</th>
                                    <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">p-hodnota</th>
                                    <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider" title="Koeficient determinace">R²</th>
                                    <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">95% CI</th>
                                    <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Síla</th>
                                    <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider" title="Statistická významnost (p < 0.05)">Význ.?</th>
                                </tr>
                                </thead>
                                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                                {/* ... tělo tabulky ... */}
                                {results.results.length === 0 && (
                                    <tr><td colSpan={8} className="text-center py-4 text-gray-500 dark:text-gray-400">Nebyly nalezeny žádné páry korelací.</td></tr>
                                )}
                                {results.results.map((r: CorrelationPairResult, idx: number) => (
                                    <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                        <td className="px-3 py-2 whitespace-nowrap text-gray-800 dark:text-gray-200">{r.var1}</td>
                                        <td className="px-3 py-2 whitespace-nowrap text-gray-800 dark:text-gray-200">{r.var2}</td>
                                        <td className="px-3 py-2 text-center font-mono">{formatCorr(r.correlation)}</td>
                                        <td className={`px-3 py-2 text-center font-mono ${!r.significant ? 'text-gray-500 dark:text-gray-400' : 'font-semibold text-green-700 dark:text-green-400'}`}>{formatPValue(r.pValue)}</td>
                                        <td className="px-3 py-2 text-center font-mono">{formatCorr(r.rSquared)}</td>
                                        <td className="px-3 py-2 text-center font-mono text-xs">
                                            {r.ciLow != null && r.ciHigh != null
                                                ? `[${formatCorr(r.ciLow)}, ${formatCorr(r.ciHigh)}]`
                                                : "–"}
                                        </td>
                                        <td className="px-3 py-2 text-center">{r.strength}</td>
                                        <td className="px-3 py-2 text-center">
                                            {r.significant
                                                ? <span title="Statisticky významné (p < 0.05)" className="text-green-600 dark:text-green-400">✔️</span>
                                                : <span title="Statisticky nevýznamné (p >= 0.05)" className="text-red-500 dark:text-red-400">✖️</span>}
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Vizualizace (beze změny) */}
                    <div className="pt-4">
                        {/* ... kód vizualizací ... */}
                        <h4 className="text-base font-semibold mb-3 text-gray-800 dark:text-gray-100">Vizualizace:</h4>
                        {results.columns.length === 2 && results.scatterData ? (
                            <div className="border rounded-lg p-2 bg-white dark:bg-gray-800 shadow-sm">
                                <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 px-1">Zobrazení vztahu mezi dvěma vybranými proměnnými.</p>
                                <Plot
                                    data={[
                                        { x: results.scatterData.x, y: results.scatterData.y, mode: 'markers', type: 'scatter', marker: { color: '#3b82f6', size: 5, opacity: 0.7 }, name: 'Data points', text: results.scatterData.x.map((_xVal: number, i: number) => `(${_xVal?.toFixed(2)}, ${results.scatterData?.y[i]?.toFixed(2)})`), hoverinfo: 'text' }
                                    ]}
                                    layout={{ autosize: true, title: `Korelace: ${results.scatterData.xLabel} vs ${results.scatterData.yLabel}`, xaxis: { title: results.scatterData.xLabel, zeroline: false }, yaxis: { title: results.scatterData.yLabel, zeroline: false }, margin: { l: 50, r: 20, t: 40, b: 50 }, hovermode: 'closest', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#374151' } }}
                                    useResizeHandler={true} className="w-full h-[500px]" config={{responsive: true, displayModeBar: false}}
                                />
                            </div>
                        ) : results.columns.length > 2 && results.pValues && results.matrix ? (
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* Correlation Matrix Heatmap */}
                                <div className="border rounded-lg p-2 bg-white dark:bg-gray-800 shadow-sm">
                                    <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 px-1">Tato heatmapa zobrazuje <strong>sílu a směr korelace</strong> (koeficient 'r') ...</p>
                                    <Plot
                                        data={[ { z: results.matrix, x: results.columns, y: results.columns, type: 'heatmap', colorscale: 'RdBu', zmin: -1, zmax: 1, zmid: 0, showscale: true, hoverongaps: false, xgap: 1, ygap: 1, colorbar: { title: 'Korelace (r)', titleside: 'right', len: 0.75 } } ]}
                                        layout={{ autosize: true, title: 'Korelační matice (r)', xaxis: { automargin: true, tickangle: -45 }, yaxis: { automargin: true }, margin: { l: 100, r: 20, t: 40, b: 100 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#374151' } }}
                                        useResizeHandler={true} className="w-full h-[500px]" config={{responsive: true, displayModeBar: false}}
                                    />
                                </div>
                                {/* P-Value Heatmap */}
                                <div className="border rounded-lg p-2 bg-white dark:bg-gray-800 shadow-sm">
                                    <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 px-1">Tato heatmapa zobrazuje <strong>statistickou významnost (p-hodnotu)</strong> ...</p>
                                    <Plot
                                        data={[ { z: results.pValues, x: results.columns, y: results.columns, type: 'heatmap', colorscale: [[0, 'rgb(0,100,0)'], [0.05, 'rgb(255,140,0)'], [1, 'rgb(220,220,220)']], zmin: 0, zmax: 0.1, showscale: true, hoverongaps: false, xgap: 1, ygap: 1, colorbar: { title: 'p-hodnota', titleside: 'right', len: 0.75, tickvals: [0, 0.01, 0.05, 0.1], ticktext: ['<0.01', '0.01', '0.05', '>0.1'] } } ]}
                                        layout={{ autosize: true, title: 'p-hodnoty (významnost)', xaxis: { automargin: true, tickangle: -45 }, yaxis: { automargin: true }, margin: { l: 100, r: 20, t: 40, b: 100 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#374151' } }}
                                        useResizeHandler={true} className="w-full h-[500px]" config={{responsive: true, displayModeBar: false}}
                                    />
                                </div>
                            </div>
                        ) : null }
                    </div>


                    {/* --- Nová Sekce pro AI Interpretaci --- */}
                    <div className="pt-6 border-t border-dashed border-gray-300 dark:border-gray-600">
                        <h4 className="text-base font-semibold mb-3 text-gray-800 dark:text-gray-100">Interpretace pomocí AI</h4>

                        {/* Tlačítko se zobrazí jen pokud NENÍ interpretace, NENÍ načítání a NENÍ chyba AI */}
                        {!aiInterpretation && !isInterpreting && !aiError && (
                            <button
                                onClick={handleInterpret}
                                disabled={isInterpreting}
                                className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out"
                            >
                                💡 Interpretovat výsledek pomocí AI
                            </button>
                        )}

                        {/* Zpráva o načítání interpretace */}
                        {isInterpreting && (
                            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 p-2 rounded bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600">
                                <svg className="animate-spin h-4 w-4 text-indigo-600 dark:text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                                AI generuje interpretaci, vyčkejte prosím...
                            </div>
                        )}

                        {/* Zobrazení chyby AI */}
                        {aiError && (
                            <div role="alert" className="mt-3 p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                                ⚠️ Chyba interpretace: {aiError}
                                <button
                                    onClick={handleInterpret}
                                    className="ml-4 text-xs font-medium text-red-800 dark:text-red-300 underline hover:text-red-900 dark:hover:text-red-200"
                                >
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
                                    Skrýt interpretaci / Generovat novou
                                </button>
                            </div>
                        )}
                    </div>
                    {/* --- Konec sekce AI Interpretace --- */}

                </div>
            )}
            {/* --- Konec Zobrazení výsledků --- */}
        </div>
    );
}