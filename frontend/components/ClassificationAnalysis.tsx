import React, { useEffect, useState, useCallback } from 'react'; // Přidán useCallback
import Plot from 'react-plotly.js'; // Pro případný graf feature importances

// Rozhraní pro typ sloupce
interface ColumnType {
    name: string;
    type: string;
}

// Rozhraní pro výsledek klasifikace (z původního kódu)
interface ClassificationResult {
    algorithm_used: string;
    reason: string;
    standardized: boolean | null;
    test_size: number;
    knn_neighbors: number | null;
    feature_columns_used: string[];
    target_column: string;
    metrics: {
        accuracy: number;
        precision_weighted: number;
        recall_weighted: number;
        f1_weighted: number;
    };
    classification_report: {
        [className: string]: {
            precision: number;
            recall: number;
            'f1-score': number;
            support: number;
        } | number | {
            precision?: number;
            recall?: number;
            'f1-score'?: number;
            support?: number;
        };
    };
    confusion_matrix: number[][];
    confusion_matrix_labels: string[];
    feature_importances: { feature: string; importance: number }[] | null;
}

// Pomocná funkce pro formátování (z původního kódu)
const formatNumber = (num: number | undefined | null, decimals = 2): string | number => {
    if (typeof num !== 'number' || isNaN(num)) {
        return '-';
    }
    return num % 1 === 0 ? num : num.toFixed(decimals);
};


export default function ClassificationAnalysis() {
    // --- Stavy z původního kódu ---
    const [allColumns, setAllColumns] = useState<ColumnType[]>([]);
    const [featureColumns, setFeatureColumns] = useState<string[]>([]);
    const [targetColumn, setTargetColumn] = useState<string | null>(null);
    const [algorithm, setAlgorithm] = useState('auto');
    const [standardize, setStandardize] = useState(true);
    const [testSize, setTestSize] = useState(0.25);
    const [knnNeighbors, setKnnNeighbors] = useState<number>(5);
    const [result, setResult] = useState<ClassificationResult | null>(null);
    const [loading, setLoading] = useState(false); // Zahrnuje načítání sloupců i analýzy
    const [error, setError] = useState<string | null>(null);

    // --- Nové stavy POUZE pro AI interpretaci ---
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    const [isInterpreting, setIsInterpreting] = useState<boolean>(false);
    const [aiError, setAiError] = useState<string | null>(null);
    // --- Konec nových stavů ---

    // --- Načtení sloupců (z původního kódu, přidán reset AI) ---
    useEffect(() => {
        setLoading(true);
        setError(null);
        setResult(null);
        setAiInterpretation(null); // Reset AI
        setAiError(null);

        fetch("http://localhost:8000/api/get_column_types")
            .then(res => {
                if (!res.ok) throw new Error('Nepodařilo se načíst sloupce');
                return res.json();
            })
            .then((data: ColumnType[]) => {
                setAllColumns(data);
                setError(null);
            })
            .catch(err => {
                console.error("Fetch columns error:", err);
                setError(err.message);
                setAllColumns([]);
            })
            .finally(() => setLoading(false));
    }, []);

    // --- Nová funkce POUZE pro reset výsledků a AI ---
    const resetResultsAndInterpretation = useCallback(() => {
        setResult(null);
        // Error nemažeme zde, aby zůstal viditelný, pokud vznikl
        setAiInterpretation(null);
        setAiError(null);
    }, []);

    // --- Handlery změn vstupů (přidán reset) ---
    const handleFeatureChange = useCallback((colName: string, isChecked: boolean) => {
        setFeatureColumns(prev =>
            isChecked ? [...prev, colName] : prev.filter(c => c !== colName)
        );
        resetResultsAndInterpretation();
    }, [resetResultsAndInterpretation]);

    const handleTargetChange = useCallback((selectedTarget: string | null) => {
        setTargetColumn(selectedTarget);
        if (selectedTarget && featureColumns.includes(selectedTarget)) {
            setFeatureColumns(prev => prev.filter(c => c !== selectedTarget));
        }
        resetResultsAndInterpretation();
    }, [featureColumns, resetResultsAndInterpretation]);

    const handleAlgorithmChange = useCallback((newAlgo: string) => {
        setAlgorithm(newAlgo);
        resetResultsAndInterpretation();
    }, [resetResultsAndInterpretation]);

    const handleStandardizeChange = useCallback((isChecked: boolean) => {
        setStandardize(isChecked);
        resetResultsAndInterpretation();
    }, [resetResultsAndInterpretation]);

    const handleTestSizeChange = useCallback((newSize: number) => {
        // Jednoduché omezení pro jistotu
        const clampedSize = Math.max(0.1, Math.min(0.5, newSize));
        setTestSize(clampedSize);
        resetResultsAndInterpretation();
    }, [resetResultsAndInterpretation]);

    const handleKnnNeighborsChange = useCallback((newK: number) => {
        setKnnNeighbors(Math.max(1, newK));
        resetResultsAndInterpretation();
    }, [resetResultsAndInterpretation]);

    // --- Handler pro spuštění analýzy (z původního kódu, s opraveným fetch a resetem) ---
    const handleRun = useCallback(async () => {
        if (!targetColumn || featureColumns.length === 0) {
            setError("Musíte vybrat cílovou proměnnou a alespoň jeden příznak.");
            setResult(null); // Vyčistit staré výsledky při validaci
            setAiInterpretation(null);
            setAiError(null);
            return;
        }
        if (featureColumns.includes(targetColumn)) {
            setError("Cílová proměnná nemůže být zároveň příznakem.");
            return;
        }

        resetResultsAndInterpretation(); // Reset AI stavu
        setLoading(true);
        setError(null); // Vyčistit starou chybu před novým pokusem

        try {
            const requestBody = {
                feature_columns: featureColumns,
                target_column: targetColumn,
                algorithm: algorithm,
                standardize: standardize,
                test_size: testSize,
                knn_neighbors: algorithm === 'knn' ? knnNeighbors : undefined,
            };

            const response = await fetch("http://localhost:8000/api/classification_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody)
            });

            const responseText = await response.text();
            let parsedData: any;

            try {
                parsedData = JSON.parse(responseText);
            } catch (e) {
                console.error("Failed to parse JSON response:", responseText);
                throw new Error(`Chyba serveru (${response.status}): Odpověď není platný JSON. Začátek: ${responseText.substring(0, 150)}...`);
            }

            if (!response.ok) {
                const errorMessage = parsedData?.detail || `Chyba ${response.status}: ${response.statusText}`;
                console.error("Backend error:", errorMessage, "Raw response:", parsedData);
                throw new Error(errorMessage);
            }

            const data: ClassificationResult = parsedData as ClassificationResult;
            setResult(data);
            // setError(null); // Chyba se čistí na začátku

        } catch (err: any) {
            console.error("Classification fetch error:", err);
            setError(err instanceof Error ? err.message : "Nastala neočekávaná chyba při klasifikaci.");
            setResult(null);
        } finally {
            setLoading(false);
        }
    }, [targetColumn, featureColumns, algorithm, standardize, testSize, knnNeighbors, resetResultsAndInterpretation]);


    // --- Nová funkce POUZE pro volání AI interpretace ---
    const handleInterpret = useCallback(async () => {
        if (!result) return;

        setIsInterpreting(true);
        setAiInterpretation(null);
        setAiError(null);

        const interpretationPayload = {
            analysis_type: "classification",
            algorithm_used: result.algorithm_used,
            target_variable: result.target_column,
            features_used: result.feature_columns_used,
            metrics: {
                accuracy: result.metrics.accuracy,
                precision: result.metrics.precision_weighted,
                recall: result.metrics.recall_weighted,
                f1_score: result.metrics.f1_weighted,
            },
            has_feature_importances: !!result.feature_importances && result.feature_importances.length > 0,
            number_of_classes: result.confusion_matrix_labels.length,
        };

        try {
            const response = await fetch("http://localhost:8000/api/interpret_classification", {
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
    }, [result]);
    // --- Konec nové funkce ---


    // Filtrované seznamy pro výběry (z původního kódu)
    const potentialFeatures = allColumns;
    const potentialTargets = allColumns.filter(col => col.type === 'Kategorie' || col.type === 'Číselný');

    // --- Renderovací funkce (z původního kódu, jen s úpravou tabulek) ---

    const renderFeatureSelector = () => (
        <div>
            <label className="block font-medium mb-2 text-gray-700 dark:text-gray-300">1. Vyberte příznaky (Features):</label>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 p-3 border rounded bg-gray-50 dark:bg-gray-800 dark:border-gray-700 max-h-60 overflow-y-auto">
                {potentialFeatures.map((col) => (
                    <label key={col.name} className={`flex items-center gap-2 text-sm p-1 rounded cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 ${targetColumn === col.name ? 'opacity-50 cursor-not-allowed' : ''}`}>
                        <input
                            type="checkbox"
                            checked={featureColumns.includes(col.name)}
                            disabled={targetColumn === col.name || loading || isInterpreting}
                            onChange={(e) => handleFeatureChange(col.name, e.target.checked)} // Použít useCallback verzi
                            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-blue-600 dark:ring-offset-gray-800 disabled:opacity-50"
                        />
                        <span className={`text-gray-800 dark:text-gray-200 ${targetColumn === col.name ? 'line-through' : ''}`}>{col.name} ({col.type})</span>
                    </label>
                ))}
            </div>
            {featureColumns.length === 0 && <p className="text-xs text-red-600 mt-1">Vyberte alespoň jeden příznak.</p>}
        </div>
    );

    const renderTargetSelector = () => (
        <div>
            <label htmlFor="target-select" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">2. Vyberte cílovou proměnnou (Target):</label>
            <select
                id="target-select"
                value={targetColumn ?? ""}
                onChange={e => handleTargetChange(e.target.value || null)}
                disabled={loading || isInterpreting}
                className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
            >
                <option value="">-- Vyberte cíl --</option>
                {potentialTargets.map(col => (
                    <option key={col.name} value={col.name}>
                        {col.name} ({col.type})
                    </option>
                ))}
            </select>
            {!targetColumn && <p className="text-xs text-red-600 mt-1">Musíte vybrat cílovou proměnnou.</p>}
        </div>
    );

    const renderParameterSelectors = () => (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
                <label htmlFor="algo-select" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">3. Klasifikační algoritmus:</label>
                <select
                    id="algo-select"
                    value={algorithm}
                    onChange={e => handleAlgorithmChange(e.target.value)}
                    disabled={loading || isInterpreting}
                    className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
                >
                    <option value="auto">Automaticky</option>
                    <option value="logistic_regression">Logistická Regrese</option>
                    <option value="knn">K-Nearest Neighbors (KNN)</option>
                    <option value="decision_tree">Rozhodovací Strom</option>
                    <option value="random_forest">Náhodný Les</option>
                    <option value="naive_bayes">Naive Bayes (Gaussian)</option>
                </select>
            </div>

            {algorithm === 'knn' && (
                <div>
                    <label htmlFor="knn-k" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">Počet sousedů (K pro KNN):</label>
                    <input
                        id="knn-k"
                        type="number"
                        min="1"
                        step="1"
                        value={knnNeighbors}
                        onChange={e => handleKnnNeighborsChange(parseInt(e.target.value, 10))}
                        disabled={loading || isInterpreting}
                        className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
                    />
                </div>
            )}

            <div>
                <label htmlFor="test-size" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">Velikost testovací sady:</label>
                <input
                    id="test-size"
                    type="number"
                    min="0.1"
                    max="0.5"
                    step="0.05"
                    value={testSize}
                    onChange={e => handleTestSizeChange(parseFloat(e.target.value))}
                    disabled={loading || isInterpreting}
                    className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Podíl dat pro testování (0.1 = 10%).</p>
            </div>

            <div className="flex items-center pt-5">
                <label className="flex items-center gap-2 cursor-pointer">
                    <input
                        type="checkbox"
                        checked={standardize}
                        onChange={(e) => handleStandardizeChange(e.target.checked)}
                        disabled={loading || isInterpreting}
                        className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-blue-600 dark:ring-offset-gray-800 disabled:opacity-50"
                    />
                    <span className="text-sm text-gray-800 dark:text-gray-200">Standardizovat numerické příznaky</span>
                </label>
            </div>
        </div>
    );

    // Zobrazení výsledků
    const renderResults = () => {
        // Tato funkce se volá jen pokud result existuje a NENÍ loading
        if (!result) return null;

        const renderConfusionMatrix = () => (
            <div className="mt-4">
                <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-200">Confusion Matrix (Matice záměn)</h4>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Řádky: Skutečná třída, Sloupce: Predikovaná třída</p>
                {/* --- Obalující DIV s omezením výšky a scrollem --- */}
                <div className="overflow-x-auto max-h-[300px] overflow-y-auto border border-gray-300 dark:border-gray-600 rounded-md shadow-sm">
                    <table className="min-w-full border-collapse bg-white dark:bg-gray-800 text-sm">
                        <thead className="bg-gray-100 dark:bg-gray-700">
                        <tr>
                            <th className="border p-2 dark:border-gray-600 text-gray-600 dark:text-gray-300 text-left whitespace-nowrap">Skutečná \ Predikovaná</th>
                            {result.confusion_matrix_labels.map(label => (
                                <th key={label} className="border p-2 dark:border-gray-600 font-medium text-gray-600 dark:text-gray-300 whitespace-nowrap">{label}</th>
                            ))}
                        </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {result.confusion_matrix.map((row, i) => (
                            <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                <td className="border p-2 font-medium dark:border-gray-600 text-gray-600 dark:text-gray-300 whitespace-nowrap">{result.confusion_matrix_labels[i]}</td>
                                {row.map((cell, j) => (
                                    <td key={j} className={`border p-2 text-center dark:border-gray-600 ${i === j ? 'bg-green-50 dark:bg-green-800/30 font-semibold' : ''} text-gray-700 dark:text-gray-300`}>
                                        {cell}
                                    </td>
                                ))}
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </div>
            </div>
        );

        const renderClassificationReport = () => {
            const reportKeys = result.confusion_matrix_labels.concat(['accuracy', 'macro avg', 'weighted avg']);
            const reportData = Object.entries(result.classification_report)
                .filter(([key]) => reportKeys.includes(key))
                .sort((a, b) => {
                    const keyA = a[0];
                    const keyB = b[0];

                    // Pomocná funkce pro získání řadící hodnoty
                    const getSortOrder = (key: string): number => {
                        // Priorita pro specifické metriky
                        if (key === 'accuracy') return 1;
                        if (key === 'macro avg') return 100;
                        if (key === 'weighted avg') return 101;

                        // Pokud to není specifická metrika, je to název třídy
                        // Najdeme její index v původním poli labelů
                        const index = result.confusion_matrix_labels.indexOf(key);

                        // Vrátíme index + 2, aby třídy byly za 'accuracy'
                        // Pokud by se náhodou nenašla (což by nemělo nastat), dáme ji na konec
                        return index !== -1 ? index + 2 : 999;
                    };

                    const orderA = getSortOrder(keyA);
                    const orderB = getSortOrder(keyB);

                    return orderA - orderB;
                });
            return (
                <div className="mt-4">
                    <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-200">Classification Report</h4>
                    {/* --- Obalující DIV s omezením výšky a scrollem --- */}
                    <div className="overflow-x-auto max-h-[300px] overflow-y-auto border border-gray-300 dark:border-gray-600 rounded-md shadow-sm">
                        <table className="min-w-full border-collapse bg-white dark:bg-gray-800 text-sm">
                            <thead className="bg-gray-100 dark:bg-gray-700">
                            <tr>
                                <th className="border p-2 dark:border-gray-600 text-gray-600 dark:text-gray-300 text-left whitespace-nowrap">Třída / Metrika</th>
                                <th className="border p-2 dark:border-gray-600 text-gray-600 dark:text-gray-300 text-center">Precision</th>
                                <th className="border p-2 dark:border-gray-600 text-gray-600 dark:text-gray-300 text-center">Recall</th>
                                <th className="border p-2 dark:border-gray-600 text-gray-600 dark:text-gray-300 text-center">F1-Score</th>
                                <th className="border p-2 dark:border-gray-600 text-gray-600 dark:text-gray-300 text-center">Support</th>
                            </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                            {reportData.map(([key, values]) => {
                                if (typeof values === 'number') { // Accuracy
                                    return ( /* ... řádek pro accuracy ... */
                                        <tr key={key} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                            <td className="border p-2 font-medium dark:border-gray-600 text-gray-600 dark:text-gray-300 whitespace-nowrap">Accuracy</td>
                                            <td colSpan={3} className="border p-2 text-center dark:border-gray-600 text-gray-700 dark:text-gray-300">{formatNumber(values, 4)}</td>
                                            <td className="border p-2 text-center dark:border-gray-600 text-gray-500 dark:text-gray-400"></td>
                                        </tr>
                                    );
                                } else { // Ostatní řádky
                                    const isAvg = key.includes('avg');
                                    return ( /* ... řádek pro třídu/průměr s formátováním ... */
                                        <tr key={key} className={`hover:bg-gray-50 dark:hover:bg-gray-700/50 ${isAvg ? 'bg-gray-50 dark:bg-gray-700/60 font-medium' : ''}`}>
                                            <td className="border p-2 dark:border-gray-600 text-gray-600 dark:text-gray-300 whitespace-nowrap">{key}</td>
                                            <td className="border p-2 text-center dark:border-gray-600 text-gray-700 dark:text-gray-300">{formatNumber(values.precision, 3)}</td>
                                            <td className="border p-2 text-center dark:border-gray-600 text-gray-700 dark:text-gray-300">{formatNumber(values.recall, 3)}</td>
                                            <td className="border p-2 text-center dark:border-gray-600 text-gray-700 dark:text-gray-300">{formatNumber(values['f1-score'], 3)}</td>
                                            <td className="border p-2 text-center dark:border-gray-600 text-gray-500 dark:text-gray-400">{values.support ?? '-'}</td>
                                        </tr>
                                    );
                                }
                            })}
                            </tbody>
                        </table>
                    </div>
                </div>
            );
        };

        const renderFeatureImportances = () => {
            if (!result.feature_importances || result.feature_importances.length === 0) return null;
            const topN = 15;
            const importancesToShow = result.feature_importances.slice(0, topN);
            return (
                <div className="mt-6">
                    <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-200">Důležitost příznaků (Feature Importances)</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Zobrazeno top {Math.min(topN, result.feature_importances.length)} nejdůležitějších příznaků.</p>
                    <div className="bg-white dark:bg-gray-800 p-2 rounded border dark:border-gray-700 shadow-sm max-h-[400px] overflow-y-auto">
                        <Plot
                            data={[{
                                y: importancesToShow.map(imp => imp.feature).reverse(),
                                x: importancesToShow.map(imp => imp.importance).reverse(),
                                type: 'bar',
                                orientation: 'h',
                                marker: { color: '#3b82f6' }
                            }]}
                            layout={{
                                autosize: true,
                                xaxis: { title: 'Importance', automargin: true, gridcolor: '#e5e7eb', gridwidth: 1, zerolinecolor: '#d1d5db', zerolinewidth: 1 },
                                yaxis: { title: 'Feature', automargin: true, type: 'category', gridcolor: '#e5e7eb', gridwidth: 1 },
                                margin: { l: 150, r: 20, t: 30, b: 50 },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                font: { color: '#374151' } // Base font color
                            }}
                            useResizeHandler={true}
                            className="w-full"
                            config={{responsive: true, displayModeBar: false}}
                        />
                    </div>
                    {/* Styly pro dark mode grafu */}
                    <style jsx global>{`
                        /* ... CSS pro dark mode grafu (stejné jako v předchozí verzi) ... */
                        .dark .js-plotly-plot .plotly .gridlayer .grid path { stroke: #4b5563 !important; }
                        .dark .js-plotly-plot .plotly .zerolinelayer .zeroline { stroke: #6b7280 !important; }
                        .dark .js-plotly-plot .plotly text { fill: #d1d5db !important; }
                     `}</style>
                </div>
            );
        };

        // --- Hlavní render bloku výsledků ---
        return (
            <div className="space-y-6 mt-6 border-t pt-6 dark:border-gray-700">
                {/* Nadpis výsledků */}
                <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Výsledky Klasifikace</h3>

                {/* Info o konfiguraci */}
                <div className="bg-gray-50 dark:bg-gray-800/60 p-4 rounded border dark:border-gray-700 shadow-sm text-sm">
                    {/* ... kód pro zobrazení konfigurace ... */}
                    <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-200">Konfigurace Analýzy</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1 text-gray-600 dark:text-gray-300">
                        <p><strong className="font-medium text-gray-700 dark:text-gray-400">Algoritmus:</strong> {result.algorithm_used} {result.reason && <span className='text-xs'>({result.reason})</span>}</p>
                        <p><strong className="font-medium text-gray-700 dark:text-gray-400">Cílová proměnná:</strong> {result.target_column}</p>
                        <p><strong className="font-medium text-gray-700 dark:text-gray-400">Standardizace:</strong> {result.standardized === null ? 'N/A' : (result.standardized ? 'Ano' : 'Ne')}</p>
                        <p><strong className="font-medium text-gray-700 dark:text-gray-400">Testovací sada:</strong> {formatNumber(result.test_size * 100, 0)}%</p>
                        {result.knn_neighbors && <p><strong className="font-medium text-gray-700 dark:text-gray-400">KNN Sousedé:</strong> {result.knn_neighbors}</p>}
                        <p className="col-span-full"><strong className="font-medium text-gray-700 dark:text-gray-400">Použité příznaky:</strong> {result.feature_columns_used.join(', ')}</p>
                    </div>
                </div>

                {/* Souhrnné metriky */}
                <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 p-4 rounded shadow-sm">
                    {/* ... kód pro zobrazení metrik ... */}
                    <h4 className="text-md font-medium mb-2 text-blue-800 dark:text-blue-200">Souhrnné Metriky (Weighted Avg)</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                        <div><p className="text-xs text-blue-600 dark:text-blue-300">Accuracy</p><p className="text-xl font-semibold text-blue-900 dark:text-blue-100">{formatNumber(result.metrics.accuracy, 4)}</p></div>
                        <div><p className="text-xs text-blue-600 dark:text-blue-300">Precision</p><p className="text-xl font-semibold text-blue-900 dark:text-blue-100">{formatNumber(result.metrics.precision_weighted, 4)}</p></div>
                        <div><p className="text-xs text-blue-600 dark:text-blue-300">Recall</p><p className="text-xl font-semibold text-blue-900 dark:text-blue-100">{formatNumber(result.metrics.recall_weighted, 4)}</p></div>
                        <div><p className="text-xs text-blue-600 dark:text-blue-300">F1-Score</p><p className="text-xl font-semibold text-blue-900 dark:text-blue-100">{formatNumber(result.metrics.f1_weighted, 4)}</p></div>
                    </div>
                </div>

                {/* Detailní report a matice */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    {renderClassificationReport()}
                    {renderConfusionMatrix()}
                </div>

                {/* Feature Importances */}
                {renderFeatureImportances()}

                {/* --- Sekce pro AI Interpretaci --- */}
                <div className="pt-6 border-t border-dashed border-gray-300 dark:border-gray-600">
                    <h4 className="text-base font-semibold mb-3 text-gray-800 dark:text-gray-100">Interpretace pomocí AI</h4>

                    {/* Tlačítko / Načítání / Chyba / Výsledek */}
                    {!aiInterpretation && !isInterpreting && !aiError && (
                        <button
                            onClick={handleInterpret}
                            disabled={isInterpreting}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out"
                        >
                            💡 Interpretovat výsledky pomocí AI
                        </button>
                    )}
                    {isInterpreting && (
                        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 p-2 rounded bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600">
                            <svg className="animate-spin h-4 w-4 text-indigo-600 dark:text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                            AI generuje interpretaci...
                        </div>
                    )}
                    {aiError && (
                        <div role="alert" className="mt-3 p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                            ⚠️ Chyba interpretace: {aiError}
                            <button onClick={handleInterpret} className="ml-4 text-xs font-medium text-red-800 dark:text-red-300 underline hover:text-red-900 dark:hover:text-red-200">
                                Zkusit znovu
                            </button>
                        </div>
                    )}
                    {aiInterpretation && !isInterpreting && (
                        <div className="mt-3 p-4 bg-gray-100 dark:bg-gray-700/60 rounded border border-gray-200 dark:border-gray-600">
                            <p className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">{aiInterpretation}</p>
                            <button onClick={() => { setAiInterpretation(null); setAiError(null); }} className="mt-3 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline">
                                Skrýt interpretaci / Generovat novou
                            </button>
                        </div>
                    )}
                </div>
                {/* --- Konec sekce AI Interpretace --- */}

            </div>
        );
    }; // Konec renderResults


    // --- Hlavní Return Komponenty ---
    return (
        <div className="space-y-6 p-4 md:p-6 lg:p-8 dark:bg-gray-900 dark:text-gray-100 min-h-screen">
            <h1 className="text-2xl font-bold mb-6 text-gray-800 dark:text-gray-100">Klasifikační Analýza</h1>

            {/* Zobrazení načítání sloupců */}
            {loading && allColumns.length === 0 && (
                <div className="text-center p-4 text-gray-500">Načítám dostupné sloupce...</div>
            )}

            {/* Formulář se vstupy (zobrazí se po načtení sloupců) */}
            {!loading && allColumns.length > 0 && (
                <fieldset disabled={loading || isInterpreting} className="space-y-6">
                    {renderFeatureSelector()}
                    {renderTargetSelector()}
                    {renderParameterSelectors()}
                </fieldset>
            )}
            {/* Zpráva pokud se nepodařilo načíst sloupce */}
            {!loading && allColumns.length === 0 && error && (
                <div role="alert" className="mt-4 p-3 bg-yellow-100 border border-yellow-400 text-yellow-700 rounded dark:bg-yellow-900/30 dark:border-yellow-700 dark:text-yellow-200">
                    Nepodařilo se načíst sloupce. Zkuste obnovit stránku nebo zkontrolujte konzoli pro více detailů.
                </div>
            )}

            {/* Tlačítko Spustit (zobrazí se po načtení sloupců) */}
            {allColumns.length > 0 && (
                <button
                    onClick={handleRun}
                    disabled={!targetColumn || featureColumns.length === 0 || loading || isInterpreting || featureColumns.includes(targetColumn ?? '')}
                    className="bg-green-600 hover:bg-green-700 text-white px-5 py-2 rounded-md shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out mt-6"
                >
                    {loading && !isInterpreting ? (
                        <div className="flex items-center justify-center">
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                            Analyzuji...
                        </div>
                    ) : "Spustit Klasifikaci"}
                </button>
            )}

            {/* Zobrazení chyby analýzy */}
            {error && !loading && ( // Zobrazit chybu jen pokud neběží loading
                <div role="alert" className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded dark:bg-red-900/30 dark:border-red-700 dark:text-red-200">
                    ⚠️ Chyba: {error}
                </div>
            )}

            {/* Zobrazení výsledků (volá renderResults, který obsahuje i AI sekci) */}
            {!loading && result && renderResults()}

        </div>
    );
}