// ClusterAnalysis.js
import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
// Není potřeba importovat ClusterSummaryTable
// ... (rozhraní ColumnType, ClusterResult a funkce formatNumber zůstávají stejné) ...

interface ColumnType {
    name: string;
    type: string;
}

interface ClusterResult {
    method: string;
    reason: string;
    clusters: number | null;
    distance_metric: string;
    silhouette_score: number | null;
    summary: {
        [clusterId: number]: {
            [statKey: string]: number;
        }
    };
    labels: number[];
    columns: string[];
    pca_2d?: { x: number[]; y: number[]; labels: number[] };
    pca_components?: { x: number; y: number; name: string }[];
    pca_variance_table?: any[];
}

const formatNumber = (num: number | undefined | null): string | number => {
    if (typeof num !== 'number' || isNaN(num)) {
        return '-';
    }
    return num % 1 === 0 ? num : num.toFixed(2);
};


export default function ClusterAnalysis() {
    // --- Stávající state ---
    const [columns, setColumns] = useState<ColumnType[]>([]);
    const [selectedCols, setSelectedCols] = useState<string[]>([]);
    const [algorithm, setAlgorithm] = useState('auto');
    const [distance, setDistance] = useState('auto');
    const [numClusters, setNumClusters] = useState<number | null>(null);
    const [standardize, setStandardize] = useState(true);
    const [result, setResult] = useState<ClusterResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // --- Nový state pro AI interpretaci ---
    const [aiLoading, setAiLoading] = useState(false);
    const [aiError, setAiError] = useState<string | null>(null);
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    // ---------------------------------------

    useEffect(() => {
        fetch("http://localhost:8000/api/get_column_types")
            .then(res => res.json())
            .then((data: ColumnType[]) => {
                const numeric = data.filter(col => col.type === "Číselný");
                setColumns(numeric);
            });
    }, []);

    const handleRun = async () => {
        setLoading(true);
        setError(null);
        setResult(null);
        // --- Resetovat AI stav při nové analýze ---
        setAiInterpretation(null);
        setAiError(null);
        // -----------------------------------------

        try {
            // ... (zbytek kódu pro handleRun zůstává stejný) ...
            const requestBody: any = {
                columns: selectedCols,
                algorithm: algorithm,
                distance: distance,
                standardize: standardize,
            };
            if (numClusters !== null && algorithm !== 'auto' && algorithm !== 'dbscan') {
                requestBody.num_clusters = numClusters;
            }

            const response = await fetch("http://localhost:8000/api/cluster_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody)
            });

            const responseText = await response.text();
            if (!response.ok) {
                try {
                    const errorJson = JSON.parse(responseText);
                    throw new Error(errorJson.detail || `Chyba ${response.status}: ${response.statusText}`);
                } catch (e) {
                    // Pokud parsování selže, použijeme původní text
                    throw new Error(`Chyba ${response.status}: ${responseText.length < 200 ? responseText : response.statusText}`);
                }
            }

            const data: ClusterResult = JSON.parse(responseText);
            setResult(data);
        } catch (err: any) {
            console.error("Fetch error details:", err);
            setError(err.message || "Nastala neočekávaná chyba.");
        } finally {
            setLoading(false);
        }
    };

    // --- Nová funkce pro volání AI interpretace ---
    const handleAiInterpretation = async () => {
        if (!result) return; // Nemělo by nastat, ale pro jistotu

        setAiLoading(true);
        setAiError(null);
        setAiInterpretation(null);

        try {
            const requestBody = {
                analysis_type: "clustering", // Jak je definováno v Pydantic modelu
                algorithm_used: result.method,
                distance_metric: result.distance_metric,
                number_of_clusters_found: result.clusters, // Může být null
                silhouette_score: result.silhouette_score, // Může být null
                columns_used: result.columns,
                // standardization_used: standardize // Případně přidat, pokud by to backend potřeboval
            };

            const response = await fetch("http://localhost:8000/api/interpret_clustering", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody)
            });

            const responseText = await response.text(); // Vždy číst text pro lepší debug chyb
            if (!response.ok) {
                try {
                    const errorJson = JSON.parse(responseText);
                    throw new Error(errorJson.detail || `Chyba ${response.status}: ${response.statusText}`);
                } catch (e) {
                    // Pokud parsování selže, použijeme původní text
                    throw new Error(`Chyba ${response.status}: ${responseText.length < 200 ? responseText : response.statusText}`);
                }
            }

            const data = JSON.parse(responseText); // Parsovat až po kontrole ok
            if (data.interpretation) {
                setAiInterpretation(data.interpretation);
            } else {
                throw new Error("AI nevrátila platnou interpretaci.");
            }

        } catch (err: any) {
            console.error("AI Interpretation fetch error:", err);
            setAiError(err.message || "Nepodařilo se získat interpretaci od AI.");
        } finally {
            setAiLoading(false);
        }
    };
    // ---------------------------------------------

    // --- Logika pro generování tabulky (renderSummaryTable) zůstává stejná ---
    const renderSummaryTable = () => {
        // ... (kód funkce renderSummaryTable beze změny) ...
        if (!result || !result.summary || Object.keys(result.summary).length === 0 || !result.columns || result.columns.length === 0) {
            return null;
        }

        const summary = result.summary;
        const analyzedColumns = result.columns;
        const clusterIds = Object.keys(summary).map(Number).sort((a, b) => a - b);
        const stats = ['mean', 'std', 'min', 'max'];
        const statLabels: { [key: string]: string } = { 'mean': 'Průměr', 'std': 'Sm. odch.', 'min': 'Min', 'max': 'Max' };

        return (
            <div className="overflow-x-auto mt-6">
                <h3 className="text-lg font-medium mb-2 text-gray-700 dark:text-gray-200">Souhrnná tabulka shluků</h3>
                <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 shadow-sm rounded-lg">
                    <thead className="bg-gray-100 dark:bg-gray-700">
                    <tr>
                        <th rowSpan={2} className="sticky left-0 z-10 bg-gray-100 dark:bg-gray-700 border-b border-r border-gray-300 dark:border-gray-600 p-2 text-left font-medium text-sm text-gray-600 dark:text-gray-200">Shluk</th>
                        {analyzedColumns.map(col => (
                            <th key={col} colSpan={stats.length} className="border-b border-r border-gray-300 dark:border-gray-600 p-2 text-center font-medium text-sm text-gray-600 dark:text-gray-200">{col}</th>
                        ))}
                    </tr>
                    <tr>
                        {analyzedColumns.map(col => stats.map(stat => (
                            <th key={`${col}-${stat}`} className="border-b border-r border-gray-300 dark:border-gray-600 p-1 text-center font-normal text-xs bg-gray-50 dark:bg-gray-600 text-gray-500 dark:text-gray-300">{statLabels[stat]}</th>
                        )))}
                    </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {clusterIds.map(clusterId => (
                        <tr key={clusterId} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                            <td className="sticky left-0 z-10 bg-white dark:bg-gray-800 border-r border-gray-300 dark:border-gray-600 p-2 font-medium text-sm text-gray-900 dark:text-gray-300 whitespace-nowrap">
                                {clusterId === -1 ? 'Šum (DBSCAN)' : `Shluk ${clusterId}`}
                            </td>
                            {analyzedColumns.map(col => stats.map(stat => {
                                const key = `${col} (${stat})`;
                                const value = summary[clusterId]?.[key];
                                return (<td key={`${clusterId}-${col}-${stat}`} className="border-r border-gray-300 dark:border-gray-600 p-2 text-right text-sm text-gray-700 dark:text-gray-300">{formatNumber(value)}</td>);
                            }))}
                        </tr>
                    ))}
                    </tbody>
                </table>
            </div>
        );
    };


    return (
        <div className="space-y-6 p-4 md:p-6 lg:p-8 dark:bg-gray-900 dark:text-gray-100 min-h-screen">
            <h1 className="text-2xl font-bold mb-4 text-gray-800 dark:text-gray-100">Shluková analýza</h1>

            {/* --- Formulář pro nastavení analýzy (zůstává stejný) --- */}
            {/* Výběr proměnných */}
            <div>
                {/* ... kód pro výběr proměnných ... */}
                <label className="block font-medium mb-2 text-gray-700 dark:text-gray-300">1. Vyberte číselné proměnné (min. 2):</label>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 p-3 border rounded bg-gray-50 dark:bg-gray-800 dark:border-gray-700">
                    {columns.length === 0 && <p className="text-sm text-gray-500 dark:text-gray-400 col-span-full">Načítání sloupců...</p>}
                    {columns.map((col) => (
                        <label key={col.name} className="flex items-center gap-2 text-sm text-gray-800 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 p-1 rounded">
                            <input
                                type="checkbox"
                                checked={selectedCols.includes(col.name)}
                                onChange={(e) => {
                                    setSelectedCols(prev =>
                                        e.target.checked ? [...prev, col.name] : prev.filter(c => c !== col.name)
                                    );
                                }}
                                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-blue-600 dark:ring-offset-gray-800"
                            />
                            <span>{col.name}</span>
                        </label>
                    ))}
                </div>
                {selectedCols.length > 0 && selectedCols.length < 2 && (
                    <p className="text-xs text-red-600 mt-1">Vyberte alespoň dvě proměnné.</p>
                )}
            </div>
            {/* Nastavení parametrů */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* ... kód pro nastavení parametrů ... */}
                <div>
                    <label htmlFor="algorithm-select" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">2. Metoda shlukování:</label>
                    <select id="algorithm-select" value={algorithm} onChange={e => setAlgorithm(e.target.value)} className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500">
                        <option value="auto">Automaticky (doporučeno)</option>
                        <option value="kmeans">K-Means</option>
                        <option value="dbscan">DBSCAN</option>
                        <option value="hierarchical">Hierarchické</option>
                    </select>
                </div>
                <div>
                    <label htmlFor="distance-select" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">3. Vzdálenostní metrika:</label>
                    <select id="distance-select" value={distance} onChange={e => setDistance(e.target.value)} className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500">
                        <option value="auto">Automaticky (doporučeno)</option>
                        <option value="euclidean">Eukleidovská</option>
                        <option value="manhattan">Manhattanská</option>
                        <option value="cosine">Kosínová</option>
                    </select>
                </div>
                {(algorithm === 'kmeans' || algorithm === 'hierarchical') && (
                    <div>
                        <label htmlFor="num-clusters-input" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">Počet shluků (K):</label>
                        <input id="num-clusters-input" type="number" min="2" placeholder="Auto (pokud nevyplněno)" value={numClusters === null ? '' : numClusters} onChange={(e) => {const val = e.target.value; setNumClusters(val ? Math.max(2, parseInt(val, 10)) : null);}} className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500"/>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Pro {algorithm === 'kmeans' ? 'K-Means' : 'Hierarchické'}. Nechte prázdné pro automatickou detekci.</p>
                    </div>
                )}
            </div>
            {/* Standardizace */}
            <div>
                {/* ... kód pro standardizaci ... */}
                <label className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={standardize} onChange={() => setStandardize(!standardize)} className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-blue-600 dark:ring-offset-gray-800"/>
                    <span className="text-sm text-gray-800 dark:text-gray-200">Standardizovat data (Z-score)</span>
                </label>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Zajišťuje, že proměnné s různým měřítkem mají stejný vliv. Doporučeno ponechat zapnuté.</p>
            </div>
            {/* Tlačítko Spustit */}
            <button onClick={handleRun} disabled={selectedCols.length < 2 || loading} className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-md shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out">
                {/* ... kód pro loading ikonu ... */}
                {loading ? (<div className="flex items-center justify-center"><svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Analyzuji...</div>) : "Spustit shlukovou analýzu"}
            </button>
            {/* --------------------------------------------------------- */}


            {/* Zobrazení chyby */}
            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative dark:bg-red-900 dark:border-red-700 dark:text-red-200" role="alert">
                    <strong className="font-bold">Chyba: </strong>
                    <span className="block sm:inline">{error}</span>
                </div>
            )}

            {/* --- Zobrazení výsledků (hlavní analýzy) --- */}
            {result && (
                <div className="space-y-6 mt-6 border-t pt-6 dark:border-gray-700">
                    <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Výsledky analýzy</h2>

                    {/* Souhrnné informace */}
                    <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded border dark:border-gray-700 shadow-sm">
                        {/* ... kód pro zobrazení základních informací ... */}
                        <h3 className="text-lg font-medium mb-2 text-gray-700 dark:text-gray-200">Základní informace</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1 text-sm text-gray-600 dark:text-gray-300">
                            <p><strong className="font-medium text-gray-700 dark:text-gray-400">Použitá metoda:</strong> {result.method}</p>
                            {result.reason && <p className="col-span-full"><strong className="font-medium text-gray-700 dark:text-gray-400">Důvod výběru metody:</strong> {result.reason}</p>}
                            <p><strong className="font-medium text-gray-700 dark:text-gray-400">Počet shluků:</strong> {result.clusters ?? 'N/A (DBSCAN)'}</p>
                            <p><strong className="font-medium text-gray-700 dark:text-gray-400">Vzdálenostní metrika:</strong> {result.distance_metric}</p>
                            {result.silhouette_score !== null && <p><strong className="font-medium text-gray-700 dark:text-gray-400">Silhouette Score:</strong> {result.silhouette_score.toFixed(3)}</p>}
                        </div>
                    </div>

                    {/* Souhrnná tabulka shluků */}
                    {renderSummaryTable()}

                    {/* Grafy */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                        {/* ... kód pro zobrazení PCA a Biplot grafů ... */}
                        {result.pca_2d && (
                            <div className="bg-white dark:bg-gray-800 p-4 rounded border dark:border-gray-700 shadow-sm">
                                <h3 className="text-lg font-medium mb-2 text-center text-gray-700 dark:text-gray-200">Vizualizace shluků (PCA 2D)</h3>
                                <Plot
                                    data={[{ x: result.pca_2d.x, y: result.pca_2d.y, type: 'scatter', mode: 'markers', marker: { color: result.pca_2d.labels, colorscale: 'Viridis', showscale: true, size: 8, opacity: 0.8, colorbar: { title: 'Shluk', titleside: 'right' } }, text: result.pca_2d.labels.map(l => l === -1 ? 'Šum' : `Shluk ${l}`), hoverinfo: 'text' }]}
                                    layout={{ autosize: true, xaxis: { title: 'První hlavní komponenta (PC1)', color: '#6b7280', gridcolor: '#e5e7eb', zerolinecolor: '#e5e7eb' }, yaxis: { title: 'Druhá hlavní komponenta (PC2)', color: '#6b7280', gridcolor: '#e5e7eb', zerolinecolor: '#e5e7eb' }, hovermode: 'closest', margin: { l: 50, r: 20, t: 30, b: 50 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#374151' } }}
                                    useResizeHandler={true} className="w-full h-96" config={{responsive: true, displayModeBar: false}}
                                />
                            </div>
                        )}
                        {result.pca_components && result.pca_2d && (
                            <div className="bg-white dark:bg-gray-800 p-4 rounded border dark:border-gray-700 shadow-sm">
                                <h3 className="text-lg font-medium mb-2 text-center text-gray-700 dark:text-gray-200">Biplot (PCA + Směry proměnných)</h3>
                                <Plot
                                    data={[ { x: result.pca_2d.x, y: result.pca_2d.y, type: 'scatter', mode: 'markers', marker: { color: result.pca_2d.labels, colorscale: 'Viridis', showscale: true, size: 8, opacity: 0.7, colorbar: { title: 'Shluk', titleside: 'right' } }, text: result.pca_2d.labels.map(l => l === -1 ? 'Šum' : `Shluk ${l}`), hoverinfo: 'text', name: 'Data body' }, ...result.pca_components.map(vec => { const scaleFactor = Math.max( Math.max(...result.pca_2d.x.map(Math.abs)), Math.max(...result.pca_2d.y.map(Math.abs)) ) / Math.sqrt(vec.x**2 + vec.y**2) * 0.3; return { x: [0, vec.x * scaleFactor], y: [0, vec.y * scaleFactor], type: 'scatter', mode: 'lines+text', line: { color: '#ef4444', width: 2 }, text: ['', `<b>${vec.name}</b>`], textfont: { size: 10, color: '#dc2626' }, textposition: 'middle right', hoverinfo: 'text', name: vec.name, showlegend: false }; }) ]}
                                    layout={{ autosize: true, xaxis: { title: 'PC1', zeroline: true, color: '#6b7280', gridcolor: '#e5e7eb', zerolinecolor: '#d1d5db' }, yaxis: { title: 'PC2', zeroline: true, color: '#6b7280', gridcolor: '#e5e7eb', zerolinecolor: '#d1d5db' }, hovermode: 'closest', legend: { traceorder: 'reversed', orientation: 'h', yanchor: 'bottom', y: -0.2, xanchor: 'center', x: 0.5 }, margin: { l: 50, r: 20, t: 30, b: 70 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#374151' } }}
                                    useResizeHandler={true} className="w-full h-96" config={{responsive: true, displayModeBar: false}}
                                />
                            </div>
                        )}
                    </div>

                    {/* --- Sekce pro AI interpretaci --- */}
                    <div className="mt-8 pt-6 border-t dark:border-gray-700">
                        <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-100">Interpretace výsledků pomocí AI</h3>
                        <button
                            onClick={handleAiInterpretation}
                            disabled={aiLoading}
                            className="bg-green-600 hover:bg-green-700 text-white px-5 py-2 rounded-md shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out mb-4"
                        >
                            {aiLoading ? (
                                <div className="flex items-center justify-center">
                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Generuji interpretaci...
                                </div>
                            ) : "Analyzovat pomocí AI"}
                        </button>

                        {/* Zobrazení stavu/výsledku AI */}
                        {aiLoading && (
                            <p className="text-sm text-gray-600 dark:text-gray-400">Počkejte prosím, AI analyzuje výsledky...</p>
                        )}
                        {aiError && (
                            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative dark:bg-red-900 dark:border-red-700 dark:text-red-200" role="alert">
                                <strong className="font-bold">Chyba AI: </strong>
                                <span className="block sm:inline">{aiError}</span>
                            </div>
                        )}
                        {aiInterpretation && !aiLoading && (
                            <div className="mt-4 p-4 border rounded bg-gray-50 dark:bg-gray-800 dark:border-gray-700 shadow-sm">
                                <h4 className="font-medium mb-2 text-gray-700 dark:text-gray-200">Interpretace AI:</h4>
                                {/* Použití whitespace-pre-wrap pro zachování formátování z AI odpovědi */}
                                <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{aiInterpretation}</p>
                            </div>
                        )}
                    </div>
                    {/* --- Konec sekce pro AI interpretaci --- */}

                </div>
            )}
        </div>
    );
}