// FactorAnalysis.js
import React, { useEffect, useState } from 'react';
// Můžeme zvážit Plotly pro heatmapu zátěží, ale začneme s tabulkou

// Rozhraní pro typ sloupce (předpoklad)
interface ColumnType {
    name: string;
    type: string; // Např. "Číselný", "Kategorie"
}

// Rozhraní pro výsledek FA (MUSÍ odpovídat backendu!)
interface FactorAnalysisResult {
    columns_used: string[];
    n_factors_requested: number | null;
    n_factors_extracted: number;
    eigenvalue_criterion_used: boolean;
    eigenvalues: (number | null)[] | null; // Může obsahovat null
    rotation_used: string;
    standardized: boolean;
    dropped_rows: number;
    data_adequacy: {
        kmo_model: number | null;
        bartlett_chi_square: number | null;
        bartlett_p_value: number | null;
    };
    factor_loadings: {
        [variable: string]: {
            [factor: string]: number | null; // Může obsahovat null
        };
    };
    factor_variance: {
        factor: string;
        ssl: number;
        variance_pct: number;
        cumulative_variance_pct: number;
    }[];
    total_variance_explained_pct: number | null;
    communalities: {
        [variable: string]: number | null; // Může obsahovat null
    };
}

// Rozhraní pro data odesílaná na backend pro AI interpretaci
interface FactorAnalysisInterpretationPayload {
    analysis_type: 'factor_analysis';
    columns_used: string[];
    n_factors_extracted: number;
    rotation_used: string;
    standardized: boolean;
    dropped_rows: number;
    kmo_model: number | null | undefined;
    bartlett_p_value: number | null | undefined;
    total_variance_explained_pct: number | null | undefined;
}


// Pomocná funkce pro formátování čísel
const formatNumber = (num: number | undefined | null, decimals = 3): string => {
    if (typeof num !== 'number' || isNaN(num)) {
        return '-';
    }
    // Zobrazíme nulu jako 0.000 pokud je to relevantní (např. p-value)
    if (num === 0 && decimals > 0) {
        return `0.${'0'.repeat(decimals)}`;
    }
    // Použijeme Intl.NumberFormat pro správné oddělovače (pokud je třeba) a zaokrouhlení
    const formatter = new Intl.NumberFormat('cs-CZ', { // Nebo 'en-US' podle potřeby
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });
    // Nahradíme případnou nezlomitelnou mezeru normální mezerou pro konzistenci
    return formatter.format(num).replace(/\s/g, ' ');
    // Alternativa: jednoduché toFixed
    // const fixed = num.toFixed(decimals);
    // return fixed;
};

// Funkce pro zvýraznění signifikantních zátěží
const getLoadingCellStyle = (loading: number | undefined | null): React.CSSProperties => {
    const threshold = 0.4; // Běžný práh pro "významnou" zátěž
    if (typeof loading === 'number' && Math.abs(loading) >= threshold) {
        return { fontWeight: 'bold', backgroundColor: Math.abs(loading) > 0.6 ? '#f0f9ff' : '#f8fafc' }; // Zvýraznění pro světlý režim
        // Pro tmavý režim (odkomentujte pokud potřebujete detekci)
        // return { fontWeight: 'bold', color: '#fff', backgroundColor: Math.abs(loading) > 0.6 ? 'rgba(59, 130, 246, 0.3)' : 'rgba(71, 85, 105, 0.3)' };
    }
    return {};
};


export default function FactorAnalysis() {
    const [numericColumns, setNumericColumns] = useState<ColumnType[]>([]);
    const [selectedCols, setSelectedCols] = useState<string[]>([]);

    // Parametry FA
    const [numFactors, setNumFactors] = useState<string>(''); // Použijeme string pro input, pak parsujeme
    const [rotation, setRotation] = useState('varimax');
    const [standardize, setStandardize] = useState(true);

    // Výsledky a stav UI
    const [result, setResult] = useState<FactorAnalysisResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Stav pro AI Interpretaci
    const [interpretationLoading, setInterpretationLoading] = useState(false);
    const [interpretationError, setInterpretationError] = useState<string | null>(null);
    const [interpretationResult, setInterpretationResult] = useState<string | null>(null);

    // Načtení sloupců
    useEffect(() => {
        setLoading(true);
        fetch("http://localhost:8000/api/get_column_types")
            .then(res => {
                if (!res.ok) throw new Error('Nepodařilo se načíst sloupce');
                return res.json();
            })
            .then((data: ColumnType[]) => {
                const numeric = data.filter(col => col.type === "Číselný");
                setNumericColumns(numeric);
            })
            .catch(err => setError(err.message))
            .finally(() => setLoading(false));
    }, []);

    // Handler pro spuštění hlavní analýzy
    const handleRun = async () => {
        if (selectedCols.length < 3) {
            setError("Pro faktorovou analýzu vyberte alespoň 3 numerické proměnné.");
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);
        // Vyčistit předchozí interpretaci při novém spuštění
        setInterpretationResult(null);
        setInterpretationError(null);

        try {
            // Parsování počtu faktorů, null pokud prázdné nebo nevalidní
            const nFactorsParsed = numFactors.trim() ? parseInt(numFactors, 10) : null;
            if (numFactors.trim() && (isNaN(nFactorsParsed!) || nFactorsParsed! < 1)) {
                throw new Error("Počet faktorů musí být kladné celé číslo, nebo nechte pole prázdné pro automatickou detekci.");
            }

            const requestBody = {
                columns: selectedCols,
                n_factors: nFactorsParsed,
                rotation: rotation,
                standardize: standardize,
            };

            const response = await fetch("http://localhost:8000/api/factor_analysis", {
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
                    throw new Error(`Chyba ${response.status}: ${responseText}`);
                }
            }

            const data: FactorAnalysisResult = JSON.parse(responseText);
            setResult(data);
        } catch (err: any) {
            console.error("Factor Analysis fetch error:", err);
            setError(err.message || "Nastala neočekávaná chyba při faktorové analýze.");
        } finally {
            setLoading(false);
        }
    };

    // Handler pro spuštění AI interpretace
    const handleInterpret = async () => {
        if (!result) return;

        setInterpretationLoading(true);
        setInterpretationError(null);
        setInterpretationResult(null);

        // Připravit data pro endpoint interpretace
        const payload: FactorAnalysisInterpretationPayload = {
            analysis_type: 'factor_analysis',
            columns_used: result.columns_used,
            n_factors_extracted: result.n_factors_extracted,
            rotation_used: result.rotation_used,
            standardized: result.standardized,
            dropped_rows: result.dropped_rows, // Může být užitečné pro kontext
            kmo_model: result.data_adequacy.kmo_model,
            bartlett_p_value: result.data_adequacy.bartlett_p_value,
            total_variance_explained_pct: result.total_variance_explained_pct,
        };

        try {
            const response = await fetch("http://localhost:8000/api/interpret_factor_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const responseText = await response.text(); // Přečíst text pro lepší chybové hlášení
            if (!response.ok) {
                try {
                    const errorJson = JSON.parse(responseText);
                    throw new Error(errorJson.detail || `Chyba ${response.status}: ${response.statusText}`);
                } catch (e) {
                    // Pokud parsování JSON selže, použijeme surový text
                    throw new Error(`Chyba ${response.status}: ${responseText}`);
                }
            }

            const data = JSON.parse(responseText);
            if (data.interpretation) {
                setInterpretationResult(data.interpretation);
            } else {
                throw new Error("AI nevrátila platnou interpretaci.");
            }

        } catch (err: any) {
            console.error("AI Interpretation fetch error:", err);
            setInterpretationError(err.message || "Nastala neočekávaná chyba při komunikaci s AI.");
        } finally {
            setInterpretationLoading(false);
        }
    };


    // --- Renderovací funkce ---

    const renderColumnSelector = () => (
        <div>
            <label className="block font-medium mb-2 text-gray-700 dark:text-gray-300">1. Vyberte numerické proměnné (min. 3):</label>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 p-3 border rounded bg-gray-50 dark:bg-gray-800 dark:border-gray-700 max-h-60 overflow-y-auto">
                {numericColumns.length === 0 && !loading && <p className="text-sm text-gray-500 dark:text-gray-400 col-span-full">Žádné numerické sloupce nenalezeny.</p>}
                {numericColumns.map((col) => (
                    <label key={col.name} className="flex items-center gap-2 text-sm p-1 rounded cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700">
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
                        <span className="text-gray-800 dark:text-gray-200">{col.name}</span>
                    </label>
                ))}
            </div>
            {selectedCols.length > 0 && selectedCols.length < 3 && (
                <p className="text-xs text-red-600 mt-1">Vyberte alespoň 3 proměnné.</p>
            )}
        </div>
    );

    const renderParameterSelectors = () => (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
                <label htmlFor="num-factors" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">2. Počet faktorů:</label>
                <input
                    id="num-factors"
                    type="number"
                    min="1"
                    step="1"
                    placeholder="Auto (Kaiserovo kr.)"
                    value={numFactors}
                    onChange={(e) => setNumFactors(e.target.value)}
                    className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Nechte prázdné pro autodetekci (eigenvalue &gt; 1).</p>
            </div>
            <div>
                <label htmlFor="rotation-select" className="block font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">3. Metoda rotace:</label>
                <select
                    id="rotation-select"
                    value={rotation}
                    onChange={e => setRotation(e.target.value)}
                    className="w-full border border-gray-300 rounded p-2 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500"
                >
                    <option value="varimax">Varimax (ortogonální)</option>
                    <option value="promax">Promax (šikmá)</option>
                    <option value="oblimin">Oblimin (šikmá)</option>
                    <option value="quartimax">Quartimax (ortogonální)</option>
                    <option value="none">Bez rotace</option>
                </select>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Varimax je běžná volba pro nezávislé faktory.</p>
            </div>
            <div className="flex items-center pt-5">
                <label className="flex items-center gap-2 cursor-pointer">
                    <input
                        type="checkbox"
                        checked={standardize}
                        onChange={() => setStandardize(!standardize)}
                        className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-blue-600 dark:ring-offset-gray-800"
                    />
                    <span className="text-sm text-gray-800 dark:text-gray-200">Standardizovat data</span>
                </label>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 ml-2">(Doporučeno)</p>
            </div>
        </div>
    );

    const renderResults = () => {
        if (!result) return null;

        const renderAdequacyTests = () => {
            const kmo = result.data_adequacy.kmo_model;
            const bartlettP = result.data_adequacy.bartlett_p_value;
            let kmoInterpretation = '';
            if (kmo !== null) {
                if (kmo < 0.5) kmoInterpretation = 'Nepřijatelné';
                else if (kmo < 0.6) kmoInterpretation = 'Mizerné';
                else if (kmo < 0.7) kmoInterpretation = 'Slabé';
                else if (kmo < 0.8) kmoInterpretation = 'Střední';
                else if (kmo < 0.9) kmoInterpretation = 'Dobré';
                else kmoInterpretation = 'Vynikající';
            }
            const bartlettInterpretation = bartlettP !== null ? (bartlettP < 0.05 ? 'Data jsou vhodná (odmítáme H0 o jednotkové korelační matici)' : 'Data nemusí být vhodná (nelze odmítnout H0)') : 'N/A';

            return (
                <div className="bg-yellow-50 dark:bg-yellow-900 border border-yellow-200 dark:border-yellow-700 p-4 rounded shadow-sm text-sm">
                    <h4 className="text-md font-medium mb-2 text-yellow-800 dark:text-yellow-200">Testy vhodnosti dat</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1">
                        <div>
                            <strong className="font-medium text-yellow-700 dark:text-yellow-300">KMO (Kaiser-Meyer-Olkin):</strong> {formatNumber(kmo, 4)}
                            {kmoInterpretation && <span className="ml-2 text-yellow-600 dark:text-yellow-400">({kmoInterpretation})</span>}
                            <p className="text-xs text-yellow-500 dark:text-yellow-500">Hodnoty &gt; 0.6 jsou obvykle považovány za přijatelné.</p>
                        </div>
                        <div>
                            <strong className="font-medium text-yellow-700 dark:text-yellow-300">Bartlettův test sfér.:</strong> p-hodnota = {formatNumber(bartlettP, 4)}
                            {bartlettP !== null && <span className={`ml-2 ${bartlettP < 0.05 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>({bartlettInterpretation})</span>}
                            <p className="text-xs text-yellow-500 dark:text-yellow-500">Signifikantní výsledek (p &lt; 0.05) naznačuje, že data jsou vhodná pro FA.</p>
                        </div>
                    </div>
                </div>
            );
        };

        const renderLoadingsTable = () => (
            <div className="mt-4">
                <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-200">Faktorové zátěže (Loadings)</h4>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Ukazují korelaci původních proměnných s extrahovanými faktory. Tučně jsou zvýrazněny hodnoty &gt; 0.4.</p>
                <div className="overflow-x-auto shadow rounded-lg border border-gray-200 dark:border-gray-700">
                    <table className="min-w-full border-collapse bg-white dark:bg-gray-800 text-sm">
                        <thead className="bg-gray-100 dark:bg-gray-700">
                        <tr>
                            <th className="sticky left-0 z-10 bg-gray-100 dark:bg-gray-700 p-2 border-b border-r dark:border-gray-600 text-left font-medium text-gray-600 dark:text-gray-300">Proměnná</th>
                            {/* Získání názvů faktorů z variance_info */}
                            {result.factor_variance.map(f => (
                                <th key={f.factor} className="p-2 border-b dark:border-gray-600 font-medium text-gray-600 dark:text-gray-300">{f.factor}</th>
                            ))}
                            <th className="p-2 border-b dark:border-gray-600 font-medium text-gray-600 dark:text-gray-300">Komunalita</th>
                        </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {/* Zajistíme, že řádky odpovídají 'columns_used' pro konzistentní pořadí */}
                        {result.columns_used.map((variable) => {
                            const loadings = result.factor_loadings[variable];
                            const communality = result.communalities[variable];
                            if (!loadings) return null; // Pojistka
                            return (
                                <tr key={variable} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                    <td className="sticky left-0 z-10 bg-white dark:bg-gray-800 p-2 border-r border-gray-300 dark:border-gray-600 font-medium text-gray-900 dark:text-gray-200 whitespace-nowrap">{variable}</td>
                                    {/* Mapujeme přes stejné factor_variance pro zajištění pořadí sloupců */}
                                    {result.factor_variance.map(f => (
                                        <td key={`${variable}-${f.factor}`}
                                            className="p-2 text-center text-gray-700 dark:text-gray-300"
                                            style={getLoadingCellStyle(loadings[f.factor])} // Aplikace stylu
                                        >
                                            {formatNumber(loadings[f.factor], 3)}
                                        </td>
                                    ))}
                                    <td className="p-2 text-center bg-gray-50 dark:bg-gray-700/50 text-gray-600 dark:text-gray-400 font-mono">{formatNumber(communality, 3)}</td>
                                </tr>
                            )
                        })}
                        </tbody>
                    </table>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">Komunalita: Podíl rozptylu proměnné vysvětlený všemi extrahovanými faktory.</p>
            </div>
        );

        const renderVarianceTable = () => (
            <div className="mt-4">
                <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-200">Vysvětlená variance faktory</h4>
                <div className="overflow-x-auto shadow rounded-lg border border-gray-200 dark:border-gray-700">
                    <table className="min-w-full border-collapse bg-white dark:bg-gray-800 text-sm">
                        <thead className="bg-gray-100 dark:bg-gray-700">
                        <tr>
                            <th className="p-2 border-b dark:border-gray-600 text-left font-medium text-gray-600 dark:text-gray-300">Faktor</th>
                            <th className="p-2 border-b dark:border-gray-600 font-medium text-gray-600 dark:text-gray-300">Sum of Squared Loadings (SSL)</th>
                            <th className="p-2 border-b dark:border-gray-600 font-medium text-gray-600 dark:text-gray-300">% Variance</th>
                            <th className="p-2 border-b dark:border-gray-600 font-medium text-gray-600 dark:text-gray-300">Kumulativní % Variance</th>
                        </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {result.factor_variance.map((f) => (
                            <tr key={f.factor} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                <td className="p-2 font-medium text-gray-900 dark:text-gray-200">{f.factor}</td>
                                <td className="p-2 text-center text-gray-700 dark:text-gray-300 font-mono">{formatNumber(f.ssl, 3)}</td>
                                <td className="p-2 text-center text-gray-700 dark:text-gray-300 font-mono">{formatNumber(f.variance_pct, 2)}%</td>
                                <td className="p-2 text-center text-gray-700 dark:text-gray-300 font-mono">{formatNumber(f.cumulative_variance_pct, 2)}%</td>
                            </tr>
                        ))}
                        </tbody>
                        <tfoot className="bg-gray-100 dark:bg-gray-700">
                        <tr>
                            <td colSpan={3} className="p-2 text-right font-medium text-gray-600 dark:text-gray-300">Celkem vysvětleno:</td>
                            <td className="p-2 text-center font-bold text-gray-700 dark:text-gray-200 font-mono">{formatNumber(result.total_variance_explained_pct, 2)}%</td>
                        </tr>
                        </tfoot>
                    </table>
                </div>
                {result.eigenvalue_criterion_used && result.eigenvalues && (
                    <details className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                        <summary className="cursor-pointer font-medium hover:text-gray-700 dark:hover:text-gray-300">Zobrazit Eigenvalues (použito pro autodetekci)</summary>
                        <pre className="mt-1 p-2 bg-gray-100 dark:bg-gray-700 rounded overflow-x-auto text-gray-600 dark:text-gray-400">
                              {result.eigenvalues.map((ev, i) => `PC${i+1}: ${formatNumber(ev, 4)}`).join('\n')}
                          </pre>
                    </details>
                )}
            </div>
        );

        // Hlavní return pro renderResults
        return (
            <div className="space-y-6 mt-6 border-t pt-6 dark:border-gray-700">
                <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">Výsledky Faktorové Analýzy</h3>

                {/* Sekce pro AI Interpretaci */}
                <div className="my-4 p-4 bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-200 dark:border-indigo-700/50 rounded shadow-sm">
                    <button
                        onClick={handleInterpret}
                        disabled={interpretationLoading || loading} // Deaktivovat pokud běží hlavní analýza nebo interpretace
                        className="inline-flex items-center bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out text-sm font-medium"
                    >
                        {interpretationLoading ? (
                            <>
                                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Generuji interpretaci...
                            </>
                        ) : "Interpretovat výsledky s AI"}
                    </button>

                    {interpretationError && (
                        <div className="mt-3 bg-red-100 border border-red-300 text-red-800 px-3 py-2 rounded text-xs dark:bg-red-900 dark:border-red-700 dark:text-red-200" role="alert">
                            <strong className="font-semibold">Chyba AI:</strong> {interpretationError}
                        </div>
                    )}

                    {interpretationResult && (
                        <div className="mt-3 p-4 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-600 shadow">
                            <h5 className="text-base font-semibold mb-2 text-indigo-800 dark:text-indigo-300">AI Interpretace:</h5>
                            {/* Použití pre-wrap pro respektování nových řádků z AI */}
                            <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">{interpretationResult}</p>
                        </div>
                    )}
                </div>
                {/* Konec Sekce AI Interpretace */}


                {/* Shrnutí konfigurace analýzy */}
                <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded border dark:border-gray-700 shadow-sm text-sm">
                    <h4 className="text-md font-medium mb-2 text-gray-700 dark:text-gray-200">Konfigurace Analýzy</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1 text-gray-600 dark:text-gray-300">
                        <p><strong className="font-medium text-gray-700 dark:text-gray-400">Extrahováno faktorů:</strong> {result.n_factors_extracted} {result.eigenvalue_criterion_used ? '(automaticky dle eigenvalue > 1)' : (result.n_factors_requested ? `(požadováno ${result.n_factors_requested})` : '')}</p>
                        <p><strong className="font-medium text-gray-700 dark:text-gray-400">Metoda rotace:</strong> {result.rotation_used}</p>
                        <p><strong className="font-medium text-gray-700 dark:text-gray-400">Standardizace:</strong> {result.standardized ? 'Ano' : 'Ne'}</p>
                        <p><strong className="font-medium text-gray-700 dark:text-gray-400">Odstraněno řádků (s NaN):</strong> {result.dropped_rows}</p>
                        <p className="col-span-full"><strong className="font-medium text-gray-700 dark:text-gray-400">Analyzované proměnné:</strong> {result.columns_used.join(', ')}</p>
                    </div>
                </div>

                {renderAdequacyTests()}
                {renderLoadingsTable()}
                {renderVarianceTable()}

            </div>
        );
    };

    // --- Hlavní Return Komponenty ---
    return (
        <div className="space-y-6 p-4 md:p-6 lg:p-8 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 min-h-screen">
            <h1 className="text-2xl font-bold mb-4 text-gray-800 dark:text-gray-100">Faktorová Analýza (FA)</h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                Identifikuje skryté (latentní) faktory vysvětlující korelace mezi pozorovanými numerickými proměnnými.
                Pomáhá redukovat dimenzionalitu dat a porozumět jejich struktuře. Vyžaduje interpretaci výsledků.
            </p>

            {/* Indikátor načítání sloupců */}
            {loading && numericColumns.length === 0 && <p className="text-gray-500 dark:text-gray-400">Načítám data sloupců...</p> }

            {/* Formulář pro výběr a parametry */}
            <div className="space-y-6">
                {renderColumnSelector()}
                {renderParameterSelectors()}
            </div>


            {/* Tlačítko Spustit analýzu */}
            <div className="mt-6">
                <button
                    onClick={handleRun}
                    disabled={selectedCols.length < 3 || loading || interpretationLoading} // Deaktivovat i během interpretace
                    className="bg-purple-600 hover:bg-purple-700 text-white px-5 py-2 rounded-md shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
                >
                    {loading ? (
                        <div className="flex items-center justify-center">
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Analyzuji...
                        </div>
                    ) : "Spustit Faktorovou Analýzu"}
                </button>
            </div>


            {/* Zobrazení hlavní chyby analýzy */}
            {error && (
                <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative dark:bg-red-900 dark:border-red-700 dark:text-red-200" role="alert">
                    <strong className="font-bold">Chyba analýzy: </strong>
                    <span className="block sm:inline">{error}</span>
                </div>
            )}

            {/* Zobrazení výsledků (včetně AI sekce) */}
            {renderResults()}

        </div>
    );
}