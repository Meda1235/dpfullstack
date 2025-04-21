import React, { useEffect, useState, useCallback } from 'react'; // Přidán useCallback

// --- Rozhraní (Beze změny) ---
interface ColumnInfo {
    name: string;
    type: 'Číselný' | 'Kategorie';
}

interface GroupTestResult { // Přejmenováno pro přehlednost
    numericColumn: string;
    test: string;
    pValue: number;
    isSignificant: boolean;
    note: string;
}

interface GroupComparisonResult {
    groupColumn: string;
    results: GroupTestResult[];
}

// --- Komponenta ---
export default function GroupComparisonDialog() {
    // --- Stávající stavy ---
    const [columns, setColumns] = useState<ColumnInfo[]>([]);
    const [selectedNumerical, setSelectedNumerical] = useState<string[]>([]);
    const [selectedCategorical, setSelectedCategorical] = useState<string[]>([]);
    const [paired, setPaired] = useState<boolean>(false);
    const [results, setResults] = useState<GroupComparisonResult[]>([]);
    const [loading, setLoading] = useState(false); // Pro načítání analýzy
    const [error, setError] = useState<string | null>(null);

    // --- Nové stavy pro AI interpretaci ---
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    const [isInterpreting, setIsInterpreting] = useState<boolean>(false);
    const [aiError, setAiError] = useState<string | null>(null);
    // --- Konec nových stavů ---

    // --- Načtení sloupců ---
    useEffect(() => {
        // Reset stavů při prvním načtení
        setError(null);
        setResults([]);
        setAiInterpretation(null);
        setAiError(null);
        // Nenastavujeme setLoading(true) zde, protože useEffect nemá přímý async fetch

        fetch('http://localhost:8000/api/get_column_types')
            .then(res => {
                if (!res.ok) throw new Error(`Chyba serveru: ${res.status}`);
                return res.json();
            })
            .then((data: ColumnInfo[]) => {
                setColumns(data);
                setError(null);
            })
            .catch(err => {
                console.error("Fetch columns error:", err);
                setError(`Nepodařilo se načíst sloupce: ${err.message}`);
                setColumns([]);
            });
        // .finally(() => {}); // Nelze použít finally přímo zde bez async/await
    }, []);

    // --- Funkce pro resetování výsledků a interpretace ---
    const resetResultsAndInterpretation = useCallback(() => {
        setResults([]);
        setError(null);
        setAiInterpretation(null);
        setAiError(null);
    }, []);


    // --- Automatické vypnutí párového režimu ---
    useEffect(() => {
        if (paired && (selectedNumerical.length !== 2 || selectedCategorical.length !== 1)) {
            setPaired(false);
            // Reset results if paired mode becomes invalid due to selection change
            resetResultsAndInterpretation();
        }
    }, [selectedNumerical, selectedCategorical, paired, resetResultsAndInterpretation]); // Přidána závislost

    // --- Přepínání výběru sloupců ---
    const toggleSelection = (col: string, type: 'numerical' | 'categorical') => {
        const target = type === 'numerical' ? selectedNumerical : selectedCategorical;
        const setter = type === 'numerical' ? setSelectedNumerical : setSelectedCategorical;

        if (target.includes(col)) {
            setter(target.filter(c => c !== col));
        } else {
            setter([...target, col]);
        }
        resetResultsAndInterpretation(); // Reset při změně výběru
    };

    // --- Změna párového režimu ---
    const handlePairedChange = (isChecked: boolean) => {
        setPaired(isChecked);
        resetResultsAndInterpretation(); // Reset při změně párování
    };


    // --- Spuštění analýzy ---
    const startAnalysis = useCallback(async () => {
        if (selectedNumerical.length === 0 || selectedCategorical.length === 0) {
            setError("Vyberte prosím alespoň jednu číselnou a jednu kategoriální proměnnou.");
            setResults([]);
            setAiInterpretation(null);
            setAiError(null);
            return;
        }
        // Reset před spuštěním
        resetResultsAndInterpretation();
        setLoading(true);

        try {
            const response = await fetch('http://localhost:8000/api/group_comparison', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    numerical: selectedNumerical,
                    categorical: selectedCategorical,
                    paired: paired
                })
            });

            const data = await response.json(); // Zkusit parsovat vždy

            if (!response.ok) {
                throw new Error(data.detail || 'Neznámá chyba při analýze.');
            }
            setResults(data.results || []); // Očekáváme { results: [...] }
            setError(null); // Vyčistit chybu

        } catch (err: any) {
            console.error("Group comparison error:", err);
            setError(`Chyba při analýze: ${err.message}`);
            setResults([]); // Vyčistit výsledky
        } finally {
            setLoading(false);
        }
    }, [selectedNumerical, selectedCategorical, paired, resetResultsAndInterpretation]); // Přidány závislosti


    // --- Nová funkce pro volání AI interpretace ---
    const handleInterpret = useCallback(async () => {
        if (!results || results.length === 0) return;

        setIsInterpreting(true);
        setAiInterpretation(null);
        setAiError(null);

        // Připravíme data pro AI - posíláme strukturovaný přehled výsledků
        const interpretationPayload = {
            analysis_type: "group_comparison",
            paired_analysis: paired,
            comparisons: results.map(groupResult => ({
                group_variable: groupResult.groupColumn,
                tests_performed: groupResult.results.map(test => ({
                    numeric_variable: test.numericColumn,
                    test_name: test.test,
                    p_value: test.pValue,
                    is_significant: test.isSignificant,
                    notes: test.note // Posíláme i poznámku o důvodu výběru testu
                }))
            }))
        };

        try {
            // Cíl: Nový endpoint na backendu
            const response = await fetch("http://localhost:8000/api/interpret_group_comparison", {
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
    }, [results, paired]); // Závislost na výsledcích a 'paired'
    // --- Konec nové funkce ---


    // --- JSX ---
    return (
        <div className="border rounded p-6 shadow bg-white dark:bg-gray-900 space-y-6">
            <fieldset disabled={loading || isInterpreting}> {/* Disable inputs during any loading */}
                <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">Výběr proměnných pro porovnání skupin</h2>
                <p className="text-sm mb-4 text-gray-600 dark:text-gray-300">
                    Vyberte jednu nebo více číselných proměnných a jednu nebo více kategoriálních proměnných.
                    Pro každou kombinaci bude provedeno samostatné statistické porovnání mezi skupinami.
                </p>

                {/* Výběr sloupců */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
                    <div>
                        <h3 className="font-medium mb-2 text-gray-800 dark:text-gray-200">Kategorie (rozdělení skupin)</h3>
                        <div className="space-y-1 max-h-48 overflow-y-auto border rounded p-2 dark:border-gray-600 bg-gray-50 dark:bg-gray-800/50">
                            {columns.filter(c => c.type === 'Kategorie').length === 0 && <p className="text-xs text-gray-500 italic">Žádné</p>}
                            {columns
                                .filter(c => c.type === 'Kategorie')
                                .map(col => (
                                    <label key={col.name} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 p-1 rounded cursor-pointer">
                                        <input
                                            type="checkbox"
                                            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800"
                                            checked={selectedCategorical.includes(col.name)}
                                            onChange={() => toggleSelection(col.name, 'categorical')}
                                        />
                                        {col.name}
                                    </label>
                                ))}
                        </div>
                    </div>
                    <div>
                        <h3 className="font-medium mb-2 text-gray-800 dark:text-gray-200">Číselné proměnné (k porovnání)</h3>
                        <div className="space-y-1 max-h-48 overflow-y-auto border rounded p-2 dark:border-gray-600 bg-gray-50 dark:bg-gray-800/50">
                            {columns.filter(c => c.type === 'Číselný').length === 0 && <p className="text-xs text-gray-500 italic">Žádné</p>}
                            {columns
                                .filter(c => c.type === 'Číselný')
                                .map(col => (
                                    <label key={col.name} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 p-1 rounded cursor-pointer">
                                        <input
                                            type="checkbox"
                                            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800"
                                            checked={selectedNumerical.includes(col.name)}
                                            onChange={() => toggleSelection(col.name, 'numerical')}
                                        />
                                        {col.name}
                                    </label>
                                ))}
                        </div>
                    </div>
                </div>

                {/* Párová data */}
                <div className="mb-4 p-3 border rounded dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
                    <h4 className="font-semibold text-sm mb-1 text-gray-800 dark:text-gray-200">Párová data (např. před/po měření)</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                        Režim pro porovnání dvou číselných proměnných u stejného subjektu (vyžaduje přesně 2 číselné a 1 kategoriální).
                    </p>
                    <label className="flex items-center text-sm text-gray-700 dark:text-gray-300 cursor-pointer">
                        <input
                            type="checkbox"
                            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                            checked={paired}
                            onChange={(e) => handlePairedChange(e.target.checked)}
                            // Podmínka pro disable musí být přesná
                            disabled={!(selectedNumerical.length === 2 && selectedCategorical.length === 1)}
                        />
                        <span className="ml-2">Aktivovat párový test</span>
                        {/* Tooltip nebo nápověda, proč je disabled */}
                        {!(selectedNumerical.length === 2 && selectedCategorical.length === 1) && (
                            <span className="ml-2 text-xs text-gray-400">(Vyberte 2 číselné a 1 kat.)</span>
                        )}
                    </label>
                </div>
            </fieldset> {/* Konec fieldsetu */}

            {/* Tlačítko Spustit Analýzu */}
            <button
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out"
                disabled={selectedNumerical.length === 0 || selectedCategorical.length === 0 || loading || isInterpreting}
                onClick={startAnalysis}
            >
                {loading ? (
                    <>
                        <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                        Analyzuji...
                    </>
                ) : 'Spustit analýzu'}
            </button>

            {/* Chyba analýzy */}
            {error && (
                <div role="alert" className="mt-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                    ⚠️ Chyba analýzy: {error}
                </div>
            )}

            {/* Výsledky analýzy */}
            {results.length > 0 && !loading && ( // Zobrazit jen pokud jsou výsledky a není loading
                <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700 space-y-6">
                    <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Výsledky porovnání skupin</h2>
                    {results.map((group, i) => (
                        <div key={i} className="border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm overflow-hidden">
                            <h3 className="text-base font-semibold mb-0 px-4 py-2 bg-gray-50 dark:bg-gray-800 text-gray-800 dark:text-gray-200 border-b dark:border-gray-700">
                                Skupinová proměnná: <span className='font-bold'>{group.groupColumn}</span>
                            </h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead className="bg-gray-100 dark:bg-gray-700/50 text-left">
                                    <tr>
                                        <th className="px-3 py-2 text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Číselná proměnná</th>
                                        <th className="px-3 py-2 text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Použitý test</th>
                                        <th className="px-3 py-2 text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider text-center">p-hodnota</th>
                                        <th className="px-3 py-2 text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider text-center">Významné?</th>
                                        <th className="px-3 py-2 text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Poznámka (důvod testu)</th>
                                    </tr>
                                    </thead>
                                    <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                                    {group.results.map((res, j) => (
                                        <tr key={j} className="hover:bg-gray-50 dark:hover:bg-gray-800/40">
                                            <td className="px-3 py-2 whitespace-nowrap text-gray-800 dark:text-gray-200">{res.numericColumn}</td>
                                            <td className="px-3 py-2 whitespace-nowrap text-gray-700 dark:text-gray-300">{res.test}</td>
                                            <td className={`px-3 py-2 whitespace-nowrap font-mono text-center ${res.isSignificant ? 'font-bold text-green-700 dark:text-green-400' : 'text-gray-600 dark:text-gray-400'}`}>
                                                {res.pValue < 0.001 ? res.pValue.toExponential(2) : res.pValue.toFixed(3)}
                                            </td>
                                            <td className="px-3 py-2 whitespace-nowrap text-center">
                                                {res.isSignificant
                                                    ? <span title="Statisticky významné (p < 0.05)" className="text-green-600 dark:text-green-400">✔️ Ano</span>
                                                    : <span title="Statisticky nevýznamné (p >= 0.05)" className="text-red-500 dark:text-red-400">✖️ Ne</span>}
                                            </td>
                                            <td className="px-3 py-2 text-xs text-gray-500 dark:text-gray-400">{res.note}</td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ))}

                    {/* --- Nová Sekce pro AI Interpretaci --- */}
                    <div className="pt-6 border-t border-dashed border-gray-300 dark:border-gray-600">
                        <h4 className="text-base font-semibold mb-3 text-gray-800 dark:text-gray-100">Interpretace pomocí AI</h4>

                        {/* Tlačítko Interpretovat */}
                        {!aiInterpretation && !isInterpreting && !aiError && (
                            <button
                                onClick={handleInterpret}
                                disabled={isInterpreting}
                                className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out"
                            >
                                💡 Interpretovat výsledky pomocí AI
                            </button>
                        )}

                        {/* Načítání interpretace */}
                        {isInterpreting && (
                            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 p-2 rounded bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600">
                                <svg className="animate-spin h-4 w-4 text-indigo-600 dark:text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                                AI generuje interpretaci, vyčkejte prosím...
                            </div>
                        )}

                        {/* Chyba interpretace */}
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

                        {/* Úspěšná interpretace */}
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
            )} {/* Konec Zobrazení výsledků */}
        </div> // Konec hlavní komponenty
    );
}