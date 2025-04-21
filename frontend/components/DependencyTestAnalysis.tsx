// components/DependencyTestAnalysis.tsx
import React, { useState, useEffect, useCallback } from 'react';

// --- Rozhraní (zůstává stejné, očekává 'test') ---
interface ColumnType {
    name: string;
    type: "Kategorie" | "Číselný" | string;
}
interface AnovaResult {
    cat_col: string;
    num_col: string;
    statistic: number | null;
    p_value: number | null;
}
interface ContingencyTable {
    [rowLabel: string]: {
        [colLabel: string]: number;
    };
}
interface DependencyResult {
    test: string; // Očekáváme tento klíč
    reason: string;
    p_value?: number | null;
    statistic?: number | null;
    statistic_name?: string | null;
    degrees_freedom?: number | string | null;
    contingency_table?: ContingencyTable | null;
    results?: AnovaResult[] | null;
    columns: string[];
    input_method: string;
    input_paired: boolean;
    warning_message?: string | null;
    // Můžeme sem volitelně přidat test_name pro typovou bezpečnost při mapování,
    // ale není to nutné, pokud k němu přistupujeme opatrně.
    // test_name?: string;
}
// Rozhraní pro payload AI interpretace
interface AIInterpretationPayload {
    analysis_type: string;
    test_name: string; // Zde používáme test_name konzistentně s tím, co AI endpoint očekává
    columns_involved: string[];
    paired_data: boolean;
    p_value?: number | null;
    statistic?: number | null;
    statistic_name?: string | null;
    degrees_freedom?: number | string | null;
    has_contingency_table: boolean;
    anova_results?: AnovaResult[] | null;
    warning_message?: string | null;
}


// --- Komponenta ---
export default function DependencyTestAnalysis() {
    // --- Stavy ---
    const [columns, setColumns] = useState<ColumnType[]>([]);
    const [categoricalCols, setCategoricalCols] = useState<ColumnType[]>([]);
    const [numericCols, setNumericCols] = useState<ColumnType[]>([]);
    const [selectedCats, setSelectedCats] = useState<string[]>([]);
    const [selectedNums, setSelectedNums] = useState<string[]>([]);
    const [method, setMethod] = useState<string>('auto');
    const [paired, setPaired] = useState<boolean>(false);
    const [result, setResult] = useState<DependencyResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [isFetchingCols, setIsFetchingCols] = useState<boolean>(false);
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    const [isInterpreting, setIsInterpreting] = useState<boolean>(false);
    const [aiError, setAiError] = useState<string | null>(null);

    // --- Načtení sloupců ---
    useEffect(() => {
        let isMounted = true;
        setIsFetchingCols(true);
        setError(null);
        fetch("http://localhost:8000/api/get_column_types")
            .then(res => {
                if (!res.ok) throw new Error(`Chyba serveru: ${res.status}`);
                return res.json();
            })
            .then((data: ColumnType[]) => {
                if (isMounted) {
                    setColumns(data);
                    setCategoricalCols(data.filter(col => col.type === "Kategorie"));
                    setNumericCols(data.filter(col => col.type === "Číselný"));
                    setError(null);
                }
            })
            .catch(err => {
                console.error("Fetch columns error:", err);
                if (isMounted) {
                    let displayError = "Nepodařilo se načíst sloupce.";
                    if (err instanceof Error) {
                        displayError = `Nepodařilo se načíst sloupce: ${err.message}`;
                        if (err.message.includes("Failed to fetch")) {
                            displayError = "Nepodařilo se připojit k serveru pro načtení sloupců. Zkontrolujte, zda běží backend.";
                        }
                    }
                    setError(displayError);
                    setColumns([]); setCategoricalCols([]); setNumericCols([]);
                }
            })
            .finally(() => {
                if (isMounted) {
                    setIsFetchingCols(false);
                }
            });

        return () => {
            isMounted = false;
        };
    }, []);

    // --- Funkce pro resetování ---
    const resetResultsAndInterpretation = useCallback(() => {
        setResult(null);
        setError(null);
        setAiInterpretation(null);
        setAiError(null);
    }, []);

    // --- Handlery změn ---
    const handleCatCheckboxChange = (col: string, checked: boolean) => {
        setSelectedCats(prev => checked ? [...prev, col] : prev.filter(c => c !== col));
        resetResultsAndInterpretation();
    };
    const handleNumCheckboxChange = (col: string, checked: boolean) => {
        setSelectedNums(prev => checked ? [...prev, col] : prev.filter(c => c !== col));
        resetResultsAndInterpretation();
    };
    const handleMethodChange = (newMethod: string) => {
        setMethod(newMethod); resetResultsAndInterpretation();
    };
    const handlePairedChange = (isChecked: boolean) => {
        setPaired(isChecked); resetResultsAndInterpretation();
    };

    // Kombinovaný seznam vybraných sloupců pro odeslání na backend
    const allSelectedCols = [...selectedCats, ...selectedNums];

    // --- Spuštění testu ---
    const runTest = useCallback(async () => {
        if (allSelectedCols.length < 2) {
            setError("Vyberte prosím alespoň dvě proměnné (kategoriální nebo číselné).");
            setResult(null); setAiInterpretation(null); setAiError(null); return;
        }
        resetResultsAndInterpretation();
        setIsLoading(true);

        let tempResult: DependencyResult | null = null;
        let tempError: string | null = null;

        try {
            console.log("runTest: Zahajuji fetch..."); // LADĚNÍ
            const res = await fetch("http://localhost:8000/api/dependency_test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ columns: allSelectedCols, method, paired }),
            });

            console.log("runTest: Fetch dokončen.", { status: res.status, ok: res.ok }); // LADĚNÍ

            const responseText = await res.text();
            let data: any = null;

            if (responseText) {
                try {
                    data = JSON.parse(responseText);
                    console.log("runTest: JSON úspěšně parsován:", data); // LADĚNÍ
                } catch (jsonError) {
                    console.error("runTest: Chyba parsování JSON odpovědi:", responseText, jsonError); // LADĚNÍ
                    if (!res.ok) {
                        tempError = `Chyba serveru (${res.status}). Odpověď nebyla validní JSON: ${responseText.substring(0, 100)}${responseText.length > 100 ? '...' : ''}`;
                    } else {
                        tempError = "Server vrátil neočekávaný formát odpovědi (nevalidní JSON).";
                    }
                    return; // Skoč do finally
                }
            } else if (!res.ok) {
                console.log(`runTest: Chyba serveru (${res.status}) s prázdnou odpovědí.`); // LADĚNÍ
                tempError = `Chyba serveru (${res.status}). Server nevrátil žádnou odpověď.`;
                return; // Skoč do finally
            }

            if (!res.ok) {
                // Zpracování chyby z backendu
                const errorMessage = data?.detail || `Chyba při analýze (${res.status}). Žádný detail nebyl poskytnut.`;
                console.log("runTest: Zjištěna chyba backendu:", errorMessage); // LADĚNÍ
                tempError = errorMessage;
                tempResult = null;
            } else {
                // Úspěšný případ - validace a mapování
                console.log("runTest: Úspěch, pokouším se validovat a mapovat výsledek."); // LADĚNÍ

                // Zkontrolujeme základní strukturu a přítomnost test_name nebo test
                if (data && typeof data === 'object' && ('test_name' in data || 'test' in data) && 'columns' in data) {

                    // MAPOVÁNÍ: Pokud existuje 'test_name' a neexistuje 'test', vytvoříme 'test'
                    if ('test_name' in data && !('test' in data)) {
                        data.test = data.test_name;
                        console.log("runTest: Mapováno 'test_name' na 'test'."); // LADĚNÍ
                        // Volitelně smazat původní klíč: delete data.test_name;
                    }

                    // Nyní by měl klíč 'test' existovat, provedeme finální přetypování
                    if ('test' in data) {
                        tempResult = data as DependencyResult;
                        console.log("runTest: Data úspěšně validována a přiřazena k tempResult."); // LADĚNÍ
                        tempError = null; // Zajistíme, že chyba je null
                    } else {
                        // Tento případ by neměl nastat po mapování, ale pro jistotu
                        console.error("runTest: Server vrátil OK, ale po mapování stále chybí klíč 'test':", data);
                        tempError = "Server vrátil neočekávaný formát odpovědi (chybí klíč 'test').";
                        tempResult = null;
                    }

                } else {
                    console.error("runTest: Server vrátil OK, ale data nemají očekávanou strukturu (nejsou objekt nebo chybí 'test_name'/'test'/'columns'):", data); // LADĚNÍ
                    tempError = "Server vrátil neočekávaný formát úspěšné odpovědi.";
                    tempResult = null;
                }
            }

        } catch (networkErr: any) {
            console.error("runTest: Síťová nebo jiná chyba:", networkErr); // LADĚNÍ
            tempResult = null;
            if (networkErr instanceof TypeError && networkErr.message.includes("Failed to fetch")) {
                tempError = "Nepodařilo se připojit k serveru. Zkontrolujte, zda běží backend a zda je adresa správná.";
            } else if (networkErr instanceof Error) {
                tempError = `Nastala neočekávaná chyba při komunikaci se serverem: ${networkErr.message}`;
            } else {
                tempError = "Nastala neočekávaná chyba při komunikaci se serverem.";
            }
        } finally {
            console.log("runTest: Finally blok. Nastavuji finální stav.", { tempError, tempResult }); // LADĚNÍ
            setError(tempError); // Použití temp proměnných
            setResult(tempResult);
            setIsLoading(false);
            console.log("runTest: Stav aktualizován, isLoading nastaveno na false."); // LADĚNÍ
        }
    }, [allSelectedCols, method, paired, resetResultsAndInterpretation]);

    // --- Volání AI interpretace ---
    const handleInterpret = useCallback(async () => {
        if (!result) return;
        // Použijeme result.test (který jsme mapovali) pro konzistenci s rozhraním,
        // ale do payloadu dáme test_name (pokud existuje) nebo test,
        // aby AI endpoint dostal název, který backend původně vygeneroval.
        const testNameToUse = (result as any).test_name || result.test; // Přístup k test_name i když není v interface

        setIsInterpreting(true); setAiInterpretation(null); setAiError(null);

        const payload: AIInterpretationPayload = {
            analysis_type: "dependency_test",
            test_name: testNameToUse, // Použijeme název z backendu
            columns_involved: result.columns,
            paired_data: result.input_paired,
            p_value: result.p_value,
            statistic: result.statistic,
            statistic_name: result.statistic_name,
            degrees_freedom: result.degrees_freedom,
            has_contingency_table: !!result.contingency_table,
            anova_results: result.results,
            warning_message: result.warning_message
        };
        console.log("AI Payload:", payload); // LADĚNÍ

        try {
            const response = await fetch("http://localhost:8000/api/interpret_dependency_test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const responseText = await response.text();
            let data: any = null;

            if(responseText){
                try {
                    data = JSON.parse(responseText);
                } catch (jsonError) {
                    console.error("AI Interpret JSON parse error:", responseText, jsonError);
                    if (!response.ok) {
                        throw new Error(`Chyba AI (${response.status}). Odpověď nebyla validní JSON.`);
                    } else {
                        throw new Error("AI endpoint vrátil neočekávaný formát odpovědi (nevalidní JSON).");
                    }
                }
            } else if (!response.ok) {
                throw new Error(`Chyba AI (${response.status}). Server nevrátil žádnou odpověď.`);
            }

            if (!response.ok) throw new Error(data?.detail || `Chyba AI (${response.status})`);

            const interpretationText = data.interpretation || data.choices?.[0]?.message?.content;
            if (interpretationText) {
                setAiInterpretation(interpretationText);
                setAiError(null);
            } else {
                throw new Error("AI nevrátila platný text.");
            }
        } catch (err: any) {
            console.error("AI Interpretation Error:", err);
            let aiDisplayError = "Chyba při získávání AI interpretace.";
            if (err instanceof Error) {
                aiDisplayError = `Chyba při získávání AI interpretace: ${err.message}`;
                if (err.message.includes("Failed to fetch")) {
                    aiDisplayError = "Nepodařilo se připojit k AI endpointu. Zkontrolujte backend.";
                }
            }
            setAiError(aiDisplayError);
            setAiInterpretation(null);
        } finally {
            setIsInterpreting(false);
        }
    }, [result]);

    // --- Formátovací funkce ---
    const formatPVal = (p: number | null | undefined): string => {
        if (p === null || p === undefined || isNaN(p)) return "-";
        if (p < 0.001) return "< 0.001";
        return p.toFixed(3);
    };
    const formatStat = (s: number | null | undefined): string => {
        if (s === null || s === undefined || isNaN(s)) return "-";
        if (Math.abs(s) >= 100) return s.toFixed(1);
        if (Math.abs(s) >= 10) return s.toFixed(2);
        return s.toFixed(3);
    };

    // --- JSX (beze změny oproti předchozí verzi) ---
    return (
        <div className="space-y-6 p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
            {/* Fieldset */}
            <fieldset disabled={isLoading || isInterpreting || isFetchingCols} className="space-y-4">
                <legend className="text-lg font-semibold text-gray-800 dark:text-gray-100">Test závislosti / Porovnání skupin</legend>
                {/* Výběr sloupců */}
                <div>
                    <p className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2">1. Vyberte proměnné (min. 2):</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Kategoriální */}
                        <div>
                            <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-1 uppercase">Kategoriální</p>
                            <div className="space-y-1 max-h-36 overflow-y-auto border rounded p-2 dark:border-gray-600 bg-gray-50 dark:bg-gray-700/50">
                                {isFetchingCols && <p className="text-xs text-gray-500 italic p-1">Načítám...</p>}
                                {!isFetchingCols && categoricalCols.length === 0 && <p className="text-xs text-gray-500 italic p-1">Žádné</p>}
                                {categoricalCols.map(col => (
                                    <label key={col.name} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600 p-1 rounded cursor-pointer">
                                        <input type="checkbox" checked={selectedCats.includes(col.name)} onChange={e => handleCatCheckboxChange(col.name, e.target.checked)} className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800" />
                                        {col.name}
                                    </label>
                                ))}
                            </div>
                        </div>
                        {/* Numerické */}
                        <div>
                            <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-1 uppercase">Číselné</p>
                            <div className="space-y-1 max-h-36 overflow-y-auto border rounded p-2 dark:border-gray-600 bg-gray-50 dark:bg-gray-700/50">
                                {isFetchingCols && <p className="text-xs text-gray-500 italic p-1">Načítám...</p>}
                                {!isFetchingCols && numericCols.length === 0 && <p className="text-xs text-gray-500 italic p-1">Žádné</p>}
                                {numericCols.map(col => (
                                    <label key={col.name} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600 p-1 rounded cursor-pointer">
                                        <input type="checkbox" checked={selectedNums.includes(col.name)} onChange={e => handleNumCheckboxChange(col.name, e.target.checked)} className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800" />
                                        {col.name}
                                    </label>
                                ))}
                            </div>
                        </div>
                    </div>
                    {/* Zobrazení chyby načítání sloupců */}
                    {error && !result && !isLoading && !isInterpreting && !isFetchingCols && (
                        <div role="alert" className="mt-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                            ⚠️ {error}
                        </div>
                    )}
                </div>
                {/* Výběr metody */}
                <div>
                    <label htmlFor='dep-method' className="block text-sm font-medium text-gray-800 dark:text-gray-200 mb-1">2. Zvolený test:</label>
                    <select id='dep-method' value={method} onChange={(e) => handleMethodChange(e.target.value)} className="mt-1 border border-gray-300 rounded px-3 py-2 text-sm w-full max-w-xs dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 focus:ring-blue-500 focus:border-blue-500">
                        <option value="auto">Automaticky (doporučeno)</option>
                        <option value="chi2">Chi-squared (χ²) (2+ Kat.)</option>
                        <option value="fisher">Fisherův přesný test (2 Kat., 2x2)</option>
                        <option value="anova">ANOVA (1+ Kat. vs 1+ Num.)</option>
                        <option value="kruskal">Kruskal-Wallis (1+ Kat. vs 1+ Num., nepar.)</option>
                        <option value="t.test">t-test (2 Num. nebo 1Kat(L=2)+1Num)</option>
                        <option value="wilcoxon">Wilcoxon (2 Num., párový)</option>
                        <option value="mannwhitney">Mann-Whitney U (2 Num. nepár., nebo 1Kat(L=2)+1Num)</option>
                    </select>
                </div>
                {/* Párová data */}
                <div className="mt-2">
                    <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 cursor-pointer">
                        <input type="checkbox" checked={paired} onChange={(e) => handlePairedChange(e.target.checked)} className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800" />
                        Párová data <span className="text-xs text-gray-500">(relevantní pro t-test, Wilcoxon)</span>
                    </label>
                </div>
            </fieldset>

            {/* Tlačítko Spustit */}
            <button onClick={runTest} disabled={allSelectedCols.length < 2 || isLoading || isInterpreting || isFetchingCols} className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out">
                {isLoading && !isInterpreting ? (<> <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg> Analyzuji... </>) : "Spustit test"}
            </button>

            {/* Zobrazení chyby analýzy */}
            {error && !isLoading && (
                <div role="alert" className="mt-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                    ⚠️ **Chyba:** {error}
                </div>
            )}

            {/* Zobrazení výsledků */}
            {result && !isLoading && (
                <div className="space-y-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                    {/* Shrnutí testu */}
                    <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded border border-gray-200 dark:border-gray-600 text-sm">
                        {/* Zde používáme result.test, protože jsme ho namapovali */}
                        <p><strong className="font-medium text-gray-800 dark:text-gray-200">Použitý test:</strong> <span className="font-semibold text-indigo-700 dark:text-indigo-400">{result.test || 'Neznámý'}</span></p>
                        {result.reason && <p className="text-gray-600 dark:text-gray-400 mt-1 text-xs italic">{result.reason}</p>}
                        {result.warning_message && <p className="text-orange-700 dark:text-orange-400 mt-1 text-xs">⚠️ Varování: {result.warning_message}</p>}

                        {/* Zobrazení hlavní p-hodnoty/statistiky */}
                        {(typeof result.p_value === "number" || typeof result.statistic === "number") && (!result.results || result.results.length === 0) && (
                            <div className='mt-2 pt-2 border-t border-gray-200 dark:border-gray-600 flex flex-wrap gap-x-4 gap-y-1'>
                                {typeof result.p_value === "number" && (
                                    <p><strong>p-hodnota:</strong>
                                        <span className={`ml-1 font-mono ${(result.p_value ?? 1) < 0.05 ? 'text-green-700 dark:text-green-400 font-bold' : 'text-gray-700 dark:text-gray-300'}`}>{formatPVal(result.p_value)}</span>
                                        {(result.p_value ?? 1) < 0.05 && <span className='ml-1 text-xs'>(významné)</span>}
                                    </p>
                                )}
                                {typeof result.statistic === "number" && ( <p><strong>{result.statistic_name || 'Test. stat.'}:</strong> <span className='ml-1 font-mono'>{formatStat(result.statistic)}</span></p> )}
                                {result.degrees_freedom !== null && result.degrees_freedom !== undefined && ( <p><strong>St. volnosti:</strong> <span className='ml-1 font-mono'>{String(result.degrees_freedom)}</span></p> )}
                            </div>
                        )}
                    </div>

                    {/* Výsledky pro ANOVU/Kruskal */}
                    {result.results && result.results.length > 0 && (
                        <div>
                            <h4 className="text-base font-semibold mb-2 text-gray-800 dark:text-gray-100">Výsledky pro kombinace proměnných:</h4>
                            <div className="overflow-x-auto border border-gray-200 dark:border-gray-600 rounded-md shadow-sm">
                                <table className="min-w-full text-sm divide-y divide-gray-200 dark:divide-gray-700">
                                    <thead className="bg-gray-100 dark:bg-gray-700">
                                    <tr>
                                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Kategorie</th>
                                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">Číselná</th>
                                        {/* Zde bereme název statistiky z prvního výsledku, pokud existuje, jinak fallback */}
                                        <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">{result.statistic_name || (result.test === 'ANOVA' ? 'Statistika (F)' : 'Statistika (H)')}</th>
                                        <th scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">p-hodnota</th>
                                    </tr>
                                    </thead>
                                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                                    {result.results.map((r, idx) => (
                                        <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                            <td className="px-3 py-2 text-gray-800 dark:text-gray-200">{r.cat_col}</td>
                                            <td className="px-3 py-2 text-gray-800 dark:text-gray-200">{r.num_col}</td>
                                            <td className="px-3 py-2 text-center font-mono text-gray-700 dark:text-gray-300">{formatStat(r.statistic)}</td>
                                            <td className={`px-3 py-2 text-center font-mono ${(r.p_value ?? 1) < 0.05 ? 'font-bold text-green-700 dark:text-green-400' : 'text-gray-600 dark:text-gray-400'}`}>{formatPVal(r.p_value)}</td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Kontingenční tabulka */}
                    {result.contingency_table && (
                        <div>
                            <h4 className="text-base font-semibold mb-2 text-gray-800 dark:text-gray-100">Kontingenční tabulka:</h4>
                            <div className="overflow-x-auto border border-gray-200 dark:border-gray-600 rounded-md shadow-sm">
                                <table className="min-w-full text-sm divide-y divide-gray-200 dark:divide-gray-700">
                                    <thead className="bg-gray-100 dark:bg-gray-700">
                                    {(() => {
                                        const tableData = result.contingency_table!;
                                        const rowLabels = Object.keys(tableData);
                                        if (rowLabels.length === 0) return null;
                                        const firstRowKey = rowLabels[0];
                                        const colLabels = Object.keys(tableData[firstRowKey]);
                                        let colNames = ['Proměnná 1', 'Proměnná 2'];
                                        if (result.columns && result.columns.length === 2) {
                                            colNames = result.columns;
                                        }
                                        return (
                                            <tr>
                                                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">{colNames[0]} \ {colNames[1]}</th>
                                                {colLabels.map((label, idx) => (
                                                    <th key={idx} scope="col" className="px-3 py-2 text-center text-xs font-medium text-gray-600 dark:text-gray-300 uppercase tracking-wider">{label}</th>
                                                ))}
                                            </tr>
                                        );
                                    })()}
                                    </thead>
                                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                                    {Object.entries(result.contingency_table).map(([rowLabel, rowValues], idx) => (
                                        <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                            <td className="px-3 py-2 font-medium text-gray-800 dark:text-gray-200">{rowLabel}</td>
                                            {Object.values(rowValues).map((val, i) => (
                                                <td key={i} className="px-3 py-2 text-center text-gray-700 dark:text-gray-300">{val}</td>
                                            ))}
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Sekce AI Interpretace */}
                    <div className="pt-6 border-t border-dashed border-gray-300 dark:border-gray-600">
                        <h4 className="text-base font-semibold mb-3 text-gray-800 dark:text-gray-100">Interpretace pomocí AI</h4>
                        {/* Tlačítko */}
                        {!aiInterpretation && !isInterpreting && !aiError && (
                            <button onClick={handleInterpret} disabled={isInterpreting || isLoading} className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition duration-150 ease-in-out">
                                💡 Interpretovat výsledek pomocí AI
                            </button>
                        )}
                        {/* Loading AI */}
                        {isInterpreting && (
                            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 p-2 rounded bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600">
                                <svg className="animate-spin h-4 w-4 text-indigo-600 dark:text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> {/* ... ikona ... */} </svg>
                                AI generuje interpretaci...
                            </div>
                        )}
                        {/* Chyba AI */}
                        {aiError && !isInterpreting && (
                            <div role="alert" className="mt-3 p-3 bg-red-100 border border-red-300 text-red-700 rounded text-sm dark:bg-red-900/30 dark:text-red-200 dark:border-red-700">
                                ⚠️ Chyba interpretace: {aiError}
                                <button onClick={handleInterpret} disabled={isInterpreting || isLoading} className="ml-4 text-xs font-medium text-red-800 dark:text-red-300 underline hover:text-red-900 dark:hover:text-red-200 disabled:opacity-50 disabled:cursor-not-allowed">
                                    Zkusit znovu
                                </button>
                            </div>
                        )}
                        {/* Výsledek AI */}
                        {aiInterpretation && !isInterpreting && (
                            <div className="mt-3 p-4 bg-gray-100 dark:bg-gray-700/60 rounded border border-gray-200 dark:border-gray-600">
                                <p className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">{aiInterpretation}</p>
                                <button onClick={() => { setAiInterpretation(null); setAiError(null); }} disabled={isInterpreting || isLoading} className="mt-3 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline disabled:opacity-50 disabled:cursor-not-allowed">
                                    Skrýt / Generovat novou
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Explicitní Loading indikátor pro hlavní test */}
            {isLoading && (
                <div className="mt-4 flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                    <svg className="animate-spin h-4 w-4 text-blue-600 dark:text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                    Probíhá analýza...
                </div>
            )}
        </div>
    );
}