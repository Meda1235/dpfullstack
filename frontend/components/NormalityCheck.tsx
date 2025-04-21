// components/NormalityCheck.tsx
import React, { useEffect, useState, useCallback } from 'react';
import SidebarNav from '@/components/SidebarNav'; // Předpokládáme existenci

// Interface odpovídá backend modelu NormalityResultModel
interface NormalityResult {
    column: string;
    test: string;
    pValue: number;
    isNormal: boolean;
    warning: string; // Varování bude vždy string, může být "-"
    hasMissing: boolean;
}

// Dostupné testovací metody pro výběr uživatelem
const TEST_METHODS = [
    { value: 'shapiro', label: 'Shapiro-Wilk' },
    { value: 'ks', label: 'Kolmogorov-Smirnov' },
];

// Pomocná funkce pro formátování p-hodnoty
const formatPValue = (pValue: number | undefined | null): string => {
    if (typeof pValue !== 'number' || isNaN(pValue)) {
        return '-'; // Explicitní ošetření NaN
    }
    // Použití exponenciálního zápisu pro velmi malé hodnoty (ale ne pro nulu)
    if (pValue < 0.0001 && pValue > 0) {
        return pValue.toExponential(2);
    }
    if (pValue === 0) {
        return "0.0000"; // Nebo třeba "< 0.0001"
    }
    // Standardní formátování na 4 desetinná místa
    return pValue.toFixed(4);
};

export default function NormalityCheck() {
    // --- Základní stav komponenty ---
    const [results, setResults] = useState<NormalityResult[]>([]); // Pole výsledků
    const [loading, setLoading] = useState(true); // Stav načítání celé stránky
    const [error, setError] = useState<string | null>(null); // Globální chyba stránky
    const [successMessage, setSuccessMessage] = useState<string | null>(null); // Globální zpráva o úspěchu (např. po transformaci)

    // --- Stav pro PŘEPOČET testu pro jednotlivé sloupce ---
    const [selectedTestOverrides, setSelectedTestOverrides] = useState<{ [column: string]: string }>({}); // Jaký test uživatel vybral pro sloupec
    const [recalculatingTestColumn, setRecalculatingTestColumn] = useState<string | null>(null); // Který sloupec se právě přepočítává
    const [recalculateError, setRecalculateError] = useState<{ [column: string]: string | null }>({}); // Chyby specifické pro přepočet

    // --- Stav pro TRANSFORMACI sloupců ---
    const [selectedTransforms, setSelectedTransforms] = useState<{ [column: string]: string }>({}); // Jaká transformace je vybrána pro sloupec
    const [transformingColumn, setTransformingColumn] = useState<string | null>(null); // Který sloupec se právě transformuje
    const [transformError, setTransformError] = useState<{ [column: string]: string | null }>({}); // Chyby specifické pro transformaci

    // --- Funkce pro načtení POČÁTEČNÍCH výsledků normality (používá heuristiku) ---
    const fetchInitialNormalityResults = useCallback(async () => {
        console.log("Fetching initial normality results...");
        setLoading(true);
        setError(null);
        setSuccessMessage(null); // Vyčistit staré zprávy
        setResults([]); // Vyčistit staré výsledky
        // Resetovat stavy specifické pro řádky
        setSelectedTestOverrides({});
        setRecalculateError({});
        setSelectedTransforms({});
        setTransformError({});
        setRecalculatingTestColumn(null);
        setTransformingColumn(null);

        try {
            // Volání endpointu, který vrací výsledky dle heuristiky
            const response = await fetch('http://localhost:8000/api/check_normality'); // GET request
            if (!response.ok) {
                let errorData = 'Chyba serveru';
                try {
                    // Pokusíme se získat detail chyby z FastAPI
                    errorData = (await response.json()).detail || await response.text();
                } catch (e) {
                    errorData = `Chyba serveru: ${response.status} ${response.statusText}`;
                }
                throw new Error(errorData);
            }
            const data = await response.json();
            // Zajistíme, že results je pole a případně odfiltrujeme řádky s chybou z backendu
            const validResults = (data.results || []).filter((r: NormalityResult) => r.test !== "Chyba" && r.test !== ""); // Filtrujeme i prázdné testy pro jistotu
            const errorResults = (data.results || []).filter((r: NormalityResult) => r.test === "Chyba" || r.test === "");

            setResults(validResults);

            // Pokud backend vrátil nějaké sloupce s chybou, zobrazíme varování
            if (errorResults.length > 0) {
                const errorCols = errorResults.map((r: NormalityResult) => r.column).join(', ');
                // Přidáme varování k případné existující chybě
                setError(prev => {
                    const newError = `Chyba při zpracování sloupců: ${errorCols}. Zkontrolujte konzoli serveru pro detaily.`;
                    return prev ? `${prev}; ${newError}` : newError;
                });
            }
        } catch (err: any) {
            console.error('Chyba při načítání počáteční normality:', err);
            setError(`Nepodařilo se načíst výsledky normality: ${err.message}`);
            setResults([]); // Vyčistit výsledky v případě chyby
        } finally {
            setLoading(false); // Ukončit načítání
        }
    }, []); // Prázdné pole závislostí = funkce se nemění

    // --- Načtení dat při prvním zobrazení komponenty ---
    useEffect(() => {
        fetchInitialNormalityResults();
    }, [fetchInitialNormalityResults]); // Závislost na stabilní referenci funkce

    // --- Zpracování změny výběru TESTU v dropdownu ---
    const handleTestOverrideChange = (column: string, testMethod: string) => {
        setSelectedTestOverrides(prev => ({ ...prev, [column]: testMethod }));
        // Vyčistit případnou předchozí chybu pro tento řádek při změně výběru
        setRecalculateError(prev => ({ ...prev, [column]: null }));
    };

    // --- Přepočet normality pro JEDEN sloupec pomocí separátního endpointu ---
    const recalculateSingleTest = async (column: string) => {
        const testMethod = selectedTestOverrides[column];
        if (!testMethod) {
            console.warn("Pokus o přepočet bez vybraného testu pro sloupec:", column);
            setRecalculateError(prev => ({ ...prev, [column]: "Nejprve vyberte test." }));
            return;
        }

        setRecalculatingTestColumn(column); // Indikátor načítání pro tento řádek
        setRecalculateError(prev => ({ ...prev, [column]: null })); // Vyčistit starou chybu
        setError(null); // Vyčistit globální chybu

        console.log(`Recalculating normality for column "${column}" using test "${testMethod}"`);

        try {
            // Volání NOVÉHO separátního endpointu
            const response = await fetch('http://localhost:8000/api/recalculate_single_normality', {
                method: 'POST', // Metoda POST
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ column: column, test_method: testMethod }), // Data v těle požadavku
            });

            const responseData = await response.json(); // Očekáváme JSON odpověď

            if (!response.ok) {
                // Zpracování chyby z FastAPI (očekáváme 'detail')
                throw new Error(responseData.detail || `Chyba ${response.status}: ${response.statusText}`);
            }

            // ÚSPĚCH: Aktualizace stavu - nahradíme data pouze pro tento jeden sloupec
            setResults(prevResults =>
                prevResults.map(res =>
                    res.column === column
                        ? { ...responseData } // Použijeme kompletní nová data z odpovědi
                        : res // Ostatní řádky zůstanou nezměněné
                )
            );
            console.log(`Successfully recalculated normality for column "${column}"`);

            // Volitelně: Resetovat výběr v dropdownu po úspěšném přepočtu
            // setSelectedTestOverrides(prev => ({...prev, [column]: ''}));

        } catch (err: any) {
            console.error(`Chyba při přepočtu testu pro sloupec ${column}:`, err);
            // Zobrazit chybu specifickou pro tento řádek
            setRecalculateError(prev => ({ ...prev, [column]: `Přepočet selhal: ${err.message}` }));
        } finally {
            setRecalculatingTestColumn(null); // Ukončit indikátor načítání pro tento řádek
        }
    };


    // --- Zpracování změny výběru TRANSFORMACE ---
    const handleTransformChange = (column: string, method: string) => {
        setSelectedTransforms(prev => ({ ...prev, [column]: method }));
        setTransformError(prev => ({ ...prev, [column]: null })); // Vyčistit chybu transformace při změně
    };

    // --- Aplikace TRANSFORMACE na sloupec ---
    const applyTransformation = async (column: string) => {
        const method = selectedTransforms[column];
        if (!method) return; // Nemělo by nastat, pokud je tlačítko správně (dis)abled

        setTransformingColumn(column); // Indikátor načítání pro transformaci
        setTransformError(prev => ({ ...prev, [column]: null })); // Vyčistit starou chybu
        setSuccessMessage(null); // Vyčistit globální zprávu
        setError(null); // Vyčistit globální chybu

        console.log(`Applying transformation "${method}" to column "${column}"`);

        try {
            // Volání endpointu pro transformaci (předpokládáme, že existuje)
            const response = await fetch('http://localhost:8000/api/transform_column', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ column, method }),
            });

            const responseData = await response.json(); // Očekáváme JSON

            if (!response.ok) {
                throw new Error(responseData.detail || `Chyba ${response.status}: ${response.statusText}`);
            }

            // ÚSPĚCH transformace
            setSuccessMessage(`Sloupec "${column}" byl úspěšně transformován metodou "${method}". Aktualizuji data...`);
            console.log(`Transformation successful for column "${column}". Refetching all results...`);

            // PO TRANSFORMACI MUSÍME ZNOVU NAČÍST VŠECHNA DATA, PROTOŽE SE ZMĚNIL ZÁKLADNÍ DATASET
            // Použijeme funkci, která načítá počáteční stav (s heuristikou)
            await fetchInitialNormalityResults(); // Toto resetuje i všechny výběry a chyby

            // Není už třeba resetovat selectedTransforms zde, protože fetchInitialNormalityResults to dělá

            // Zpráva o úspěchu zmizí automaticky při přenačtení, nebo můžeme dát timeout
            // setTimeout(() => setSuccessMessage(null), 5000); // Pokud by fetch trval dlouho

        } catch (err: any) {
            console.error(`Chyba při transformaci sloupce ${column}:`, err);
            setTransformError(prev => ({ ...prev, [column]: `Transformace selhala: ${err.message}` }));
        } finally {
            setTransformingColumn(null); // Ukončit indikátor transformace
        }
    };

    // --- Renderovací logika ---

    // Vykreslení ovládacích prvků pro VÝBĚR a PŘEPOČET TESTU
    const renderTestSelectionControls = (res: NormalityResult) => {
        const isRecalculatingThis = recalculatingTestColumn === res.column;
        const selectedValue = selectedTestOverrides[res.column] || ''; // Aktuálně vybraná hodnota v dropdownu
        // Zjistíme kód ('shapiro'/'ks') aktuálně použitého testu pro srovnání
        const currentTestValue = res.test.toLowerCase().includes('shapiro') ? 'shapiro' : (res.test.toLowerCase().includes('kolmogorov') ? 'ks' : '');
        // Povolit tlačítko "Přepočítat" pouze pokud je něco vybráno v dropdownu
        const canRecalculate = !!selectedValue;

        return (
            <div className="mt-1 space-y-1 text-xs">
                {/* Popisek pro screen readery */}
                <label htmlFor={`test-select-${res.column}`} className="sr-only">
                    Změnit test normality pro sloupec {res.column}
                </label>
                <div className="flex gap-2 items-center flex-wrap"> {/* flex-wrap pro menší obrazovky */}
                    <select
                        id={`test-select-${res.column}`}
                        className="border rounded px-2 py-0.5 text-xs bg-white dark:bg-gray-700 dark:border-gray-600 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
                        value={selectedValue}
                        onChange={(e) => handleTestOverrideChange(res.column, e.target.value)}
                        disabled={isRecalculatingThis} // Zakázat během přepočtu
                        aria-label={`Vybrat nový test normality pro ${res.column}`}
                    >
                        <option value="" disabled>-- Změnit test --</option>
                        {TEST_METHODS.map(opt => (
                            <option key={opt.value} value={opt.value}>
                                {opt.label}
                                {/* Označit aktuálně použitý test pro informaci */}
                                {opt.value === currentTestValue ? ' (aktuální)' : ''}
                            </option>
                        ))}
                    </select>
                    <button
                        onClick={() => recalculateSingleTest(res.column)}
                        className={`bg-gray-500 hover:bg-gray-600 text-white px-2 py-0.5 rounded text-xs font-medium transition duration-150 ease-in-out focus:outline-none focus:ring-1 focus:ring-offset-1 focus:ring-gray-400 disabled:opacity-50 disabled:cursor-not-allowed`}
                        // Zakázat, pokud nic není vybráno nebo pokud se právě přepočítává
                        disabled={!canRecalculate || isRecalculatingThis}
                        title={!canRecalculate ? "Nejprve vyberte test z nabídky" : `Přepočítat normalitu pro ${res.column} pomocí ${TEST_METHODS.find(t=>t.value===selectedValue)?.label || 'vybraného testu'}`}
                    >
                        {isRecalculatingThis ? (
                            // Ikona načítání
                            <svg className="animate-spin h-3 w-3 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        ) : (
                            'Přepočítat' // Text tlačítka
                        )}
                    </button>
                </div>
                {/* Zobrazení chyby specifické pro přepočet tohoto řádku */}
                {recalculateError[res.column] && (
                    <p className="text-xs text-red-600 dark:text-red-400 mt-1">{recalculateError[res.column]}</p>
                )}
            </div>
        );
    }

    // Vykreslení ovládacích prvků pro TRANSFORMACI (pokud data nejsou normální)
    const renderTransformationControls = (res: NormalityResult) => {
        // Pokud jsou data normální, zobrazíme jen informaci
        if (res.isNormal) {
            return <span className="text-xs text-gray-500 italic dark:text-gray-400">Data jsou normálně distribuovaná.</span>;
        }

        const isTransformingThis = transformingColumn === res.column; // Probíhá transformace tohoto sloupce?

        // Pokud data nejsou normální, zobrazíme možnost transformace
        return (
            <div className="space-y-2">
                <label htmlFor={`transform-${res.column}`} className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                    Data nejsou normální. Zkusit transformaci:
                </label>
                <div className="flex gap-2 items-center flex-wrap"> {/* flex-wrap pro menší obrazovky */}
                    <select
                        id={`transform-${res.column}`}
                        className="border rounded px-2 py-1 text-xs bg-white dark:bg-gray-700 dark:border-gray-600 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:bg-gray-100"
                        value={selectedTransforms[res.column] || ''}
                        onChange={(e) => handleTransformChange(res.column, e.target.value)}
                        disabled={isTransformingThis} // Zakázat během transformace
                        aria-label={`Vybrat metodu transformace pro ${res.column}`}
                    >
                        <option value="">-- vyber metodu --</option>
                        <option value="log">Logaritmická (log)</option>
                        <option value="sqrt">Odmocnina (sqrt)</option>
                        <option value="boxcox">Box-Cox</option>
                        {/* Zde můžete přidat další relevantní transformace */}
                    </select>
                    <button
                        onClick={() => applyTransformation(res.column)}
                        className={`bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-xs font-medium transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-blue-400`}
                        // Zakázat, pokud není vybrána metoda nebo probíhá transformace
                        disabled={!selectedTransforms[res.column] || isTransformingThis}
                        title={!selectedTransforms[res.column] ? "Nejprve vyberte metodu transformace" : `Aplikovat transformaci ${selectedTransforms[res.column]} na sloupec ${res.column}`}
                    >
                        {isTransformingThis ? (
                            // Ikona načítání
                            <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        ) : (
                            'Transformovat' // Text tlačítka
                        )}
                    </button>
                </div>
                {/* Zobrazení chyby specifické pro transformaci tohoto řádku */}
                {transformError[res.column] && (
                    <p className="text-xs text-red-600 dark:text-red-400 mt-1">{transformError[res.column]}</p>
                )}
            </div>
        );
    };

    // --- Pomocné proměnné pro render ---
    const hasMissingValues = !loading && results.some(r => r.hasMissing);
    const missingValueColumns = results.filter(r => r.hasMissing).map(r => r.column).join(', ');

    // --- JSX Struktura komponenty ---
    return (
        <div className="flex min-h-screen bg-gray-100 dark:bg-gray-900">
            <SidebarNav /> {/* Váš navigační panel */}
            <main className="flex-1 p-4 md:p-6 space-y-6"> {/* Přidáno responzivní padding */}
                <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">📊 Test normality</h1>

                {/* --- Oblast pro zprávy (úspěch, chyba, varování) --- */}
                {successMessage && (
                    <div className="p-3 bg-green-100 border border-green-300 text-green-800 rounded-md text-sm shadow-sm dark:bg-green-900/30 dark:border-green-700 dark:text-green-200" role="alert">
                        ✅ {successMessage}
                    </div>
                )}
                {error && ( // Globální chyba stránky
                    <div className="p-3 bg-red-100 border border-red-300 text-red-800 rounded-md text-sm shadow-sm dark:bg-red-900/30 dark:border-red-700 dark:text-red-200" role="alert">
                        ❌ {error}
                    </div>
                )}
                {hasMissingValues && !loading && ( // Zobrazit jen pokud nejsou data načítána
                    <div className="p-3 bg-yellow-100 border border-yellow-300 text-yellow-800 rounded-md text-sm shadow-sm dark:bg-yellow-900/30 dark:border-yellow-700 dark:text-yellow-200" role="alert">
                        ⚠️ <strong>Pozor:</strong> Sloupce ({missingValueColumns}) obsahují chybějící hodnoty. Testy normality byly provedeny bez těchto hodnot. Zvažte jejich doplnění v sekci "Chybějící hodnoty".
                    </div>
                )}

                <p className="text-gray-600 dark:text-gray-400 text-sm">
                    Zde vidíte výsledky testování normality pro číselné sloupce. Výchozí test je vybrán automaticky podle počtu hodnot (méně než 50: Shapiro-Wilk, N≥50: Kolmogorov-Smirnov). Můžete vybrat jiný test a přepočítat výsledek pro konkrétní sloupec. Pokud data nejsou normálně distribuovaná (p ≤ 0.05), můžete zkusit aplikovat transformaci.
                </p>

                {/* --- Indikátor načítání --- */}
                {loading && (
                    <div className="flex items-center justify-center p-10 text-gray-500 dark:text-gray-400">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        <span>Načítání výsledků normality...</span>
                    </div>
                )}

                {/* --- Zpráva pokud nejsou data/výsledky --- */}
                {!loading && !error && results.length === 0 && (
                    <p className="text-center text-gray-500 dark:text-gray-400 py-8">
                        Nebyly nalezeny žádné vhodné číselné sloupce pro testování normality, nebo váš dataset ještě nebyl nahrán. Zkontrolujte, zda data obsahují alespoň 3 platné číselné hodnoty ve sloupcích.
                    </p>
                )}

                {/* --- Tabulka výsledků --- */}
                {!loading && results.length > 0 && (
                    <div className="overflow-x-auto shadow-md rounded-lg border border-gray-200 dark:border-gray-700">
                        <table className="w-full table-auto text-sm text-left text-gray-700 dark:text-gray-300">
                            <thead className="text-xs text-gray-700 uppercase bg-gray-100 dark:bg-gray-800 dark:text-gray-400">
                            <tr>
                                <th scope="col" className="px-4 py-3 whitespace-nowrap">Sloupec</th>
                                <th scope="col" className="px-4 py-3 whitespace-nowrap">Použitý test / Změnit</th> {/* Upravený nadpis */}
                                <th scope="col" className="px-4 py-3 whitespace-nowrap">p-hodnota</th>
                                <th scope="col" className="px-4 py-3 whitespace-nowrap">Normální? (α=0.05)</th>
                                <th scope="col" className="px-4 py-3">Poznámka / Transformace</th> {/* Může být delší */}
                            </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                            {results.map((res) => (
                                <tr key={res.column} className={`hover:bg-gray-50 dark:hover:bg-gray-700/50 ${res.isNormal ? 'bg-white dark:bg-gray-800' : 'bg-red-50 dark:bg-red-900/20'}`}>

                                    {/* Název sloupce */}
                                    <td className="px-4 py-3 font-medium text-gray-900 dark:text-white whitespace-nowrap">{res.column}</td>

                                    {/* Použitý test a ovládání pro změnu */}
                                    <td className="px-4 py-3 align-top">
                                        <div className='font-medium'>{res.test || '-'}</div> {/* Zobrazení aktuálně použitého testu */}
                                        {/* Ovládací prvky pro změnu testu */}
                                        {renderTestSelectionControls(res)}
                                    </td>

                                    {/* p-hodnota */}
                                    <td className={`px-4 py-3 font-mono align-top ${res.isNormal ? '' : 'font-semibold text-red-700 dark:text-red-400'}`}>
                                        {formatPValue(res.pValue)}
                                    </td>

                                    {/* Výsledek (Normální Ano/Ne) */}
                                    <td className="px-4 py-3 align-top">
                                        {res.isNormal
                                            ? <span className="text-green-600 dark:text-green-400 font-medium">✅ Ano</span>
                                            : <span className="text-red-600 dark:text-red-400 font-medium">❌ Ne</span>
                                        }
                                    </td>

                                    {/* Poznámky a ovládání transformace */}
                                    <td className="px-4 py-3 text-xs text-gray-500 dark:text-gray-400 align-top space-y-2">
                                        {/* Zobrazit varování z backendu, pokud existuje a není jen "-" */}
                                        {res.warning && res.warning !== "-" && <div className="mb-2 border-l-2 border-yellow-400 pl-2">{res.warning}</div>}
                                        {/* Ovládací prvky pro transformaci (zobrazí se jen pokud není normální) */}
                                        {renderTransformationControls(res)}
                                    </td>
                                </tr>
                            ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </main>
        </div>
    );
}