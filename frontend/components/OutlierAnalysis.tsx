import React, { useEffect, useState, useCallback } from 'react'; // Přidán useCallback
import SidebarNav from './SidebarNav'; // Ujisti se, že cesta je správná
import Plot from 'react-plotly.js';
import { toast } from 'react-toastify'; // Přidáno pro notifikace

interface OutlierInfo {
    column: string;
    count: number; // Celkový počet validních hodnot (původní význam byl počet outlierů Z>3)
    percent: number; // Procento outlierů Z>3 (původní význam)
    mean: number;
    std: number;
    values: number[]; // Všechny validní hodnoty z backendu
    outliers: number[]; // Pouze outliery s |Z| > 3 z backendu (pro informaci)
}

export default function OutlierAnalysis() {
    const [outlierInfo, setOutlierInfo] = useState<OutlierInfo[]>([]);
    const [selectedColumn, setSelectedColumn] = useState<string>('');
    const [lowerThreshold, setLowerThreshold] = useState<number>(-2); // Výchozí
    const [upperThreshold, setUpperThreshold] = useState<number>(2);  // Výchozí
    const [strategy, setStrategy] = useState<string>('replace_mean');
    const [applyToAll, setApplyToAll] = useState<boolean>(false);
    const [showOverview, setShowOverview] = useState(true);
    const [showBoxplot, setShowBoxplot] = useState(true);
    const [showDetails, setShowDetails] = useState(true);
    const [isLoading, setIsLoading] = useState<boolean>(true); // Načítání dat
    const [isProcessing, setIsProcessing] = useState<boolean>(false); // Zpracování outlierů
    const [error, setError] = useState<string | null>(null); // Chyba načítání
    const [customValue, setCustomValue] = useState<string>(''); // Pro vlastní hodnotu

    // --- Funkce pro načtení dat ---
    const fetchData = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('http://localhost:8000/api/get_outliers');
            if (!response.ok) {
                const errorData = await response.text();
                throw new Error(`Chyba serveru (${response.status}): ${errorData.substring(0,100)}`);
            }
            const data = await response.json();
            if (Array.isArray(data)) {
                setOutlierInfo(data);
                // Nastav výchozí vybraný sloupec, pokud existují data a žádný není vybrán
                if (data.length > 0 && !selectedColumn && data.some(d => d.column)) {
                    setSelectedColumn(data[0].column);
                } else if (data.length === 0) {
                    setSelectedColumn(''); // Žádná data, žádný vybraný sloupec
                }
                setError(null);
            } else {
                console.error("Neplatná odpověď z /get_outliers:", data);
                setError("Server vrátil neočekávaný formát dat.");
                setOutlierInfo([]);
                setSelectedColumn('');
            }
        } catch (err: any) {
            console.error("Fetch outliers error:", err);
            setError(err.message || "Nepodařilo se načíst data odlehlých hodnot.");
            setOutlierInfo([]);
            setSelectedColumn('');
        } finally {
            setIsLoading(false);
        }
    }, [selectedColumn]); // Znovu načti, pokud se změní vybraný sloupec (pro jistotu)

    useEffect(() => {
        fetchData();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Zavolá se jen při prvním načtení komponenty


    // --- Data pro graf a detailní tabulku vybraného sloupce ---
    const selectedData = React.useMemo(() => {
        const selected = outlierInfo.find(col => col.column === selectedColumn);
        if (!selected || !selected.values || selected.std === 0 || isNaN(selected.mean) || isNaN(selected.std)) {
            return { allData: [], yRegular: [], yOutliers: [], filteredOutliers: [] };
        }

        const lowerBound = selected.mean + lowerThreshold * selected.std;
        const upperBound = selected.mean + upperThreshold * selected.std;

        const allData = selected.values
            .filter(val => typeof val === 'number' && !isNaN(val)) // Jen validní čísla
            .map((val, index) => { // Použijeme index původního pole pro konzistenci? Nebo lokální index? Raději lokální.
                const z = (val - selected.mean) / selected.std;
                const isOutlier = z < lowerThreshold || z > upperThreshold;
                return { index, value: val, z, isOutlier };
            });

        const yRegular = allData.filter(d => !d.isOutlier).map(d => d.value);
        const yOutliers = allData.filter(d => d.isOutlier).map(d => d.value);
        const filteredOutliers = allData.filter(d => d.isOutlier); // Outliery podle sliderů

        return { allData, yRegular, yOutliers, filteredOutliers };
    }, [selectedColumn, outlierInfo, lowerThreshold, upperThreshold]);

    // --- Handler pro aplikaci strategie ---
    const handleApplyStrategy = useCallback(async () => {
        if (!selectedColumn && !applyToAll) {
            toast.warn("Vyberte sloupec nebo zaškrtněte 'Aplikovat na všechny'.");
            return;
        }
        if (strategy === 'replace_custom' && !applyToAll && customValue.trim() === '') {
            toast.warn('Zadejte konstantu pro nahrazení.');
            return;
        }
        if (strategy === 'replace_custom' && !applyToAll) {
            const numValue = parseFloat(customValue);
            if (isNaN(numValue)) {
                toast.error('Zadaná konstanta není platné číslo.');
                return;
            }
        }


        setIsProcessing(true);
        toast.info("Zpracovávám odlehlé hodnoty...");

        const body = {
            method: strategy,
            // Použijeme `columns` jako pole i pro jeden sloupec
            columns: applyToAll ? outlierInfo.map(o => o.column) : [selectedColumn],
            // Pošleme hranice Z-skóre a vlastní hodnotu, backend to může využít
            lower_z: lowerThreshold,
            upper_z: upperThreshold,
            custom_value: strategy === 'replace_custom' ? parseFloat(customValue) : undefined, // Pošleme číslo
        };

        try {
            const response = await fetch('http://localhost:8000/api/handle_outliers', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
            });

            if (!response.ok) {
                const errorData = await response.text(); // Zkusíme získat text chyby
                throw new Error(`Chyba serveru (${response.status}): ${errorData.substring(0,150)}`);
            }

            const result = await response.json();
            toast.success(`Odlehlé hodnoty zpracovány (${result?.status || 'OK'}). Načítám aktualizovaná data...`);
            // Znovu načteme data, aby se tabulky a grafy aktualizovaly
            await fetchData();

        } catch (err: any) {
            console.error("Handle outliers error:", err);
            toast.error(`Chyba při zpracování odlehlých hodnot: ${err.message}`);
        } finally {
            setIsProcessing(false);
        }
    }, [strategy, applyToAll, selectedColumn, outlierInfo, fetchData, lowerThreshold, upperThreshold, customValue]); // Přidány závislosti

    // Výpočet mediánu (pomocná funkce)
    const calculateMedian = (arr: number[]): number => {
        if (!arr || arr.length === 0) return NaN;
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
    };


    return (
        <div className="flex dark:bg-gray-900 dark:text-gray-100 min-h-screen">
            <SidebarNav/>
            <main className="flex-1 p-4 md:p-6 lg:p-8 space-y-6">
                <h1 className="text-2xl font-bold">Analýza odlehlých hodnot</h1>

                {/* Načítání */}
                {isLoading && (
                    <div className="text-center py-10">
                        <svg className="animate-spin h-8 w-8 text-blue-600 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                        <p className="mt-2 text-gray-500 dark:text-gray-400">Načítám data...</p>
                    </div>
                )}

                {/* Chyba načítání */}
                {error && !isLoading && (
                    <div role="alert" className="p-4 bg-red-100 border border-red-400 text-red-700 rounded dark:bg-red-900/30 dark:border-red-700 dark:text-red-200">
                        <p className="font-bold">Chyba načítání dat!</p>
                        <p className="text-sm">{error}</p>
                        <button onClick={fetchData} className="mt-2 text-sm font-medium text-red-800 dark:text-red-300 underline hover:text-red-900 dark:hover:text-red-200">
                            Zkusit znovu načíst
                        </button>
                    </div>
                )}

                {/* Zobrazení, pokud nejsou data */}
                {!isLoading && !error && outlierInfo.length === 0 && (
                    <div className="p-4 text-center text-gray-500 dark:text-gray-400 border rounded dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
                        Nebyly nalezeny žádné numerické sloupce vhodné pro analýzu odlehlých hodnot nebo data ještě nebyla načtena.
                    </div>
                )}


                {/* Obsah se zobrazí, jen když máme data a není chyba */}
                {!isLoading && !error && outlierInfo.length > 0 && (
                    <>
                        {/* Slidery */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 border rounded dark:border-gray-700 bg-gray-50 dark:bg-gray-800 shadow-sm">
                            <div>
                                <label htmlFor="upperZ" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Horní práh Z-score:</label>
                                <div className="flex items-center gap-2">
                                    <input id="upperZ" type="range" min={0.1} max={5} step={0.1}
                                           value={upperThreshold} onChange={(e) => setUpperThreshold(Number(e.target.value))}
                                           className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
                                    <span className="text-sm font-semibold w-12 text-right">{upperThreshold.toFixed(1)}</span>
                                </div>
                            </div>
                            <div>
                                <label htmlFor="lowerZ" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Dolní práh Z-score:</label>
                                <div className="flex items-center gap-2">
                                    <input id="lowerZ" type="range" min={-5} max={-0.1} step={0.1}
                                           value={lowerThreshold} onChange={(e) => setLowerThreshold(Number(e.target.value))}
                                           className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
                                    <span className="text-sm font-semibold w-12 text-right">{lowerThreshold.toFixed(1)}</span>
                                </div>
                            </div>
                        </div>

                        {/* Přehledová tabulka všech sloupců */}
                        <div className="border rounded dark:border-gray-700 bg-white dark:bg-gray-800 shadow">
                            <h2
                                className="text-lg font-semibold p-3 cursor-pointer select-none hover:text-blue-600 dark:hover:text-blue-400 transition flex justify-between items-center border-b dark:border-gray-700"
                                onClick={() => setShowOverview(prev => !prev)}
                                aria-expanded={showOverview}
                            >
                                Přehled odlehlých hodnot ve všech sloupcích
                                <span className={`transform transition-transform duration-200 ${showOverview ? 'rotate-180' : 'rotate-0'}`}>▼</span>
                            </h2>

                            {showOverview && (
                                <div className="overflow-x-auto p-3 max-h-[400px] overflow-y-auto">
                                    <table className="w-full border-collapse text-sm">
                                        <thead className="sticky top-0 bg-gray-100 dark:bg-gray-700 z-10">
                                        <tr className="text-left text-xs">
                                            <th className="px-2 py-1 border dark:border-gray-600">Sloupec</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Počet Hodnot</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Průměr</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Směr. odch.</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Medián</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Dolní Hranice (Z={lowerThreshold.toFixed(1)})</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Horní Hranice (Z={upperThreshold.toFixed(1)})</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Odlehlé (celkem)</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">% Odlehlých</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Pod Hranicí</th>
                                            <th className="px-2 py-1 border dark:border-gray-600 text-right">Nad Hranicí</th>
                                        </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-200 dark:divide-gray-600">
                                        {outlierInfo.map((col) => {
                                            const values = col.values || [];
                                            const validValues = values.filter(v => typeof v === 'number' && !isNaN(v));

                                            if (validValues.length === 0 || col.std === 0 || isNaN(col.mean) || isNaN(col.std)) {
                                                return (
                                                    <tr key={col.column} className="bg-gray-50 dark:bg-gray-700/50 opacity-60">
                                                        <td className="px-2 py-1 border dark:border-gray-600">{col.column}</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-right">{values.length}</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                        <td className="px-2 py-1 border dark:border-gray-600 text-center">-</td>
                                                    </tr>
                                                );
                                            }

                                            const median = calculateMedian(validValues);
                                            const lowerBound = col.mean + lowerThreshold * col.std;
                                            const upperBound = col.mean + upperThreshold * col.std;

                                            let belowCount = 0;
                                            let aboveCount = 0;
                                            validValues.forEach(v => {
                                                if (v < lowerBound) belowCount++;
                                                else if (v > upperBound) aboveCount++;
                                            });
                                            const totalOutliersCount = belowCount + aboveCount;
                                            const percentOutliers = (totalOutliersCount / validValues.length) * 100;

                                            return (
                                                <tr
                                                    key={col.column}
                                                    className={`cursor-pointer hover:bg-blue-50 dark:hover:bg-blue-900/20 ${selectedColumn === col.column ? 'bg-blue-100 dark:bg-blue-900/40' : ''}`}
                                                    onClick={() => setSelectedColumn(col.column)}
                                                >
                                                    <td className="px-2 py-1 border dark:border-gray-600 font-medium">{col.column}</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{validValues.length}</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{col.mean.toFixed(2)}</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{col.std.toFixed(2)}</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{median.toFixed(2)}</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{lowerBound.toFixed(2)}</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{upperBound.toFixed(2)}</td>
                                                    <td className={`px-2 py-1 border dark:border-gray-600 text-right font-semibold ${totalOutliersCount > 0 ? 'text-red-600 dark:text-red-400' : ''}`}>{totalOutliersCount}</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{percentOutliers.toFixed(1)}%</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{belowCount}</td>
                                                    <td className="px-2 py-1 border dark:border-gray-600 text-right">{aboveCount}</td>
                                                </tr>
                                            );
                                        })}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>

                        {/* Výběr sloupce + strategie */}
                        <div className="flex flex-col sm:flex-row flex-wrap items-center gap-4 p-4 border rounded dark:border-gray-700 bg-gray-50 dark:bg-gray-800 shadow-sm">
                            <div className="flex-shrink-0">
                                <label htmlFor="colSelect" className="font-medium text-sm mr-2">Vybraný sloupec:</label>
                                <select
                                    id="colSelect"
                                    className="border border-gray-300 dark:border-gray-600 px-2 py-1 rounded text-sm dark:bg-gray-700 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500"
                                    value={selectedColumn}
                                    onChange={e => setSelectedColumn(e.target.value)}
                                    disabled={isProcessing}
                                >
                                    {outlierInfo.map(col => (
                                        <option key={col.column} value={col.column}>{col.column}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="flex-grow flex flex-col sm:flex-row flex-wrap items-center gap-3">
                                <div className="flex-shrink-0">
                                    <label htmlFor="strategySelect" className="font-medium text-sm mr-2">Strategie:</label>
                                    <select
                                        id="strategySelect"
                                        className="border border-gray-300 dark:border-gray-600 px-2 py-1 rounded text-sm dark:bg-gray-700 dark:text-gray-200 focus:ring-blue-500 focus:border-blue-500"
                                        value={strategy}
                                        onChange={(e) => setStrategy(e.target.value)}
                                        disabled={isProcessing}
                                    >
                                        <option value="remove">Odstranit řádky s outliery</option>
                                        <option value="replace_mean">Nahradit průměrem</option>
                                        <option value="replace_median">Nahradit mediánem</option>
                                        <option value="replace_custom">Nahradit konstantou</option>
                                        <option value="clip">Omezit na hranice (Winzorizace)</option>
                                    </select>
                                </div>

                                {strategy === 'replace_custom' && (
                                    <div className="flex-shrink-0">
                                        <label htmlFor="customValue" className="font-medium text-sm mr-2">Konstanta:</label>
                                        <input
                                            id="customValue"
                                            type="number"
                                            step="any"
                                            value={customValue}
                                            onChange={(e) => setCustomValue(e.target.value)}
                                            placeholder="Zadejte číslo"
                                            className="border border-gray-300 dark:border-gray-600 px-2 py-1 rounded text-sm dark:bg-gray-700 dark:text-gray-200 w-28 focus:ring-blue-500 focus:border-blue-500"
                                            disabled={isProcessing}
                                        />
                                    </div>
                                )}

                                <label className="ml-2 inline-flex items-center text-sm cursor-pointer flex-shrink-0">
                                    <input
                                        type="checkbox"
                                        className="mr-1 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-blue-600 dark:ring-offset-gray-800"
                                        checked={applyToAll}
                                        onChange={(e) => setApplyToAll(e.target.checked)}
                                        disabled={isProcessing}
                                    />
                                    Aplikovat na všechny sloupce
                                </label>

                                <button
                                    onClick={handleApplyStrategy}
                                    disabled={isProcessing || (!selectedColumn && !applyToAll) || (strategy === 'replace_custom' && !applyToAll && customValue.trim() === '')}
                                    className="ml-auto px-3 py-1.5 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                                >
                                    {isProcessing ? (
                                        <>
                                            <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
                                            Zpracovávám...
                                        </>
                                    ) : "Aplikovat Strategii"}
                                </button>
                            </div>
                        </div>

                        {/* Graf a detaily pro vybraný sloupec */}
                        {selectedColumn && outlierInfo.find(col => col.column === selectedColumn) && ( // Zobrazit jen pokud je vybrán validní sloupec
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                                {/* Box plot */}
                                <div className="border rounded dark:border-gray-700 bg-white dark:bg-gray-800 shadow">
                                    <h2
                                        className="text-lg font-semibold p-3 cursor-pointer select-none hover:text-blue-600 dark:hover:text-blue-400 transition flex justify-between items-center border-b dark:border-gray-700"
                                        onClick={() => setShowBoxplot(prev => !prev)}
                                        aria-expanded={showBoxplot}
                                    >
                                        Boxplot pro: {selectedColumn} (Outliery dle Z-skóre {lowerThreshold.toFixed(1)} / {upperThreshold.toFixed(1)})
                                        <span className={`transform transition-transform duration-200 ${showBoxplot ? 'rotate-180' : 'rotate-0'}`}>▼</span>
                                    </h2>
                                    {showBoxplot && (

                                            <div className="p-2">
                                                <Plot
                                                    data={[
                                                        { // Box trace
                                                            y: selectedData.yRegular,
                                                            // --- ZMĚNA: Přidáno explicitní X ---
                                                            x: Array(selectedData.yRegular.length).fill(selectedColumn),
                                                            type: 'box',
                                                            name: 'Normální hodnoty', // Legenda pro box
                                                            marker: {color: '#3b82f6'},
                                                            boxpoints: false, // Nezobrazovat defaultní outliery boxplotu
                                                        },
                                                        { // Scatter trace for outliers
                                                            // --- ZMĚNA: Použití stejného X jako boxplot ---
                                                            x: Array(selectedData.yOutliers.length).fill(selectedColumn),
                                                            y: selectedData.yOutliers,
                                                            type: 'scatter',
                                                            mode: 'markers',
                                                            name: 'Odlehlé hodnoty', // Legenda pro outliery
                                                            marker: {color: '#ef4444', size: 6, symbol: 'x'}, // Červené křížky
                                                        },
                                                    ]}
                                                    layout={{
                                                        autosize: true,
                                                        yaxis: {title: selectedColumn, zeroline: false, gridcolor: '#e5e7eb', gridwidth: 1 },
                                                        // --- ZMĚNA: Upravená osa X ---
                                                        xaxis: {
                                                            title: '', // Bez popisku osy X
                                                            zeroline: false,
                                                            showgrid: false, // Bez mřížky na X
                                                            // type: 'category', // Není nutné explicitně, plotly pozná
                                                            // Můžeme nechat tick label viditelný nebo ho skrýt
                                                            showticklabels: true // Zobrazí název sloupce pod boxem
                                                        },
                                                        margin: {t: 30, b: 40, l: 60, r: 20}, // Upraven dolní okraj pro tick label
                                                        showlegend: true, // Legenda je užitečná
                                                        legend: {orientation: "h", yanchor: "bottom", y: -0.25, xanchor: "center", x: 0.5}, // Legenda dole
                                                        // boxmode: 'group', // Není potřeba pro jeden box
                                                        paper_bgcolor: 'rgba(0,0,0,0)',
                                                        plot_bgcolor:'rgba(0,0,0,0)',
                                                        font: { color: document.documentElement.classList.contains('dark') ? '#d1d5db' : '#374151' }
                                                    }}
                                                    style={{width: '100%', height: '400px'}}
                                                    config={{responsive: true, displayModeBar: false}} // Skryjeme plotly tool bar
                                                />
                                                {/* Styly pro dark mode zůstávají stejné */}
                                                <style jsx global>{`
                                            .dark .js-plotly-plot .plotly .gridlayer .grid path { stroke: #4b5563 !important; }
                                            .dark .js-plotly-plot .plotly .zerolinelayer .zeroline { stroke: #6b7280 !important; }
                                            .dark .js-plotly-plot .plotly text { fill: #d1d5db !important; }
                                        `}</style>
                                            </div>
                                        )}

                                </div>

                                {/* Detailní tabulka outlierů */}
                                <div className="border rounded dark:border-gray-700 bg-white dark:bg-gray-800 shadow">
                                    <h2
                                        className="text-lg font-semibold p-3 cursor-pointer select-none hover:text-blue-600 dark:hover:text-blue-400 transition flex justify-between items-center border-b dark:border-gray-700"
                                        onClick={() => setShowDetails(prev => !prev)}
                                        aria-expanded={showDetails}
                                    >
                                        Detail odlehlých hodnot pro: {selectedColumn}
                                        <span className={`transform transition-transform duration-200 ${showDetails ? 'rotate-180' : 'rotate-0'}`}>▼</span>
                                    </h2>
                                    {showDetails && (
                                        <div className="p-3 max-h-[400px] overflow-y-auto">
                                            {selectedData.filteredOutliers.length > 0 ? (
                                                <table className="w-full border-collapse text-sm">
                                                    <thead className="sticky top-0 bg-gray-100 dark:bg-gray-700 z-10">
                                                    <tr className="text-left text-xs">
                                                        <th className="px-2 py-1 border dark:border-gray-600">Původní Index</th>
                                                        <th className="px-2 py-1 border dark:border-gray-600 text-right">Hodnota</th>
                                                        <th className="px-2 py-1 border dark:border-gray-600 text-right">Z-score</th>
                                                    </tr>
                                                    </thead>
                                                    <tbody className="divide-y divide-gray-200 dark:divide-gray-600">
                                                    {selectedData.filteredOutliers
                                                        // Volitelně seřadit podle absolutního Z-skóre
                                                        .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
                                                        .map((row) => (
                                                            <tr key={row.index} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                                                <td className="px-2 py-1 border dark:border-gray-600">{row.index}</td>
                                                                <td className="px-2 py-1 border dark:border-gray-600 text-right">{row.value.toFixed(3)}</td>
                                                                <td className="px-2 py-1 border dark:border-gray-600 text-right">{row.z.toFixed(2)}</td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            ) : (
                                                <p className="text-sm text-gray-500 dark:text-gray-400 p-4 text-center">
                                                    Pro sloupec '{selectedColumn}' nebyly nalezeny žádné odlehlé hodnoty podle aktuálních prahů Z-skóre ({lowerThreshold.toFixed(1)} / {upperThreshold.toFixed(1)}).
                                                </p>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </>
                )} {/* Konec podmínky pro zobrazení obsahu */}
            </main>
        </div>
    );
}