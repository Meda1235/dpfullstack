// components/FileUpload.tsx
import React, { useState, DragEvent, ChangeEvent } from 'react';
// Import potřebných typů z papaparse
import Papa, { ParseConfig, ParseResult, ParseError } from 'papaparse';
import { useRouter } from 'next/router';

export default function FileUpload() {
    const [headers, setHeaders] = useState<string[]>([]);
    const [rows, setRows] = useState<string[][]>([]); // Data budou string[][] z PapaParse
    const [selectedColumns, setSelectedColumns] = useState<boolean[]>([]);
    const [dragOver, setDragOver] = useState(false);
    const router = useRouter();

    // Stavy pro oddělovače
    const [delimiter, setDelimiter] = useState<string>(',');
    const [decimalSeparator, setDecimalSeparator] = useState<string>('.');

    const handleFile = (file: File) => {
        setHeaders([]);
        setRows([]);
        setSelectedColumns([]);

        // Explicitní definice typu pro konfigurační objekt
        const papaConfig: ParseConfig<string[]> = {
            delimiter: delimiter,
            skipEmptyLines: true,
            // Explicitní typ pro result v complete callbacku
            complete: (result: ParseResult<string[]>) => {
                console.log("Parse result:", result);

                // Zkontrolujeme i chyby hlášené v result objektu
                if (result.errors.length > 0) {
                    console.error("PapaParse errors:", result.errors);
                    alert(`Chyba při parsování souboru: ${result.errors[0].message}. Zkontrolujte formát, oddělovač a obsah souboru.`);
                    setHeaders([]);
                    setRows([]);
                    setSelectedColumns([]);
                    return;
                }

                const data = result.data;

                if (!data || data.length === 0) {
                    console.error("Parsování vrátilo prázdná data.");
                    alert("Nepodařilo se načíst data ze souboru (výsledek je prázdný). Zkontrolujte formát a obsah souboru.");
                    return;
                }

                // Vezmeme první řádek jako hlavičku, i když je prázdný (validace níže)
                const [headerRow, ...dataRows] = data;

                // Základní validace hlavičky - měla by existovat a mít alespoň jeden (i prázdný) název, pokud je oddělovač správný
                // Kontrola, jestli headerRow vůbec existuje (pro případ zcela prázdného souboru po skipEmptyLines)
                if (!headerRow) {
                    console.warn("Parsování selhalo - nebyla nalezena hlavička.");
                    alert(`Nepodařilo se načíst hlavičku souboru. Zkontrolujte formát souboru a zvolený oddělovač ('${delimiter}').`);
                    setHeaders([]);
                    setRows([]);
                    setSelectedColumns([]);
                    return;
                }
                // Pokud je hlavička jen jeden prázdný string a nejsou žádná data, je to podezřelé
                if (headerRow.length === 1 && headerRow[0]?.trim() === '' && dataRows.filter(row => row.some(cell => cell?.trim())).length === 0) {
                    console.warn("Parsování pravděpodobně selhalo - zkontrolujte oddělovač. Hlavička je prázdná a data chybí.");
                    alert(`Nepodařilo se správně načíst sloupce (hlavička prázdná, data chybí). Zkontrolujte, zda jste vybrali správný oddělovač dat ('${delimiter}').`);
                    setHeaders([]);
                    setRows([]);
                    setSelectedColumns([]);
                    return;
                }


                setHeaders(headerRow);
                // Odfiltrujeme řádky, které jsou zcela prázdné
                const nonEmptyRows = dataRows.filter(row => row.some(cell => cell !== null && cell !== undefined && String(cell).trim() !== ''));
                setRows(nonEmptyRows);
                setSelectedColumns(new Array(headerRow.length).fill(true));
            },
            // Správné místo pro error callback - jako vlastnost config objektu
            error: (error: ParseError, file: File) => {
                console.error("PapaParse critical error:", error, file);
                alert(`Kritická chyba při čtení nebo parsování souboru: ${error.message}`);
                setHeaders([]);
                setRows([]);
                setSelectedColumns([]);
            }
        };

        // Volání Papa.parse - první argument je File, druhý config objekt
        Papa.parse(file, papaConfig);
    };

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleFile(file);
    };

    const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) handleFile(file);
    };

    const toggleColumn = (index: number) => {
        setSelectedColumns((prev) => {
            const updated = [...prev];
            updated[index] = !updated[index];
            return updated;
        });
    };

    const storeDataInBackend = async () => {
        const filteredHeaders = headers.filter((_, i) => selectedColumns[i]);
        // Zajistíme, že i řádky mají správný počet buněk a jsou filtrovány
        const filteredRows = rows.map((row) => {
            // Vytvoříme pole správné délky (jako hlavička) a naplníme ho daty z řádku
            const fullRow = Array(headers.length).fill(''); // Výchozí hodnota pro chybějící buňky
            row.forEach((cell, index) => {
                if (index < headers.length) {
                    fullRow[index] = cell ?? ''; // Použijeme ?? pro null/undefined
                }
            });
            // Filtrujeme podle vybraných sloupců
            return fullRow.filter((_, i) => selectedColumns[i]);
        });

        console.log('Data being sent:', { headers: filteredHeaders, data: filteredRows });

        try {
            const response = await fetch('http://localhost:8000/api/store_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    headers: filteredHeaders,
                    data: filteredRows,
                    // Můžete přidat info o oddělovačích, pokud je backend potřebuje
                    // delimiter: delimiter,
                    // decimalSeparator: decimalSeparator
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ message: 'Neznámá chyba backendu' }));
                console.error('Backend error:', response.status, errorData);
                alert(`Chyba při ukládání dat na serveru: ${errorData.message || response.statusText}`);
                throw new Error(`Backend error: ${response.status}`);
            }
            console.log('✅ Data úspěšně odeslána na backend.');

        } catch (error) {
            console.error('❌ Chyba při komunikaci s backendem:', error);
            alert(`Došlo k chybě při komunikaci se serverem. Zkuste to prosím znovu.`);
            throw error; // Znovu vyhodíme chybu, aby se přesměrování neprovedlo
        }
    };

    return (
        <div className="p-8 max-w-5xl mx-auto text-gray-900 dark:text-gray-100">
            <h1 className="text-2xl font-bold mb-6">📁 Nahrání dat</h1>

            {/* --- Výběr oddělovačů s dark mode styly --- */}
            <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4 p-4 border rounded bg-gray-50 dark:bg-gray-800 border-gray-300 dark:border-gray-600">
                <div>
                    <label htmlFor="delimiter" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Oddělovač dat (sloupců):
                    </label>
                    <select
                        id="delimiter"
                        value={delimiter}
                        onChange={(e: ChangeEvent<HTMLSelectElement>) => setDelimiter(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded shadow-sm focus:ring-indigo-500 focus:border-indigo-500 bg-white dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 text-gray-900"
                    >
                        <option value=",">Čárka (,)</option>
                        <option value=";">Středník (;)</option>
                        <option value="\t">Tabulátor (Tab)</option>
                        <option value="|">Svislá čára (|)</option>
                    </select>
                </div>
                <div>
                    <label htmlFor="decimalSeparator" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Oddělovač desetinných míst:
                    </label>
                    <select
                        id="decimalSeparator"
                        value={decimalSeparator}
                        onChange={(e: ChangeEvent<HTMLSelectElement>) => setDecimalSeparator(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded shadow-sm focus:ring-indigo-500 focus:border-indigo-500 bg-white dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 text-gray-900"
                    >
                        <option value=".">Tečka (.)</option>
                        <option value=",">Čárka (,)</option>
                    </select>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Ovlivňuje interpretaci čísel.</p>
                </div>
            </div>
            {/* --- Konec výběru oddělovačů --- */}

            {/* --- Drag & Drop oblast s dark mode styly --- */}
            <label
                htmlFor="file-upload"
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                className={`w-full h-40 flex flex-col items-center justify-center border-2 border-dashed rounded cursor-pointer transition ${
                    dragOver
                        ? 'bg-blue-100 border-blue-400 dark:bg-gray-600 dark:border-blue-500'
                        : 'bg-white border-gray-300 hover:border-gray-400 dark:bg-gray-700 dark:border-gray-600 dark:hover:border-gray-500'
                }`}
            >
                <input type="file" accept=".csv,.txt" onChange={handleFileUpload} className="hidden" id="file-upload"/>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-gray-400 dark:text-gray-500 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <span className="text-gray-600 dark:text-gray-300">Přetáhněte soubor sem nebo <span className="font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300">klikněte pro výběr</span></span>
                <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">Povolené typy: CSV, TXT</span>
            </label>
            {/* --- Konec Drag & Drop oblasti --- */}

            {/* --- Zobrazení náhledu dat s dark mode styly --- */}
            {headers.length > 0 && (
                <div className="mt-6">
                    <h2 className="text-lg font-semibold mb-2">Náhled dat (prvních 10 řádků)</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Kliknutím na název sloupce jej vyberete/odvyberete pro další zpracování.</p>
                    <div className="overflow-auto border rounded shadow border-gray-200 dark:border-gray-700">
                        <table className="min-w-full text-sm table-auto">
                            <thead className="bg-gray-100 dark:bg-gray-700 sticky top-0 z-10"> {/* Přidán z-index */}
                            <tr>
                                {headers.map((h, i) => (
                                    <th
                                        key={i}
                                        className={`px-3 py-2 border-b border-r border-gray-200 dark:border-gray-600 cursor-pointer whitespace-nowrap ${
                                            selectedColumns[i]
                                                ? 'bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-100'
                                                : 'bg-gray-300 text-gray-500 line-through hover:bg-gray-400 dark:bg-gray-600 dark:text-gray-400 dark:hover:bg-gray-500'
                                        }`}
                                        onClick={() => toggleColumn(i)}
                                        title={selectedColumns[i] ? `Kliknutím odeberete sloupec '${h}'` : `Kliknutím přidáte sloupec '${h}'`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={selectedColumns[i]}
                                            readOnly
                                            tabIndex={-1}
                                            className="mr-2 pointer-events-none align-middle"
                                        />
                                        <span className="align-middle">{h || `Sloupec ${i + 1}`}</span>
                                    </th>
                                ))}
                            </tr>
                            </thead>
                            <tbody>
                            {rows.slice(0, 10).map((row, ri) => (
                                <tr key={ri} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                                    {/* Zajistíme zobrazení správného počtu buněk podle hlavičky */}
                                    {headers.map((_, ci) => (
                                        <td key={ci} className={`px-3 py-1 border-b border-r border-gray-200 dark:border-gray-700 truncate ${
                                            !selectedColumns[ci]
                                                ? 'text-gray-400 bg-gray-50 dark:text-gray-500 dark:bg-gray-800' // Nevybraná buňka
                                                : 'text-gray-900 dark:text-gray-200' // Vybraná buňka
                                        }`}>
                                            {/* Zobrazíme hodnotu buňky nebo prázdný řetězec */}
                                            {row[ci] !== undefined && row[ci] !== null ? String(row[ci]) : ''}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                            {rows.length > 10 && (
                                <tr>
                                    <td colSpan={headers.length} className="px-3 py-1 text-center text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700">
                                        ... a dalších {rows.length - 10} řádků
                                    </td>
                                </tr>
                            )}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
            {/* --- Konec náhledu dat --- */}

            {/* --- Tlačítko pro pokračování s dark mode styly --- */}
            {headers.length > 0 && (
                <button
                    onClick={async () => {
                        // Kontrola, zda je vybrán alespoň jeden sloupec
                        if (!selectedColumns.some(isSelected => isSelected)) {
                            alert("Musíte vybrat alespoň jeden sloupec pro pokračování.");
                            return;
                        }
                        try {
                            console.log('⬆️ Ukládám data do backendu...');
                            await storeDataInBackend(); // storeDataInBackend nyní může vyhodit chybu a zastavit zde
                            console.log('✅ Odesláno. Přesměrovávám...');
                            await router.push('/preanalysis');
                        } catch (error) {
                            // Chyba byla zalogována a zobrazena v storeDataInBackend
                            console.error('❌ Přesměrování zrušeno kvůli chybě při ukládání dat.');
                        }
                    }}
                    disabled={!selectedColumns.some(isSelected => isSelected)}
                    className={`mt-6 px-6 py-3 rounded-lg shadow-lg transition text-white font-semibold ${
                        selectedColumns.some(isSelected => isSelected)
                            ? 'bg-gradient-to-r from-green-500 to-green-700 hover:from-green-600 hover:to-green-800 dark:from-green-600 dark:to-green-800 dark:hover:from-green-700 dark:hover:to-green-900 cursor-pointer' // Aktivní tlačítko
                            : 'bg-gray-400 dark:bg-gray-600 dark:text-gray-400 cursor-not-allowed' // Neaktivní tlačítko
                    }`}
                >
                    Pokračovat na předběžnou analýzu
                </button>
            )}
            {/* --- Konec tlačítka --- */}
        </div>
    );
}