import React, { useState } from 'react';
import CorrelationAnalysis from './CorrelationAnalysis';
import RegressionAnalysis from '@/components/RegressionAnalysis';
import DependencyTestAnalysis from '@/components/DependencyTestAnalysis';

import { useEffect } from 'react';

const relationshipOptions = [
    {
        value: 'correlation',
        label: 'Korelace',
        description: 'Zjišťuje sílu a směr vztahu mezi dvěma číselnými proměnnými.',
    },
    {
        value: 'regression',
        label: 'Regrese',
        description: 'Modeluje vztah mezi závislou a nezávislou proměnnou (např. předpověď).',
    },
    {
        value: 'dependency',
        label: 'Test závislosti',
        description: 'Zkoumá, zda mezi kategoriálními proměnnými existuje statistická závislost.',
    },
];

export default function VariableRelationshipDialog() {
    const [selected, setSelected] = useState<string | null>(null);
    const [freeformText, setFreeformText] = useState('');
    const [aiResponse, setAiResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [aiSuggested, setAiSuggested] = useState<string | null>(null);

    const handleAiSuggest = async () => {
        setLoading(true);
        setAiResponse('');
        setAiSuggested(null);

        try {
            const res = await fetch('http://localhost:8000/api/ai_suggest_relationship', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: freeformText }),
            });

            const data = await res.json();
            const content = data.choices?.[0]?.message?.content ?? '';
            setAiResponse(content);
            const firstLine = content.split('\n')[0].trim();
            setAiSuggested(firstLine);
            setSelected(firstLine.toLowerCase().includes('korelace') ? 'correlation'
                : firstLine.toLowerCase().includes('regrese') ? 'regression'
                    : firstLine.toLowerCase().includes('závislost') ? 'dependency'
                        : null);
        } catch (err) {
            setAiResponse('⚠️ Chyba při získávání doporučení od AI.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-6 border rounded shadow bg-white dark:bg-gray-900">
            <h2 className="text-xl font-semibold mb-4">Zkoumání vztahů mezi proměnnými</h2>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                Zvol, jaký typ vztahu chceš mezi proměnnými zkoumat. Níže se ti zobrazí další kroky.
            </p>

            <div className="space-y-3 mb-6">
                {relationshipOptions.map((opt) => (
                    <label key={opt.value} className="block">
                        <input
                            type="radio"
                            name="relationship"
                            value={opt.value}
                            checked={selected === opt.value}
                            onChange={() => setSelected(opt.value)}
                            className="mr-2"
                        />
                        <strong>{opt.label}</strong> – <span className="text-sm text-gray-700">{opt.description}</span>
                        {aiSuggested === opt.label && (
                            <span className="ml-2 text-xs text-blue-600">(Doporučeno AI)</span>
                        )}
                    </label>
                ))}
            </div>

            <div className="mb-8">
                <h3 className="font-semibold"> AI nápověda:</h3>
                <textarea
                    className="w-full border border-gray-300 dark:border-gray-700 p-2 rounded bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
                    rows={4}
                    placeholder="Popiš, co chceš z dat zjistit..."
                    value={freeformText}
                    onChange={e => setFreeformText(e.target.value)}
                />
                <button
                    className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    onClick={handleAiSuggest}
                    disabled={loading}
                >
                    {loading ? 'Načítám...' : 'Navrhnout pomocí AI'}
                </button>
                {aiResponse && (
                    <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded">
                        <strong className="block mb-1 text-sm text-gray-600 dark:text-gray-400">Odpověď AI:</strong>
                        <p className="text-gray-800 dark:text-gray-100 whitespace-pre-line">{aiResponse}</p>
                    </div>
                )}
            </div>

            {selected === 'correlation' && (
                <div className="mt-6">
                    <h3 className="font-semibold">🔍 Korelace:</h3>
                    <CorrelationAnalysis />
                </div>
            )}

            {selected === 'regression' && (
                <div className="mt-6">
                    <h3 className="font-semibold">📈 Regrese:</h3>
                    <RegressionAnalysis />
                </div>
            )}

            {selected === 'dependency' && (
                <div className="mt-6">
                    <h3 className="font-semibold">📊 Test závislosti:</h3>
                    <DependencyTestAnalysis />
                </div>
            )}
        </div>
    );
}
