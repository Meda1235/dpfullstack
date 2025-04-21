import React, { useState } from 'react';
import SidebarNav from '@/components/SidebarNav';
import { useRouter } from 'next/router';

export default function MainAnalysis() {
    const [freeformText, setFreeformText] = useState('');
    const [aiResponse, setAiResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [aiSuggestedLabel, setAiSuggestedLabel] = useState<string | null>(null);
    const router = useRouter();

    const handleAiSuggest = async () => {
        setLoading(true);
        setAiResponse('');
        setAiSuggestedLabel(null);

        try {
            const res = await fetch('http://localhost:8000/api/ai_suggest_analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: freeformText }),
            });

            if (!res.ok) {
                const text = await res.text();
                throw new Error(`Server error ${res.status}: ${text}`);
            }

            const data = await res.json();
            const content = data.choices?.[0]?.message?.content ?? '';
            setAiResponse(content);

            const firstLine = content.split('\n')[0].trim();
            setAiSuggestedLabel(firstLine);
        } catch (err) {
            setAiResponse('⚠️ Chyba při získávání doporučení od AI.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const options = [
        {
            label: 'Porovnání skupin',
            description: 'Např. rozdíl v platech mezi odděleními',
            path: '/groupComparison'
        },
        {
            label: 'Vztah mezi proměnnými',
            description: 'Např. závislost věku a příjmů',
            path: '/relationship'
        },
        {
            label: 'Klasifikace',
            description: 'Např. rozpoznání typu zákazníka',
            path: '/classification'
        },
        {
            label: 'Shluková analýza',
            description: 'Např. hledání skupin zákazníků podle chování',
            path: '/clustering'
        },
        {
            label: 'Faktorová analýza',
            description: 'Např. redukce dimenzí nebo identifikace latentních faktorů',
            path: '/factor-analysis'
        }
    ];

    return (
        <div className="flex">
            <SidebarNav />
            <main className="flex-1 p-6 space-y-6">
                <h1 className="text-2xl font-bold">Hlavní analýza</h1>
                <p className="text-gray-600 dark:text-gray-300">
                    Vyber typ analýzy nebo popiš svůj cíl a nech si poradit pomocí AI.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {options.map((opt, idx) => {
                        const isSuggested = aiSuggestedLabel === opt.label;
                        return (
                            <div
                                key={idx}
                                className={`p-4 border rounded-xl shadow-sm cursor-pointer transition 
                                ${isSuggested
                                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900'
                                    : 'border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800'}`}
                                onClick={() => router.push(opt.path)}
                            >
                                <div className="font-semibold text-gray-800 dark:text-gray-100">{opt.label}</div>
                                <div className="text-sm text-gray-600 dark:text-gray-400">{opt.description}</div>
                                {isSuggested && (
                                    <div className="mt-2 text-xs text-blue-700 bg-blue-100 dark:bg-blue-800 dark:text-blue-200 px-2 py-0.5 rounded inline-block">
                                        Doporučeno AI
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                <div className="mt-6">
                    <label className="block mb-1 font-medium">Popiš, co chceš z dat zjistit:</label>
                    <textarea
                        className="w-full border border-gray-300 dark:border-gray-700 p-2 rounded bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
                        rows={4}
                        placeholder="Např. chci vědět, jestli vzdělání ovlivňuje výši mzdy..."
                        value={freeformText}
                        onChange={e => setFreeformText(e.target.value)}
                    />
                    <button
                        className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                        onClick={handleAiSuggest}
                        disabled={loading}
                    >
                        {loading ? 'Načítám...' : 'Navrhnout analýzu pomocí AI'}
                    </button>

                    {aiResponse && (
                        <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded">
                            <strong className="block mb-1 text-sm text-gray-600 dark:text-gray-400">Odpověď AI:</strong>
                            <p className="text-gray-800 dark:text-gray-100 whitespace-pre-line">{aiResponse}</p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}