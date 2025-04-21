import Link from 'next/link';
import { useRouter } from 'next/router';

export default function SidebarNav() {
    const router = useRouter();

    return (
        <div className="w-60 pr-4 border-r border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 p-4 rounded-xl shadow-sm">
            <nav className="space-y-6">
                <div>
                    <Link
                        href="/"
                        className={`block px-4 py-2 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900 transition font-medium ${
                            router.pathname === '/' ? 'bg-blue-200 dark:bg-blue-800 text-blue-900 dark:text-blue-100' : 'text-gray-800 dark:text-gray-200'
                        }`}
                    >
                         Nahrání dat
                    </Link>
                </div>

                <details className="group" open>
                    <summary className="cursor-pointer px-4 py-2 font-semibold text-gray-700 dark:text-gray-100 hover:bg-gray-200 dark:hover:bg-gray-800 rounded-md">
                        📂 Explorativní analýza
                    </summary>
                    <ul className="mt-2 space-y-1 border-l border-gray-300 dark:border-gray-600 ml-2 pl-2">
                        <li>
                            <Link
                                href="/preanalysis/columns"
                                className={`block px-3 py-1.5 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900 transition ${
                                    router.pathname === '/preanalysis/columns' ? 'bg-blue-200 dark:bg-blue-800 text-blue-900 dark:text-blue-100' : 'text-gray-800 dark:text-gray-200'
                                }`}
                            >
                                 Sloupce
                            </Link>
                        </li>
                        <li>
                            <Link
                                href="/preanalysis/missing"
                                className={`block px-3 py-1.5 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900 transition ${
                                    router.pathname === '/preanalysis/missing' ? 'bg-blue-200 dark:bg-blue-800 text-blue-900 dark:text-blue-100' : 'text-gray-800 dark:text-gray-200'
                                }`}
                            >
                                 Chybějící hodnoty
                            </Link>
                        </li>
                        <li>
                            <Link
                                href="/preanalysis/outliers"
                                className={`block px-3 py-1.5 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900 transition ${
                                    router.pathname === '/preanalysis/outliers' ? 'bg-blue-200 dark:bg-blue-800 text-blue-900 dark:text-blue-100' : 'text-gray-800 dark:text-gray-200'
                                }`}
                            >
                                 Extrémní hodnoty
                            </Link>
                        </li>
                        <li>
                            <Link
                                href="/preanalysis/normality"
                                className={`block px-3 py-1.5 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900 transition ${
                                    router.pathname === '/preanalysis/normality' ? 'bg-blue-200 dark:bg-blue-800 text-blue-900 dark:text-blue-100' : 'text-gray-800 dark:text-gray-200'
                                }`}
                            >
                                 Normalita
                            </Link>
                        </li>

                    </ul>
                </details>

                <div>
                    <Link
                        href="/analysis"
                        className={`block px-4 py-2 mt-4 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900 transition font-medium ${
                            router.pathname === '/analysis' ? 'bg-blue-200 dark:bg-blue-800 text-blue-900 dark:text-blue-100' : 'text-gray-800 dark:text-gray-200'
                        }`}
                    >
                         Hlavní analýza
                    </Link>
                </div>
            </nav>
        </div>
    );
}
