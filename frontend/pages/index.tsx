// // pages/index.tsx
// import { NextPage } from 'next';
// import Head from 'next/head';
// import FileUpload from '../components/FileUpload';
//
// const Home: NextPage = () => {
//     return (
//         <div className="min-h-screen bg-gray-100 dark:bg-gray-900 dark:text-white">
//         <Head>
//                 <title>Analýza dat - Nahrání souborů</title>
//                 <meta name="description" content="Aplikace pro analýzu dat" />
//                 <link rel="icon" href="/favicon.ico" />
//             </Head>
//
//             <header className="bg-white shadow">
//                 <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
//                     <h1 className="text-3xl font-bold text-gray-900">Analýza dat</h1>
//                 </div>
//             </header>
//
//             <main>
//                 <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
//                     <FileUpload />
//                 </div>
//             </main>
//
//             <footer className="bg-white mt-auto py-4 shadow-inner">
//                 <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//                     <p className="text-center text-gray-500 text-sm">© 2025 Data Analýza</p>
//                 </div>
//             </footer>
//         </div>
//     );
// };
//
// export default Home;


import Head from 'next/head';
import FileUpload from '../components/FileUpload';

export default function Home() {
    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white flex items-center justify-center px-4">
            <Head>
                <title>Analýza dat - Nahrání</title>
            </Head>
            <main className="w-full max-w-4xl py-12 px-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
                <FileUpload />
            </main>
        </div>
    );
}

