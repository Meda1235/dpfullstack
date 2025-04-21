import '../styles/globals.css';
import type { AppProps } from 'next/app';
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { DataProvider } from '../context/DataContext'; // přidáno

function MyApp({ Component, pageProps }: AppProps) {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      document.documentElement.classList.add('dark');
      setDarkMode(true);
    } else {
      document.documentElement.classList.remove('dark');
      setDarkMode(false);
    }
  }, []);

  const toggleDarkMode = () => {
    if (darkMode) {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      setDarkMode(false);
    } else {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      setDarkMode(true);
    }
  };

  return (
      <DataProvider>
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
        >
          <button
              onClick={toggleDarkMode}
              className="fixed top-4 right-4 p-2 bg-gray-800 text-white rounded-md dark:bg-gray-100 dark:text-black"
          >
            {darkMode ? '☀️ Světlý režim' : '🌙 Tmavý režim'}
          </button>

          <Component {...pageProps} />
        </motion.div>
      </DataProvider>
  );
}

export default MyApp;