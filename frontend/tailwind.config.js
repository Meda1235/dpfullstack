/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class', // Ujisti se, že je zde "class"
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
