/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#09090b', // Deepest black/gray
          800: '#18181b', // Standard dark background
          700: '#27272a', // Lighter panel background
        },
        brand: {
          indigo: '#4f46e5',
          purple: '#9333ea',
          cyan: '#06b6d4',
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}