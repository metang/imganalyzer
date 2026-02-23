/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/renderer/**/*.{html,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          50: '#f8f8f8',
          100: '#f0f0f0',
          200: '#e0e0e0',
          800: '#1a1a1a',
          900: '#111111',
          950: '#0a0a0a',
        }
      }
    }
  },
  plugins: []
}
