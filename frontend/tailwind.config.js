/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          900: '#102F26', // Deep forest
          800: '#173F32', // Dark pine
          700: '#225243',
          600: '#2E6754',
          500: '#426B4F', // Moss green
          400: '#5A8868',
          300: '#79B86A', // Bright leaf accent
          200: '#ADCFA3',
          100: '#DDE8D8', // Soft sage
          50: '#EDF4EB',  // Pale mint surface
        },
        emerald: {
          accent: '#2E8B57',
        },
        leaf: {
          accent: '#79B86A',
        },
        ivory: {
          DEFAULT: '#F6F7F0',
          dark: '#EAECE0',
        },
        primary: {
          text: '#1C2B23',
        },
        muted: {
          text: '#6B786E',
        },
        border: {
          DEFAULT: '#DCE4D9',
          subtle: '#EAEFE8',
        },
        status: {
          amber: '#C98227',
          red: '#C94C4C',
          blue: '#3977B8',
          green: '#2E8B57',
        },
        chart: {
          teal: '#2A7B88',
          gold: '#D4A338',
          sky: '#4A90E2',
          lavender: '#8E7CC3',
          emerald: '#2E8B57',
          leaf: '#79B86A',
          amber: '#C98227',
          coral: '#D95D39',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      borderRadius: {
        'card': '16px',
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgba(16, 47, 38, 0.05), 0 1px 2px -1px rgba(16, 47, 38, 0.05)',
        'card-hover': '0 4px 12px 0 rgba(16, 47, 38, 0.08), 0 2px 4px -2px rgba(16, 47, 38, 0.06)',
        'dropdown': '0 10px 25px -5px rgba(16, 47, 38, 0.12), 0 8px 10px -6px rgba(16, 47, 38, 0.08)',
      }
    },
  },
  plugins: [],
}
