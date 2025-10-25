import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['var(--font-poppins)'],
      },
      colors: {
        brand: {
          950: '#0B0219',
          900: '#12042A',
          800: '#1C073F',
          700: '#2D0E65',
          600: '#4B1BA2',
          500: '#7B3FE4',
        },
      },
      boxShadow: {
        'inner-soft': 'inset 0 1px 40px rgba(255,255,255,0.05)',
        'glow': '0 0 30px rgba(123,63,228,0.55)',
      },
      backgroundImage: {
        'spotlight':
          'radial-gradient(1200px 500px at 50% 15%, rgba(123,63,228,0.25), rgba(123,63,228,0.10) 40%, transparent 70%)',
      },
      transitionTimingFunction: {
        'back': 'cubic-bezier(.2,.8,.2,1)',
      },
      keyframes: {
        'btn-pulse': {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.04)' },
        },
      },
      animation: {
        'btn-pulse': 'btn-pulse 2.5s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
export default config
