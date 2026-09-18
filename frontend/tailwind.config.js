/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [require("daisyui")],
  daisyui: {
    // One restrained enterprise theme. A governance tool should not offer a
    // theme picker that changes what a severity colour means.
    themes: [
      {
        assurance: {
          primary: "#1e3a5f",
          "primary-content": "#ffffff",
          secondary: "#475569",
          accent: "#0f766e",
          neutral: "#1f2937",
          "base-100": "#ffffff",
          "base-200": "#f6f7f9",
          "base-300": "#e5e7eb",
          "base-content": "#111827",
          info: "#0369a1",
          success: "#15803d",
          warning: "#b45309",
          error: "#b91c1c",
          "--rounded-box": "0.5rem",
          "--rounded-btn": "0.375rem",
          "--tab-radius": "0.375rem",
        },
      },
    ],
  },
};
