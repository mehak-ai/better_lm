import type { Config } from "tailwindcss";

const config: Config = {
    darkMode: "class",
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ["'Plus Jakarta Sans'", "Inter", "sans-serif"],
            },
            colors: {
                accent: "#7c6af7",
                "bg-primary": "#0f0f13",
                "bg-secondary": "#16161e",
                "bg-surface": "#1c1c26",
                "bg-elevated": "#222230",
            },
            animation: {
                "fade-up": "fadeSlideUp 0.25s ease-out forwards",
            },
        },
    },
    plugins: [],
};

export default config;
