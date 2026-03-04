import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

/* ================================================================
   BNNR Dashboard — Theme System
   Dark (default) and Light themes with warm peach/orange brand accent (#f0a069).
   ================================================================ */

export type Theme = "dark" | "light";

interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "dark",
  toggleTheme: () => {},
});

const STORAGE_KEY = "bnnr-dashboard-theme";

function getInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* localStorage unavailable */
  }
  return "dark";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  // Apply data-theme attribute to <html>
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}

/* ----------------------------------------------------------------
   Chart color constants — used by Recharts/SVG components
   that need JS-level color values (not CSS variables).
   ---------------------------------------------------------------- */
export const CHART_THEME = {
  dark: {
    grid: "#2a2a3d",
    text: "#94a3b8",
    tooltipBg: "#1a1a2e",
    tooltipBorder: "rgba(240,160,105,0.3)",
    axisLine: "#333",
    referenceLine: "#f0a069",
    refLabelFill: "#f5b888",
    flowBg: "#0d0d18",
    cardBg: "rgba(20,20,35,0.85)",
    arcStroke: "#1a1a2e",
    chordLabelFill: "#cbd5e1",
    matrixTextLight: "#f1f5f9",
    matrixTextDark: "#0f172a",
    neutralBg: "#1e1e30",
  },
  light: {
    grid: "#e2e8f0",
    text: "#64748b",
    tooltipBg: "#ffffff",
    tooltipBorder: "rgba(200,106,53,0.25)",
    axisLine: "#cbd5e1",
    referenceLine: "#c96a35",
    refLabelFill: "#8b4a25",
    flowBg: "#faf7f2",
    cardBg: "#ffffff",
    arcStroke: "#ffffff",
    chordLabelFill: "#334155",
    matrixTextLight: "#ffffff",
    matrixTextDark: "#1e293b",
    neutralBg: "#f1f5f9",
  },
} as const;

export function useChartColors() {
  const { theme } = useTheme();
  return CHART_THEME[theme];
}
