export const THEME_STORAGE_KEYS = {
  mode: "aicarus-vnext-theme",
  light: "aicarus-vnext-theme-light-palette",
  dark: "aicarus-vnext-theme-dark-palette",
};

export const DEFAULT_THEME_PREFERENCES = {
  mode: "system",
  lightPalette: "paper",
  darkPalette: "graphite",
};

export const THEME_PALETTES = {
  light: [
    {
      id: "paper",
      label: "纸张",
      note: "当前默认 · 温暖专注",
      preview: ["#f4f2eb", "#fdfcf8", "#1d5f59"],
    },
    {
      id: "snow",
      label: "雪白",
      note: "清爽中性 · 高对比",
      preview: ["#f7f8fa", "#ffffff", "#315c8c"],
    },
    {
      id: "mist",
      label: "雾蓝",
      note: "冷静柔和 · 长时阅读",
      preview: ["#edf4f3", "#fbfdfc", "#377f78"],
    },
  ],
  dark: [
    {
      id: "graphite",
      label: "石墨",
      note: "当前默认 · 柔和低照度",
      preview: ["#101719", "#192223", "#79b6a8"],
    },
    {
      id: "oled",
      label: "纯黑",
      note: "旧版极客风 · OLED",
      preview: ["#000000", "#090909", "#58e596"],
    },
    {
      id: "midnight",
      label: "午夜",
      note: "冷蓝技术感 · 低眩光",
      preview: ["#09111d", "#101b2b", "#82b5ff"],
    },
  ],
};

const THEME_MODES = new Set(["system", "light", "dark"]);

export function isValidPalette(tone, paletteId) {
  return THEME_PALETTES[tone]?.some((palette) => palette.id === paletteId) ?? false;
}

export function getThemePalette(tone, paletteId) {
  return THEME_PALETTES[tone]?.find((palette) => palette.id === paletteId)
    ?? THEME_PALETTES[tone]?.[0];
}

export function loadThemePreferences(storage) {
  const storedMode = storage.getItem(THEME_STORAGE_KEYS.mode);
  const storedLightPalette = storage.getItem(THEME_STORAGE_KEYS.light);
  const storedDarkPalette = storage.getItem(THEME_STORAGE_KEYS.dark);

  return {
    mode: THEME_MODES.has(storedMode) ? storedMode : DEFAULT_THEME_PREFERENCES.mode,
    lightPalette: isValidPalette("light", storedLightPalette)
      ? storedLightPalette
      : DEFAULT_THEME_PREFERENCES.lightPalette,
    darkPalette: isValidPalette("dark", storedDarkPalette)
      ? storedDarkPalette
      : DEFAULT_THEME_PREFERENCES.darkPalette,
  };
}
