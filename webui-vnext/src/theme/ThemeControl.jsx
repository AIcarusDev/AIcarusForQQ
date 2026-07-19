"use client";

import { useEffect, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronUp,
  Monitor,
  Moon,
  Palette,
  Sun,
} from "lucide-react";
import { THEME_PALETTES } from "./themePalettes.js";

const THEME_MODES = [
  ["light", "浅色", Sun],
  ["system", "跟随系统", Monitor],
  ["dark", "深色", Moon],
];

export function ThemeControl({
  value,
  lightPalette,
  darkPalette,
  onChange,
  onPaletteChange,
  compact = false,
}) {
  const [openPalette, setOpenPalette] = useState(null);
  const rootRef = useRef(null);
  const triggerRefs = useRef({});

  useEffect(() => {
    if (!openPalette) return undefined;

    const closeOnOutsideClick = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpenPalette(null);
    };
    const closeOnEscape = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      const activeTrigger = triggerRefs.current[openPalette];
      setOpenPalette(null);
      window.requestAnimationFrame(() => activeTrigger?.focus());
    };

    document.addEventListener("pointerdown", closeOnOutsideClick);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [openPalette]);

  const selectMode = (mode) => {
    onChange(mode);
    if (mode === "system") {
      setOpenPalette(null);
      return;
    }
    setOpenPalette((current) => current === mode ? null : mode);
  };

  const selectPalette = (tone, paletteId) => {
    onPaletteChange(tone, paletteId);
    setOpenPalette(null);
    window.requestAnimationFrame(() => triggerRefs.current[tone]?.focus());
  };

  if (compact) {
    const current = THEME_MODES.find(([id]) => id === value) || THEME_MODES[1];
    const Icon = current[2];
    const next = value === "system" ? "light" : value === "light" ? "dark" : "system";
    return (
      <button
        className="theme-compact"
        type="button"
        onClick={() => onChange(next)}
        aria-label={`主题：${current[1]}，点击切换；展开侧栏可选择配色`}
        title={`主题：${current[1]} · 展开侧栏可选择配色`}
      >
        <Icon size={17} />
      </button>
    );
  }

  const selectedPalette = openPalette === "light" ? lightPalette : darkPalette;
  const popoverLabel = openPalette === "light" ? "浅色配色" : "深色配色";

  return (
    <div className="theme-control-wrap" ref={rootRef}>
      {openPalette && (
        <div
          className="theme-palette-popover"
          id={`theme-${openPalette}-palette-menu`}
          role="menu"
          aria-label={popoverLabel}
        >
          <div className="theme-palette-heading">
            <span><Palette size={14} /> {popoverLabel}</span>
            <small>选择后立即应用</small>
          </div>
          <div className="theme-palette-options">
            {THEME_PALETTES[openPalette].map((palette) => (
              <button
                key={palette.id}
                className={selectedPalette === palette.id ? "active" : ""}
                type="button"
                role="menuitemradio"
                aria-checked={selectedPalette === palette.id}
                onClick={() => selectPalette(openPalette, palette.id)}
              >
                <span className="theme-palette-swatch" aria-hidden="true">
                  {palette.preview.map((color) => <i key={color} style={{ background: color }} />)}
                </span>
                <span className="theme-palette-copy">
                  <strong>{palette.label}</strong>
                  <small>{palette.note}</small>
                </span>
                {selectedPalette === palette.id && <Check size={15} />}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="theme-control" aria-label="主题模式">
        {THEME_MODES.map(([id, label, Icon]) => {
          const hasPaletteMenu = id !== "system";
          const menuOpen = openPalette === id;
          const MenuIcon = menuOpen ? ChevronDown : ChevronUp;
          return (
            <button
              key={id}
              ref={(element) => { triggerRefs.current[id] = element; }}
              className={value === id ? "active" : ""}
              type="button"
              onClick={() => selectMode(id)}
              aria-pressed={value === id}
              aria-haspopup={hasPaletteMenu ? "menu" : undefined}
              aria-expanded={hasPaletteMenu ? menuOpen : undefined}
              aria-controls={hasPaletteMenu ? `theme-${id}-palette-menu` : undefined}
            >
              <Icon size={14} />
              <span>{label}</span>
              {hasPaletteMenu && <MenuIcon className="theme-mode-chevron" size={10} />}
            </button>
          );
        })}
      </div>
    </div>
  );
}
