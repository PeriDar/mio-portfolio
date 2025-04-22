"use client";

import { ThemeProvider as NextThemes } from "next-themes";
import type { ThemeProviderProps } from "next-themes";

export function ThemeProvider(
  props: Omit<
    ThemeProviderProps,
    "forcedTheme" | "enableSystem" | "defaultTheme"
  >
) {
  return (
    <NextThemes
      attribute="class"
      forcedTheme="dark"
      enableSystem={false}
      defaultTheme="dark"
      disableTransitionOnChange
      {...props}
    />
  );
}
