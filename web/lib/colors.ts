/**
 * Colour for the primary encoding, country.
 *
 * Ten muted OKLCH hues (L 0.55–0.66, C ≈ 0.13) in an order that alternates warm and cool, for
 * a near-black surface. Checked with the dataviz palette validator against #0b0d12: lightness,
 * chroma and contrast pass; adjacent legend neighbours are separable with normal vision
 * (ΔE ≥ 20) and sit in the 6–8 CVD band, which is acceptable only because the legend always
 * prints the name next to the swatch. Ten mutually CVD-safe hues do not exist, so the toggles
 * remain the exact way to tell countries apart. Countries are assigned in alphabetical order of
 * the countries present; an eleventh and later fold into grey.
 */
export const COUNTRY_PALETTE: readonly string[] = [
  "#4d97de", // blue
  "#c48225", // amber
  "#18a7a1", // teal
  "#d46e80", // rose
  "#77a246", // green
  "#ab7acd", // violet
  "#b65937", // rust
  "#6a6cc2", // indigo
  "#008e5c", // emerald
  "#a8568f", // magenta
];

export const OTHER_COLOR = "#8a8f99";

export function countryColor(index: number): string {
  return COUNTRY_PALETTE[index] ?? OTHER_COLOR;
}
