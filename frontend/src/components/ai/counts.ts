/** Entries of a count record in a preferred key order first, then any other keys alphabetically. */
export function orderedCounts(counts: Record<string, number>, preferred: readonly string[]): Array<[string, number]> {
  const known = preferred.filter((k) => k in counts).map((k): [string, number] => [k, counts[k]]);
  const rest = Object.entries(counts)
    .filter(([k]) => !preferred.includes(k))
    .sort(([a], [b]) => a.localeCompare(b));
  return [...known, ...rest];
}
