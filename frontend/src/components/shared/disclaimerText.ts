/** Required disclaimer copy (master prompt section 9.7). Reuse these strings; do not reword. */
export const DISCLAIMERS = {
  synthetic: 'Synthetic dataset. Model outputs are experimental, not validated plant diagnoses.',
  anomaly: 'Unusual reading, check the sensor. Not a plant-health verdict.',
  envState: 'Environmental state group. Not a stress level.',
  recommendation: 'Advisory heuristic.',
  modelPerformance: 'Metrics measured on synthetic labels.',
  conditionGuess: 'Heuristic guess of the stress type from the recommendation categories. Not a diagnosis.',
  runPrompt: 'Press Run to analyze the bundled plant dataset.',
} as const;

export type DisclaimerKey = keyof typeof DISCLAIMERS;
