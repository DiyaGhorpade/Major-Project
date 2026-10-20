import {
  Plant,
  SensorRecord,
  Session,
  Device,
  LocalShapExplanation,
  GlobalShapData,
  ModelMetrics,
  Recommendation,
  SystemSettings,
} from '../types';

export const mockPlants: Plant[] = [
  // Cluster 1 — Healthy Control (Node 1: Points 1 to 4)
  {
    id: 1,
    name: 'Pot 1: Rama Tulsi (Green Leaf)',
    species: 'Ocimum sanctum',
    variety: 'Rama Tulsi (Green)',
    cluster: 'Cluster 1',
    clusterLabel: 'Cluster 1: Healthy Control',
    stressType: 'Healthy',
    nodeId: '1',
    samplingPoint: 1,
    location: 'Bay 1 — Control Bench',
    description: 'Control specimen grown under ideal room conditions. Vibrant green leaves with rich clove-like aroma and active new shoot growth.',
    simpleCareTip: 'Keep doing what you’re doing! Ideal moisture is 45–70% with plenty of light.',
    status: 'healthy',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Too Dry (<45%)', idealLabel: 'Ideal (45–70%)', highLabel: 'Too Wet (>70%)' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold (<20°C)', idealLabel: 'Warm & Cosy (20–28°C)', highLabel: 'Hot (>28°C)' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry Air (<50%)', idealLabel: 'Pleasant (50–75%)', highLabel: 'Damp (>75%)' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim (<2500 lx)', idealLabel: 'Bright Sun (2500–6000 lx)', highLabel: 'Excessive' },
    },
    conditionPrediction: {
      condition: 'Healthy',
      confidence: 0.98,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'All readings match ideal Tulsi conditions. Roots have plenty of moisture and leaves receive ample light for photosynthesis.',
    },
  },
  {
    id: 2,
    name: 'Pot 2: Krishna Tulsi (Purple Leaf)',
    species: 'Ocimum tenuiflorum',
    variety: 'Krishna Tulsi (Purple)',
    cluster: 'Cluster 1',
    clusterLabel: 'Cluster 1: Healthy Control',
    stressType: 'Healthy',
    nodeId: '1',
    samplingPoint: 2,
    location: 'Bay 1 — Control Bench',
    description: 'Anthocyanin-rich dark purple leaf variety. High eugenol essential oil concentration and balanced vegetative canopy.',
    simpleCareTip: 'Steady watering twice a week and sunny indirect light keeps purple leaves vibrant.',
    status: 'healthy',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Too Dry (<45%)', idealLabel: 'Ideal (45–70%)', highLabel: 'Too Wet (>70%)' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold (<20°C)', idealLabel: 'Warm & Cosy (20–28°C)', highLabel: 'Hot (>28°C)' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry Air (<50%)', idealLabel: 'Pleasant (50–75%)', highLabel: 'Damp (>75%)' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim (<2500 lx)', idealLabel: 'Bright Sun (2500–6000 lx)', highLabel: 'Excessive' },
    },
    conditionPrediction: {
      condition: 'Healthy',
      confidence: 0.97,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Root moisture and temperature are stable. Purple pigment synthesis is proceeding normally.',
    },
  },
  {
    id: 3,
    name: 'Pot 3: Rama Tulsi (Green Leaf)',
    species: 'Ocimum sanctum',
    variety: 'Rama Tulsi (Green)',
    cluster: 'Cluster 1',
    clusterLabel: 'Cluster 1: Healthy Control',
    stressType: 'Healthy',
    nodeId: '1',
    samplingPoint: 3,
    location: 'Bay 1 — Control Bench',
    description: 'Control group reference pot. Uniform stem node distance and strong natural resilience.',
    simpleCareTip: 'Healthy and thriving. No action needed.',
    status: 'healthy',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Too Dry (<45%)', idealLabel: 'Ideal (45–70%)', highLabel: 'Too Wet (>70%)' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold (<20°C)', idealLabel: 'Warm & Cosy (20–28°C)', highLabel: 'Hot (>28°C)' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry Air (<50%)', idealLabel: 'Pleasant (50–75%)', highLabel: 'Damp (>75%)' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim (<2500 lx)', idealLabel: 'Bright Sun (2500–6000 lx)', highLabel: 'Excessive' },
    },
    conditionPrediction: {
      condition: 'Healthy',
      confidence: 0.96,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Transpiration balance is optimal. Leaves are firm, green, and producing aromatic oils.',
    },
  },
  {
    id: 4,
    name: 'Pot 4: Krishna Tulsi (Purple Leaf)',
    species: 'Ocimum tenuiflorum',
    variety: 'Krishna Tulsi (Purple)',
    cluster: 'Cluster 1',
    clusterLabel: 'Cluster 1: Healthy Control',
    stressType: 'Healthy',
    nodeId: '1',
    samplingPoint: 4,
    location: 'Bay 1 — Control Bench',
    description: 'Control pot. Demonstrates steady daytime transpiration and robust leaf stomata responsiveness.',
    simpleCareTip: 'Optimal condition. Keep soil lightly moist.',
    status: 'healthy',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Too Dry (<45%)', idealLabel: 'Ideal (45–70%)', highLabel: 'Too Wet (>70%)' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold (<20°C)', idealLabel: 'Warm & Cosy (20–28°C)', highLabel: 'Hot (>28°C)' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry Air (<50%)', idealLabel: 'Pleasant (50–75%)', highLabel: 'Damp (>75%)' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim (<2500 lx)', idealLabel: 'Bright Sun (2500–6000 lx)', highLabel: 'Excessive' },
    },
    conditionPrediction: {
      condition: 'Healthy',
      confidence: 0.95,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Sensor values confirm the pot has healthy root aeration and adequate moisture.',
    },
  },

  // Cluster 2 — Water Stress (Node 1: Points 5 to 8)
  {
    id: 5,
    name: 'Pot 5: Rama Tulsi (Thirsty / Dry)',
    species: 'Ocimum sanctum',
    variety: 'Rama Tulsi (Green)',
    cluster: 'Cluster 2',
    clusterLabel: 'Cluster 2: Water Stress',
    stressType: 'Water Stress',
    nodeId: '1',
    samplingPoint: 5,
    location: 'Bay 2 — Drought Trial Row',
    description: 'Thirsty pot in drought trial. Soil moisture dropped to 16.4%, causing leaf stems to lose firmness (wilting).',
    simpleCareTip: '💧 Thirsty! Needs water soon. Add 1 cup (250ml) to moisten the root ball.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Bone Dry (<45%)', idealLabel: 'Moist (45–70%)', highLabel: 'Soggy (>70%)' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry Air', idealLabel: 'Pleasant (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Sun (2500–6000 lx)', highLabel: 'Intense' },
    },
    conditionPrediction: {
      condition: 'Water Stress',
      confidence: 0.94,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'The soil is parched at 16.4% moisture (target: 45–70%). The plant is closing its pores to conserve moisture, which slows down growth.',
    },
  },
  {
    id: 6,
    name: 'Pot 6: Krishna Tulsi (Thirsty / Dry)',
    species: 'Ocimum tenuiflorum',
    variety: 'Krishna Tulsi (Purple)',
    cluster: 'Cluster 2',
    clusterLabel: 'Cluster 2: Water Stress',
    stressType: 'Water Stress',
    nodeId: '1',
    samplingPoint: 6,
    location: 'Bay 2 — Drought Trial Row',
    description: 'Drought test pot. Lower leaves are softening and drooping downward due to moisture depletion.',
    simpleCareTip: '💧 Thirsty! Water thoroughly until it gently drains out the bottom.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Bone Dry (<45%)', idealLabel: 'Moist (45–70%)', highLabel: 'Soggy (>70%)' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry Air', idealLabel: 'Pleasant (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Sun (2500–6000 lx)', highLabel: 'Intense' },
    },
    conditionPrediction: {
      condition: 'Water Stress',
      confidence: 0.92,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Capacitive soil sensor registers under 18% moisture. The roots cannot draw enough water to sustain leaf firmness.',
    },
  },
  {
    id: 7,
    name: 'Pot 7: Rama Tulsi (Thirsty / Dry)',
    species: 'Ocimum sanctum',
    variety: 'Rama Tulsi (Green)',
    cluster: 'Cluster 2',
    clusterLabel: 'Cluster 2: Water Stress',
    stressType: 'Water Stress',
    nodeId: '1',
    samplingPoint: 7,
    location: 'Bay 2 — Drought Trial Row',
    description: 'Water stress research focus pot. Primary reference subject for the local SHAP explainability waterfall.',
    simpleCareTip: '💧 Thirsty! Water needs to be restored to protect root hairs from drying out.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Bone Dry (<45%)', idealLabel: 'Moist (45–70%)', highLabel: 'Soggy (>70%)' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry Air', idealLabel: 'Pleasant (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Sun (2500–6000 lx)', highLabel: 'Intense' },
    },
    conditionPrediction: {
      condition: 'Water Stress',
      confidence: 0.95,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Soil moisture is down to 15.8%. The AI model identifies low soil hydration as the 58% dominant decision factor.',
    },
  },
  {
    id: 8,
    name: 'Pot 8: Krishna Tulsi (Thirsty / Dry)',
    species: 'Ocimum tenuiflorum',
    variety: 'Krishna Tulsi (Purple)',
    cluster: 'Cluster 2',
    clusterLabel: 'Cluster 2: Water Stress',
    stressType: 'Water Stress',
    nodeId: '1',
    samplingPoint: 8,
    location: 'Bay 2 — Drought Trial Row',
    description: 'Drought test pot showing curling leaf margins from moisture loss.',
    simpleCareTip: '💧 Thirsty! Moderate watering will perk leaves back up within 2 hours.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Bone Dry (<45%)', idealLabel: 'Moist (45–70%)', highLabel: 'Soggy (>70%)' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry Air', idealLabel: 'Pleasant (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Sun (2500–6000 lx)', highLabel: 'Intense' },
    },
    conditionPrediction: {
      condition: 'Water Stress',
      confidence: 0.91,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Sustained low soil moisture is limiting cellular hydration.',
    },
  },

  // Cluster 3 — Light Stress (Node 1: Points 9 to 12)
  {
    id: 9,
    name: 'Pot 9: Rama Tulsi (Shaded / Dim)',
    species: 'Ocimum sanctum',
    variety: 'Rama Tulsi (Green)',
    cluster: 'Cluster 3',
    clusterLabel: 'Cluster 3: Light Stress',
    stressType: 'Light Stress',
    nodeId: '1',
    samplingPoint: 9,
    location: 'Bay 3 — Low-Light Shaded Box',
    description: 'Kept in deep shade (under 450 lux vs 3,000+ lux target). Stems are stretching tall and thin (etiolation) trying to find light.',
    simpleCareTip: '☀️ Needs More Light! Move closer to grow light or a bright sunny window.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (45–70%)', highLabel: 'Wet' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Too Dim (<2500 lx)', idealLabel: 'Sunlit (2500–6000 lx)', highLabel: 'Blinding' },
    },
    conditionPrediction: {
      condition: 'Light Stress',
      confidence: 0.95,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'The photodiode measures only 412 lux. Tulsi needs bright light (>2,500 lx) to power chlorophyll energy production.',
    },
  },
  {
    id: 10,
    name: 'Pot 10: Krishna Tulsi (Shaded / Dim)',
    species: 'Ocimum tenuiflorum',
    variety: 'Krishna Tulsi (Purple)',
    cluster: 'Cluster 3',
    clusterLabel: 'Cluster 3: Light Stress',
    stressType: 'Light Stress',
    nodeId: '1',
    samplingPoint: 10,
    location: 'Bay 3 — Low-Light Shaded Box',
    description: 'Light deficiency pot. New leaves are fading to dull green because purple anthocyanin pigment requires sunlight to develop.',
    simpleCareTip: '☀️ Needs Light! At least 6 hours of bright light is required for dark purple leaves.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (45–70%)', highLabel: 'Wet' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Too Dim (<2500 lx)', idealLabel: 'Sunlit (2500–6000 lx)', highLabel: 'Blinding' },
    },
    conditionPrediction: {
      condition: 'Light Stress',
      confidence: 0.93,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Low sunlight is causing anthocyanin fading. Plant is starved of photon energy.',
    },
  },
  {
    id: 11,
    name: 'Pot 11: Rama Tulsi (Shaded / Dim)',
    species: 'Ocimum sanctum',
    variety: 'Rama Tulsi (Green)',
    cluster: 'Cluster 3',
    clusterLabel: 'Cluster 3: Light Stress',
    stressType: 'Light Stress',
    nodeId: '1',
    samplingPoint: 11,
    location: 'Bay 3 — Low-Light Shaded Box',
    description: 'Light deprivation trial. Weak stem growth and pale chlorophyll tone from lack of photon flux.',
    simpleCareTip: '☀️ Needs Light! Switch on full-spectrum LED light overhead.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (45–70%)', highLabel: 'Wet' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Too Dim (<2500 lx)', idealLabel: 'Sunlit (2500–6000 lx)', highLabel: 'Blinding' },
    },
    conditionPrediction: {
      condition: 'Light Stress',
      confidence: 0.94,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Lighting is at 380 lux. Photosynthesis is severely limited, though soil water is plentiful.',
    },
  },
  {
    id: 12,
    name: 'Pot 12: Krishna Tulsi (Shaded / Dim)',
    species: 'Ocimum tenuiflorum',
    variety: 'Krishna Tulsi (Purple)',
    cluster: 'Cluster 3',
    clusterLabel: 'Cluster 3: Light Stress',
    stressType: 'Light Stress',
    nodeId: '1',
    samplingPoint: 12,
    location: 'Bay 3 — Low-Light Shaded Box',
    description: 'Light deficiency test pot. Water uptake is unusually slow because leaves lack light to drive transpiration.',
    simpleCareTip: '☀️ Provide bright lighting for 10–12 hours daily.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (45–70%)', highLabel: 'Wet' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Too Dim (<2500 lx)', idealLabel: 'Sunlit (2500–6000 lx)', highLabel: 'Blinding' },
    },
    conditionPrediction: {
      condition: 'Light Stress',
      confidence: 0.91,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'The plant has plenty of water, but cannot process it without light photons.',
    },
  },

  // Cluster 4 — Nutrient Stress (Node 2: Points 1 to 4 -> Pots 13 to 16)
  {
    id: 13,
    name: 'Pot 13: Rama Tulsi (Nutrient Deficit)',
    species: 'Ocimum sanctum',
    variety: 'Rama Tulsi (Green)',
    cluster: 'Cluster 4',
    clusterLabel: 'Cluster 4: Nutrient Stress',
    stressType: 'Nutrient Stress',
    nodeId: '2',
    samplingPoint: 1,
    location: 'Bay 4 — Depleted Substrate Row',
    description: 'Potted in inert coir with no added plant food. Lower leaves are turning pale yellow (chlorosis) as the plant pulls stored nitrogen to top buds.',
    simpleCareTip: '🧪 Needs Plant Food! Feed with dilute organic seaweed or nitrogen-rich fertilizer.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Dry', idealLabel: 'Moist (45–70%)', highLabel: 'Wet' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Sun (2500–6000 lx)', highLabel: 'Intense' },
    },
    conditionPrediction: {
      condition: 'Nutrient Stress',
      confidence: 0.90,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Water and light are normal, but soil mineral conductivity is depleted. The plant has used up its available nitrogen and phosphorus.',
    },
  },
  {
    id: 14,
    name: 'Pot 14: Krishna Tulsi (Nutrient Deficit)',
    species: 'Ocimum tenuiflorum',
    variety: 'Krishna Tulsi (Purple)',
    cluster: 'Cluster 4',
    clusterLabel: 'Cluster 4: Nutrient Stress',
    stressType: 'Nutrient Stress',
    nodeId: '2',
    samplingPoint: 2,
    location: 'Bay 4 — Depleted Substrate Row',
    description: 'Nutrient exhaustion pot. Vegetative stems are thin and leaf expansion is stunted.',
    simpleCareTip: '🧪 Needs Plant Food! Give balanced liquid compost tea or organic NPK.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Dry', idealLabel: 'Moist (45–70%)', highLabel: 'Wet' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Sun (2500–6000 lx)', highLabel: 'Intense' },
    },
    conditionPrediction: {
      condition: 'Nutrient Stress',
      confidence: 0.88,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Nutrient depletion is restricting branch growth even though watering is regular.',
    },
  },
  {
    id: 15,
    name: 'Pot 15: Rama Tulsi (Nutrient Deficit)',
    species: 'Ocimum sanctum',
    variety: 'Rama Tulsi (Green)',
    cluster: 'Cluster 4',
    clusterLabel: 'Cluster 4: Nutrient Stress',
    stressType: 'Nutrient Stress',
    nodeId: '2',
    samplingPoint: 3,
    location: 'Bay 4 — Depleted Substrate Row',
    description: 'Nutrient study pot. Bottom leaves are visibly yellowing due to nitrogen translocation.',
    simpleCareTip: '🧪 Needs Plant Food! Add a teaspoon of vermicompost or liquid plant nutrient.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Dry', idealLabel: 'Moist (45–70%)', highLabel: 'Wet' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Sun (2500–6000 lx)', highLabel: 'Intense' },
    },
    conditionPrediction: {
      condition: 'Nutrient Stress',
      confidence: 0.91,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Chlorosis (yellowing) pattern indicates mobile nitrogen deficiency in inert potting mix.',
    },
  },
  {
    id: 16,
    name: 'Pot 16: Krishna Tulsi (Nutrient Deficit)',
    species: 'Ocimum tenuiflorum',
    variety: 'Krishna Tulsi (Purple)',
    cluster: 'Cluster 4',
    clusterLabel: 'Cluster 4: Nutrient Stress',
    stressType: 'Nutrient Stress',
    nodeId: '2',
    samplingPoint: 4,
    location: 'Bay 4 — Depleted Substrate Row',
    description: 'Substrate nutrient exhaustion trial. Plant is alive and hydrated, but needs mineral nutrition to thrive.',
    simpleCareTip: '🧪 Needs Plant Food! Enrich potting soil with micronutrients.',
    status: 'attention',
    optimalRanges: {
      soil: { min: 45, max: 70, unit: '%', lowLabel: 'Dry', idealLabel: 'Moist (45–70%)', highLabel: 'Wet' },
      temperature: { min: 20, max: 28, unit: '°C', lowLabel: 'Cold', idealLabel: 'Warm (20–28°C)', highLabel: 'Hot' },
      humidity: { min: 50, max: 75, unit: '%', lowLabel: 'Dry', idealLabel: 'Good (50–75%)', highLabel: 'Humid' },
      light: { min: 2500, max: 6000, unit: 'lux', lowLabel: 'Dim', idealLabel: 'Sun (2500–6000 lx)', highLabel: 'Intense' },
    },
    conditionPrediction: {
      condition: 'Nutrient Stress',
      confidence: 0.89,
      timestamp: '2026-10-20T12:00:00',
      modelVersion: 'v1.2.0-jetson',
      simpleExplanation: 'Sensors show adequate moisture (51%) and strong sun (4,200 lx), isolating nutrient scarcity as the primary growth bottleneck.',
    },
  },
];

// Generate 48 hours of sensor records strictly adhering to user's schema & ranges:
// timestamp: ISO 8601 string, e.g. 2024-01-15T12:00:00
// session_id: integer >= 1
// node_id: "1" or "2"
// sampling_point: 1 to 12
// plant_id: 1 to 16
// soil: 0.0 to 100.0
// temperature: -40.0 to 80.0
// humidity: 0.0 to 100.0
// light: 0.0 to 100000.0
export function generateMockRecords(): SensorRecord[] {
  const records: SensorRecord[] = [];

  const sessions = [
    { sessionId: 1, ts: '2026-10-18T08:00:00' },
    { sessionId: 2, ts: '2026-10-18T13:00:00' },
    { sessionId: 3, ts: '2026-10-18T18:00:00' },
    { sessionId: 4, ts: '2026-10-19T08:00:00' },
    { sessionId: 5, ts: '2026-10-19T13:00:00' },
    { sessionId: 6, ts: '2026-10-19T18:00:00' },
    { sessionId: 7, ts: '2026-10-20T08:00:00' },
    { sessionId: 8, ts: '2026-10-20T12:00:00' }, // Recent midday pass
  ];

  let recordCounter = 1000;

  sessions.forEach(({ sessionId, ts }) => {
    const baseDate = new Date(ts);
    const hour = baseDate.getHours();
    const isMidday = hour >= 11 && hour <= 14;
    const isMorning = hour >= 7 && hour <= 9;

    mockPlants.forEach((plant, index) => {
      recordCounter++;
      const isNode1 = plant.nodeId === '1';
      // Slight delay for sequential channel sampling
      const delayMs = isNode1 ? (plant.samplingPoint - 1) * 1200 : (plant.samplingPoint - 1) * 1500;
      const sampleDate = new Date(baseDate.getTime() + delayMs);
      const isoTimestamp = sampleDate.toISOString().slice(0, 19); // e.g. 2024-01-15T12:00:00

      let soilVal = 58.0;
      let tempVal = 24.2;
      let humVal = 62.0;
      let lightVal = 4200.0;

      if (plant.cluster === 'Cluster 1') {
        // Healthy control: optimal in all aspects
        soilVal = 56.5 + Math.sin(index) * 5.0; // 51.5% - 61.5%
        tempVal = isMidday ? 25.4 : (isMorning ? 22.1 : 23.5);
        humVal = isMidday ? 58.5 : 66.0;
        lightVal = isMidday ? 4850.0 : (isMorning ? 3100.0 : 1250.0);
      } else if (plant.cluster === 'Cluster 2') {
        // Water stress: low soil moisture 14-19%
        soilVal = 16.2 + (index % 4) * 0.9;
        tempVal = isMidday ? 27.8 : (isMorning ? 23.8 : 25.0);
        humVal = isMidday ? 42.0 : 48.5;
        lightVal = isMidday ? 4600.0 : (isMorning ? 2950.0 : 1100.0);
      } else if (plant.cluster === 'Cluster 3') {
        // Light stress: shaded box (< 500 lux)
        soilVal = 54.0 + Math.cos(index) * 3.5;
        tempVal = isMidday ? 23.2 : 21.0;
        humVal = isMidday ? 64.0 : 71.0;
        lightVal = isMidday ? 450.0 : (isMorning ? 320.0 : 140.0);
      } else {
        // Nutrient stress: adequate moisture & light, inert potting mix
        soilVal = 50.0 + (index % 3) * 2.5;
        tempVal = isMidday ? 24.6 : 22.2;
        humVal = isMidday ? 54.5 : 61.5;
        lightVal = isMidday ? 4350.0 : (isMorning ? 2900.0 : 1150.0);
      }

      // Check boundary clamps
      soilVal = Math.max(0, Math.min(100, Number(soilVal.toFixed(1))));
      tempVal = Math.max(-40, Math.min(80, Number(tempVal.toFixed(1))));
      humVal = Math.max(0, Math.min(100, Number(humVal.toFixed(1))));
      lightVal = Math.max(0, Math.min(100000, Number(lightVal.toFixed(1))));

      const isFlagged = sessionId === 5 && plant.id === 5;

      records.push({
        record_id: `REC-${recordCounter}`,
        timestamp: isoTimestamp,
        session_id: sessionId,
        node_id: plant.nodeId,
        sampling_point: plant.samplingPoint,
        plant_id: plant.id,
        soil: soilVal,
        temperature: tempVal,
        humidity: humVal,
        light: lightVal,
        received_at: new Date(sampleDate.getTime() + 180).toISOString().slice(0, 19),
        accepted_at: isFlagged ? null : new Date(sampleDate.getTime() + 180).toISOString().slice(0, 19),
        status: isFlagged ? 'flagged' : 'accepted',
        validation_error: isFlagged ? 'Soil moisture below 18% acute drought threshold' : undefined,
        rssi: plant.nodeId === '1' ? -56 : -63,
        crc_ok: true,
      });
    });
  });

  return records;
}

export const mockRecords: SensorRecord[] = generateMockRecords();

// Update latest readings for all plants
mockPlants.forEach((plant) => {
  const plantRecords = mockRecords.filter((r) => r.plant_id === plant.id);
  if (plantRecords.length > 0) {
    plant.latestReading = plantRecords[plantRecords.length - 1];
  }
});

export const mockDevices: Device[] = [
  {
    id: 'jetson-orin-nano-01',
    name: 'NVIDIA Jetson Orin Nano (Edge Brain)',
    type: 'jetson',
    ip_address: '192.168.1.120',
    mac_address: '48:b0:2d:4f:91:2c',
    status: 'online',
    last_heartbeat: '2026-10-20T12:00:15',
    last_transmission: '2026-10-20T12:00:32',
    role: 'Central AI Ingest, SHAP Explanations & SQLite Storage',
    simpleRole: 'The "Brain" — runs AI models and stores sensor logs',
    hardware_details: {
      model: 'Jetson Orin Nano Developer Kit (8GB)',
      cpu_usage: 17.5,
      gpu_usage: 6.2,
      ram_used_gb: 2.8,
      ram_total_gb: 7.6,
      disk_used_gb: 46.2,
      disk_total_gb: 238.5,
      temperature_c: 41.2,
      uptime: '14d 6h 50m',
      firmware_version: 'JetPack 5.1.2 [L4T 35.4.1]',
    },
  },
  {
    id: 'esp32-node-1',
    name: 'ESP32 Node 1 — 12-Channel Multiplexer',
    nodeNumber: '1',
    type: 'esp32',
    ip_address: '192.168.1.141',
    mac_address: '24:6f:28:a1:03:b4',
    status: 'online',
    last_heartbeat: '2026-10-20T11:59:40',
    last_transmission: '2026-10-20T12:00:18',
    role: 'Samples Points 1 to 12 (Pots 1–4 Healthy, 5–8 Water Stress, 9–12 Light Stress)',
    simpleRole: 'Reads Pots 1 to 12 via 16-channel multiplexer',
    hardware_details: {
      model: 'ESP32-WROOM-32D (CD74HC4067 Multiplexer)',
      channels: 12,
      rssi_dbm: -56,
      uptime: '4d 18h 35m',
      firmware_version: 'v2.1.4-sequential-adc',
    },
  },
  {
    id: 'esp32-node-2',
    name: 'ESP32 Node 2 — 4-Channel Analog Node',
    nodeNumber: '2',
    type: 'esp32',
    ip_address: '192.168.1.142',
    mac_address: '24:6f:28:a1:03:dc',
    status: 'online',
    last_heartbeat: '2026-10-20T11:59:42',
    last_transmission: '2026-10-20T12:00:26',
    role: 'Samples Points 1 to 4 (Pots 13–16 Nutrient Stress Tulsi)',
    simpleRole: 'Reads Pots 13 to 16 direct analog channels',
    hardware_details: {
      model: 'ESP32-WROOM-32D (Direct ADC)',
      channels: 4,
      rssi_dbm: -62,
      uptime: '4d 18h 34m',
      firmware_version: 'v2.1.4-sequential-adc',
    },
  },
];

export const mockSessions: Session[] = [
  {
    session_id: 8,
    scheduled_at: '2026-10-20T12:00:00',
    started_at: '2026-10-20T12:00:01',
    ended_at: '2026-10-20T12:00:32',
    status: 'completed',
    expected_records: 16,
    received_records: 16,
    rejected_records: 0,
    duration_seconds: 31,
    trigger_type: 'scheduled',
    node_progress: {
      '1': { expected: 12, received: 12, status: 'completed' },
      '2': { expected: 4, received: 4, status: 'completed' },
    },
  },
  {
    session_id: 7,
    scheduled_at: '2026-10-20T08:00:00',
    started_at: '2026-10-20T08:00:01',
    ended_at: '2026-10-20T08:00:30',
    status: 'completed',
    expected_records: 16,
    received_records: 16,
    rejected_records: 0,
    duration_seconds: 29,
    trigger_type: 'scheduled',
    node_progress: {
      '1': { expected: 12, received: 12, status: 'completed' },
      '2': { expected: 4, received: 4, status: 'completed' },
    },
  },
  {
    session_id: 6,
    scheduled_at: '2026-10-19T18:00:00',
    started_at: '2026-10-19T18:00:01',
    ended_at: '2026-10-19T18:00:33',
    status: 'completed',
    expected_records: 16,
    received_records: 16,
    rejected_records: 0,
    duration_seconds: 32,
    trigger_type: 'scheduled',
    node_progress: {
      '1': { expected: 12, received: 12, status: 'completed' },
      '2': { expected: 4, received: 4, status: 'completed' },
    },
  },
  {
    session_id: 5,
    scheduled_at: '2026-10-19T13:00:00',
    started_at: '2026-10-19T13:00:01',
    ended_at: '2026-10-19T13:00:35',
    status: 'partial',
    expected_records: 16,
    received_records: 15,
    rejected_records: 1,
    duration_seconds: 34,
    trigger_type: 'scheduled',
    node_progress: {
      '1': { expected: 12, received: 11, status: 'completed' },
      '2': { expected: 4, received: 4, status: 'completed' },
    },
  },
];

// Local SHAP Explanation for Pot 5 (Cluster 2 — Water Stress)
export const mockLocalShapExplanation: LocalShapExplanation = {
  plant_id: 5,
  record_id: 'REC-1125',
  model_version: 'v1.2.0-jetson',
  target_class: 'Water Stress',
  predicted_probability: 0.94,
  base_value: 0.25, // Neutral baseline: 25% chance across 4 categories
  model_output: 0.94,
  laymanSummary:
    'Think of this like a tug-of-war: The AI started with a neutral guess (25%). The bone-dry soil reading (16.4%) pushed the prediction up by +58%, convincing the AI that this Tulsi pot urgently needs water.',
  explanation_note:
    'SHAP explains which sensor readings influenced the AI model’s classification. It helps us see inside the "black box" so you can trust the recommendation.',
  features: [
    {
      name: 'soil',
      displayName: 'Soil Moisture (%)',
      observed_value: 16.4,
      unit: '%',
      shap_value: 0.58,
      direction: 'increases_prediction',
      description: 'Extremely dry soil (16.4% vs 45–70% target) is the #1 reason for the water stress call.',
      laymanExplanation: 'The soil is parched like sand. This pulls the AI heavily toward "Water Stress".',
    },
    {
      name: 'humidity',
      displayName: 'Air Humidity (%)',
      observed_value: 42.0,
      unit: '%',
      shap_value: 0.11,
      direction: 'increases_prediction',
      description: 'Dry room air (42%) pulls water out of leaves faster through transpiration.',
      laymanExplanation: 'Dry air makes the plant lose moisture even faster.',
    },
    {
      name: 'temperature',
      displayName: 'Temperature (°C)',
      observed_value: 27.8,
      unit: '°C',
      shap_value: 0.04,
      direction: 'increases_prediction',
      description: 'Warm room temperature slightly accelerates evaporation from the pot.',
      laymanExplanation: 'Warm temperature speeds up drying.',
    },
    {
      name: 'light',
      displayName: 'Light Level (lux)',
      observed_value: 4600.0,
      unit: 'lux',
      shap_value: -0.04,
      direction: 'decreases_prediction',
      description: 'Lighting is bright and healthy (4,600 lux), which proves the problem is lack of water, not lack of light.',
      laymanExplanation: 'Light is great! This rules out light problems.',
    },
  ],
};

export const mockGlobalShapData: GlobalShapData = {
  model_version: 'v1.2.0-jetson',
  sample_count: 1420,
  features: [
    {
      name: 'soil',
      displayName: 'Soil Moisture (%)',
      mean_abs_shap: 0.468,
      importance_rank: 1,
      interpretation: 'Most powerful clue: distinguishes Thirsty Pots (Cluster 2) from others.',
      laymanTip: 'Soil moisture is the single most important factor the AI looks at first.',
    },
    {
      name: 'light',
      displayName: 'Light Intensity (lux)',
      mean_abs_shap: 0.384,
      importance_rank: 2,
      interpretation: 'Second clue: instantly separates Dim Shaded Pots (Cluster 3, < 500 lx).',
      laymanTip: 'Light intensity easily flags plants sitting in the dark.',
    },
    {
      name: 'humidity',
      displayName: 'Air Humidity (%)',
      mean_abs_shap: 0.182,
      importance_rank: 3,
      interpretation: 'Third clue: indicates whether room air is drying out leaves or trapping moisture.',
      laymanTip: 'Humidity shows how fast water evaporates from leaves.',
    },
    {
      name: 'temperature',
      displayName: 'Temperature (°C)',
      mean_abs_shap: 0.115,
      importance_rank: 4,
      interpretation: 'Fine-tunes thermal stress and seasonal day/night cycles.',
      laymanTip: 'Temperature helps check if the indoor space is too hot or cold.',
    },
  ],
  beeswarm_points: [
    { id: 'b-1', feature: 'soil', feature_value_norm: 0.12, shap_value: 0.58, plant_id: 5 },
    { id: 'b-2', feature: 'soil', feature_value_norm: 0.16, shap_value: 0.52, plant_id: 7 },
    { id: 'b-3', feature: 'soil', feature_value_norm: 0.22, shap_value: 0.44, plant_id: 6 },
    { id: 'b-4', feature: 'soil', feature_value_norm: 0.50, shap_value: -0.12, plant_id: 13 },
    { id: 'b-5', feature: 'soil', feature_value_norm: 0.58, shap_value: -0.38, plant_id: 1 },
    { id: 'b-6', feature: 'soil', feature_value_norm: 0.65, shap_value: -0.48, plant_id: 3 },
    { id: 'b-7', feature: 'light', feature_value_norm: 0.08, shap_value: 0.55, plant_id: 9 },
    { id: 'b-8', feature: 'light', feature_value_norm: 0.12, shap_value: 0.49, plant_id: 11 },
    { id: 'b-9', feature: 'light', feature_value_norm: 0.18, shap_value: 0.41, plant_id: 10 },
    { id: 'b-10', feature: 'light', feature_value_norm: 0.75, shap_value: -0.28, plant_id: 1 },
    { id: 'b-11', feature: 'light', feature_value_norm: 0.88, shap_value: -0.42, plant_id: 5 },
    { id: 'b-12', feature: 'humidity', feature_value_norm: 0.25, shap_value: 0.24, plant_id: 5 },
    { id: 'b-13', feature: 'humidity', feature_value_norm: 0.65, shap_value: -0.05, plant_id: 1 },
    { id: 'b-14', feature: 'humidity', feature_value_norm: 0.85, shap_value: -0.18, plant_id: 9 },
    { id: 'b-15', feature: 'temperature', feature_value_norm: 0.85, shap_value: 0.18, plant_id: 5 },
    { id: 'b-16', feature: 'temperature', feature_value_norm: 0.45, shap_value: -0.04, plant_id: 1 },
  ],
};

// 4-Class Model Performance for Tulsi Stress Classifier
export const mockModelMetrics: ModelMetrics = {
  model_name: 'Tulsi Phenotype Stress Classifier (XGBoost)',
  model_version: 'v1.2.0-jetson',
  evaluated_at: '2026-10-18T14:30:00',
  dataset_name: 'tulsi_indoor_microclimate_stress_benchmark',
  total_samples: 1420,
  split: '70% Train, 15% Validation, 15% Holdout Test',
  overall: {
    accuracy: 0.953, // 95.3%
    precision: 0.951,
    recall: 0.953,
    f1_score: 0.952,
  },
  per_class: [
    {
      class_name: 'Healthy (Control)',
      precision: 0.971,
      recall: 0.975,
      f1_score: 0.973,
      support: 360,
      laymanStatus: 'Excellent (97.3%) — very reliable detection of healthy plants',
    },
    {
      class_name: 'Water Stress',
      precision: 0.954,
      recall: 0.948,
      f1_score: 0.951,
      support: 360,
      laymanStatus: 'High (95.1%) — catches almost all thirsty pots',
    },
    {
      class_name: 'Light Stress',
      precision: 0.962,
      recall: 0.968,
      f1_score: 0.965,
      support: 350,
      laymanStatus: 'Exceptional (96.5%) — easily recognizes light-starved plants',
    },
    {
      class_name: 'Nutrient Stress',
      precision: 0.916,
      recall: 0.920,
      f1_score: 0.918,
      support: 350,
      laymanStatus: 'Good (91.8%) — accurately spots mineral exhaustion',
    },
  ],
  confusion_matrix: {
    labels: ['Healthy', 'Water Stress', 'Light Stress', 'Nutrient Stress'],
    matrix: [
      [351, 3, 2, 4],
      [4, 341, 2, 13],
      [1, 3, 339, 7],
      [5, 11, 6, 328],
    ],
  },
};

export const mockRecommendations: Recommendation[] = [
  {
    id: 'rec-01',
    plant_id: 5,
    plant_name: 'Pot 5: Rama Tulsi',
    cluster: 'Cluster 2',
    title: 'Water Pot 5 — Soil is Getting Very Dry',
    simpleAction: '💧 Pour 1 cup (250 ml) of room-temperature water around the base of the stem.',
    priority: 'high',
    status: 'new',
    category: 'irrigation',
    supporting_sensor_values: [
      { metric: 'Soil Moisture', value: '16.4%', target: '45% – 70%', status: 'low' },
      { metric: 'Air Temperature', value: '27.8°C', target: '20°C – 28°C', status: 'good' },
      { metric: 'Light Level', value: '4,600 lx', target: '2,500 – 6,000 lx', status: 'good' },
    ],
    reason:
      'The soil has dropped to 16.4% moisture, which is well below the 45% threshold Tulsi needs to stay hydrated.',
    scientificContext:
      'When soil moisture drops below 20%, root hair water uptake ceases. Tulsi leaves close their stomatal pores to conserve internal water, causing petiole wilting and halting photosynthesis.',
    guidance_source: 'Tulsi Hydration Guidelines (Agronomic Heuristics v2.0)',
    created_at: '2026-10-20T12:05:00',
  },
  {
    id: 'rec-02',
    plant_id: 9,
    plant_name: 'Pot 9: Rama Tulsi',
    cluster: 'Cluster 3',
    title: 'Increase Light for Pot 9 — Plant is Sitting in the Dark',
    simpleAction: '☀️ Turn on the grow light or move Pot 9 closer to sunlight (aim for 12 hours of bright light daily).',
    priority: 'high',
    status: 'in_progress',
    category: 'lighting',
    supporting_sensor_values: [
      { metric: 'Light Level', value: '450 lx', target: '2,500 – 6,000 lx', status: 'low' },
      { metric: 'Soil Moisture', value: '54.0%', target: '45% – 70%', status: 'good' },
      { metric: 'Temperature', value: '23.2°C', target: '20°C – 28°C', status: 'good' },
    ],
    reason:
      'Light level is only 450 lux. Tulsi is a sun-loving plant that needs at least 2,500 lux to make food.',
    scientificContext:
      'Photosynthetically Active Radiation (PAR) is deficient. In prolonged low light, Tulsi undergoes etiolation (stretching stems and producing small, pale leaves) with reduced essential oil synthesis.',
    guidance_source: 'Indoor Medicinal Flora Lighting SOP v1.4',
    created_at: '2026-10-20T12:10:00',
  },
  {
    id: 'rec-03',
    plant_id: 13,
    plant_name: 'Pot 13: Rama Tulsi',
    cluster: 'Cluster 4',
    title: 'Feed Pot 13 — Add Organic Plant Nutrients',
    simpleAction: '🧪 Mix 1 teaspoon of organic liquid fertilizer or seaweed extract into watering can and feed Pot 13.',
    priority: 'medium',
    status: 'new',
    category: 'nutrition',
    supporting_sensor_values: [
      { metric: 'Substrate State', value: 'Inert Coir Mix', target: 'Organic Soil Mix', status: 'attention' },
      { metric: 'Soil Moisture', value: '50.0%', target: '45% – 70%', status: 'good' },
      { metric: 'Light Level', value: '4,350 lx', target: '2,500 – 6,000 lx', status: 'good' },
    ],
    reason:
      'The soil has enough moisture and light, but the plant food is depleted. Bottom leaves are turning yellow.',
    scientificContext:
      'Inert substrate without nitrogen/potassium forces Tulsi to cannibalize mobile nitrogen from older lower leaves (causing chlorosis) to feed upper shoots. Replenishing balanced N-P-K restores vegetative vigor.',
    guidance_source: 'Herbal Agronomy Nutrition Protocol v1.2',
    created_at: '2026-10-20T12:15:00',
  },
  {
    id: 'rec-04',
    plant_id: 1,
    plant_name: 'Pot 1: Rama Tulsi',
    cluster: 'Cluster 1',
    title: 'Pot 1 is Thriving — Keep Current Care Routine',
    simpleAction: '🌿 No action needed! Water is good (56.5%), light is bright (4,850 lx), and the plant is healthy.',
    priority: 'low',
    status: 'completed',
    category: 'climate',
    supporting_sensor_values: [
      { metric: 'Soil Moisture', value: '56.5%', target: '45% – 70%', status: 'good' },
      { metric: 'Temperature', value: '25.4°C', target: '20°C – 28°C', status: 'good' },
      { metric: 'Light Level', value: '4,850 lx', target: '2,500 – 6,000 lx', status: 'good' },
    ],
    reason:
      'All sensors are in the green zone. Pot 1 is a healthy control specimen.',
    scientificContext:
      'Physiological parameters align within optimal vapor pressure deficit and maximum stomatal photosynthetic efficiency for Ocimum sanctum.',
    guidance_source: 'Tulsi Baseline Control Guidelines',
    created_at: '2026-10-19T14:00:00',
    resolved_at: '2026-10-20T08:00:00',
  },
];

export const initialSystemSettings: SystemSettings = {
  api_base_url: 'http://localhost:5000',
  polling_interval_seconds: 5,
  demo_mode: true,
  jetson_receiver_port: 5000,
  ph_sensor_installed: false,
  collection_times: ['08:00', '13:00', '18:00'],
  log_retention_days: 30,
  storage_path: '/var/log/plant-monitor/records.db',
  soil_adc_dry: 3200,
  soil_adc_wet: 1450,
};
