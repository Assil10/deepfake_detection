export interface EvaluationRow {
  model: string;
  label: string;
  architecture: string;
  trainDomain: string;
  testedOn: string;
  auc: number;
  accuracy: number;
  recallFake: number;
  fpr: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  isCombined?: boolean;
}

export interface AblationRow {
  trainedOn: string;
  model: 'B0' | 'V2';
  testedOn: string;
  auc: number;
  accuracy: number;
  recallFake: number;
  fpr: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface DatasetSummary {
  dataset: string;
  content: string;
  totalImages: number;
  splits: string;
  trainFake: number;
  trainReal: number;
  testFake: number;
  testReal: number;
  validCount?: number;
  resolution: string;
  meanSizeKbFake: number;
  meanSizeKbReal: number;
}

// Directly sourced from data/processed/evaluation_results.csv
export const EVALUATION_RESULTS: EvaluationRow[] = [
  {
    model: 'model_combined',
    label: 'Combined EfficientNet-B0',
    architecture: 'efficientnet_b0',
    trainDomain: 'CIFAKE + Faces',
    testedOn: 'CIFAKE',
    auc: 0.9919,
    accuracy: 0.9565,
    recallFake: 0.962,
    fpr: 0.049,
    tp: 962,
    fp: 49,
    tn: 951,
    fn: 38,
    isCombined: true,
  },
  {
    model: 'model_combined',
    label: 'Combined EfficientNet-B0',
    architecture: 'efficientnet_b0',
    trainDomain: 'CIFAKE + Faces',
    testedOn: 'Faces',
    auc: 0.9721,
    accuracy: 0.905,
    recallFake: 0.904,
    fpr: 0.094,
    tp: 904,
    fp: 94,
    tn: 906,
    fn: 96,
    isCombined: true,
  },
  {
    model: 'model_cifake',
    label: 'CIFAKE EfficientNet-B0',
    architecture: 'efficientnet_b0',
    trainDomain: 'CIFAKE',
    testedOn: 'CIFAKE',
    auc: 0.9831,
    accuracy: 0.936,
    recallFake: 0.937,
    fpr: 0.065,
    tp: 937,
    fp: 65,
    tn: 935,
    fn: 63,
  },
  {
    model: 'model_cifake',
    label: 'CIFAKE EfficientNet-B0',
    architecture: 'efficientnet_b0',
    trainDomain: 'CIFAKE',
    testedOn: 'Faces',
    auc: 0.4599,
    accuracy: 0.4865,
    recallFake: 0.902,
    fpr: 0.929,
    tp: 902,
    fp: 929,
    tn: 71,
    fn: 98,
  },
  {
    model: 'model_faces',
    label: 'Faces EfficientNet-B0',
    architecture: 'efficientnet_b0',
    trainDomain: 'Faces',
    testedOn: 'CIFAKE',
    auc: 0.493,
    accuracy: 0.5005,
    recallFake: 0.071,
    fpr: 0.07,
    tp: 71,
    fp: 70,
    tn: 930,
    fn: 929,
  },
  {
    model: 'model_faces',
    label: 'Faces EfficientNet-B0',
    architecture: 'efficientnet_b0',
    trainDomain: 'Faces',
    testedOn: 'Faces',
    auc: 0.8799,
    accuracy: 0.7925,
    recallFake: 0.785,
    fpr: 0.2,
    tp: 785,
    fp: 200,
    tn: 800,
    fn: 215,
  },
];

// Directly sourced from data/processed/results_b0_vs_v2.csv
export const ABLATION_B0_VS_V2: AblationRow[] = [
  {
    trainedOn: 'CIFAKE',
    model: 'B0',
    testedOn: 'CIFAKE test',
    auc: 0.9829,
    accuracy: 0.9358,
    recallFake: 0.9365,
    fpr: 0.065,
    tp: 1887,
    fp: 129,
    tn: 1856,
    fn: 128,
  },
  {
    trainedOn: 'CIFAKE',
    model: 'B0',
    testedOn: 'Faces test',
    auc: 0.4376,
    accuracy: 0.483,
    recallFake: 0.8958,
    fpr: 0.936,
    tp: 1805,
    fp: 1858,
    tn: 127,
    fn: 210,
  },
  {
    trainedOn: 'CIFAKE',
    model: 'V2',
    testedOn: 'CIFAKE test',
    auc: 0.9841,
    accuracy: 0.9365,
    recallFake: 0.9484,
    fpr: 0.0756,
    tp: 1911,
    fp: 150,
    tn: 1835,
    fn: 104,
  },
  {
    trainedOn: 'CIFAKE',
    model: 'V2',
    testedOn: 'Faces test',
    auc: 0.4452,
    accuracy: 0.4998,
    recallFake: 0.9732,
    fpr: 0.9809,
    tp: 1961,
    fp: 1947,
    tn: 38,
    fn: 54,
  },
  {
    trainedOn: 'Faces',
    model: 'B0',
    testedOn: 'CIFAKE test',
    auc: 0.4857,
    accuracy: 0.493,
    recallFake: 0.0685,
    fpr: 0.0761,
    tp: 138,
    fp: 151,
    tn: 1834,
    fn: 1877,
  },
  {
    trainedOn: 'Faces',
    model: 'B0',
    testedOn: 'Faces test',
    auc: 0.8754,
    accuracy: 0.7948,
    recallFake: 0.7856,
    fpr: 0.196,
    tp: 1583,
    fp: 389,
    tn: 1596,
    fn: 432,
  },
  {
    trainedOn: 'Faces',
    model: 'V2',
    testedOn: 'CIFAKE test',
    auc: 0.4516,
    accuracy: 0.4965,
    recallFake: 0.0074,
    fpr: 0.0071,
    tp: 15,
    fp: 14,
    tn: 1971,
    fn: 2000,
  },
  {
    trainedOn: 'Faces',
    model: 'V2',
    testedOn: 'Faces test',
    auc: 0.9151,
    accuracy: 0.8322,
    recallFake: 0.8432,
    fpr: 0.1788,
    tp: 1699,
    fp: 355,
    tn: 1630,
    fn: 316,
  },
];

// Directly sourced from data/processed/data_exploration_summary.csv & class_distribution.csv
export const DATASET_SUMMARIES: DatasetSummary[] = [
  {
    dataset: 'CIFAKE',
    content: 'CIFAR-10 objects vs Stable Diffusion synthetics',
    totalImages: 120000,
    splits: 'train, test',
    trainFake: 50000,
    trainReal: 50000,
    testFake: 10000,
    testReal: 10000,
    resolution: '32×32 px',
    meanSizeKbFake: 0.9,
    meanSizeKbReal: 0.9,
  },
  {
    dataset: '140k Real & Fake Faces',
    content: 'Flickr human portraits vs StyleGAN faces',
    totalImages: 140000,
    splits: 'train, valid, test',
    trainFake: 50000,
    trainReal: 50000,
    testFake: 10000,
    testReal: 10000,
    validCount: 20000,
    resolution: '256×256 px',
    meanSizeKbFake: 27.7,
    meanSizeKbReal: 28.1,
  },
];

export const PIPELINE_SPECS = [
  {
    step: '01',
    title: 'JPEG Compression Augmentation',
    detail: 'RandomJPEG (p=0.6, quality 40–95) during training to prevent compression-level shortcuts.',
    source: 'scripts/train_combined.py',
  },
  {
    step: '02',
    title: 'Spatial Tensor Resize',
    detail: 'T.Resize((128, 128)) standardizes both 32×32 CIFAKE and 256×256 Faces to a unified input grid.',
    source: 'deepfake_detection/utils.py',
  },
  {
    step: '03',
    title: 'ImageNet Channel Normalization',
    detail: 'Mean [0.485, 0.456, 0.406] and Std [0.229, 0.224, 0.225] applied across RGB tensors.',
    source: 'deepfake_detection/utils.py',
  },
  {
    step: '04',
    title: 'Binary Softmax Output',
    detail: 'Class 0 mapped to Fake (AI-Generated) and Class 1 mapped to Real (Authentic Photo).',
    source: 'README.md / utils.py',
  },
];

export const SAMPLE_IMAGES = [
  {
    id: 'sample-portrait',
    title: 'Window Portrait',
    category: 'Portrait',
    src: '/src/assets/images/sample_portrait_studio_1795438372111.jpg',
  },
  {
    id: 'sample-courtyard',
    title: 'Travertine Courtyard',
    category: 'Architecture',
    src: '/src/assets/images/sample_architectural_scene_1795438371640.jpg',
  },
  {
    id: 'sample-watch',
    title: 'Crystal Tourbillon',
    category: 'Studio Object',
    src: '/src/assets/images/sample_surreal_object_1795438371913.jpg',
  },
];
