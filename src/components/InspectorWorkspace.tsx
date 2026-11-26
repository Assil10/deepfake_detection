import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type RefObject,
} from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  HelpCircle,
  Image as ImageIcon,
  Layers,
  Loader2,
  RefreshCw,
  Trash2,
  Upload,
} from 'lucide-react';
import { SAMPLE_IMAGES } from '../data/repoMetrics';
import type {
  DetectionEngine,
  DetectionResult,
  HistoryEntry,
  ModelCheckpointInfo,
} from '../types/detection';
import { analyzeImagePixels, type ComputedImageSignals } from '../utils/imageForensics';

type ViewMode = 'original' | 'tensor128' | 'edges';

const HISTORY_STORAGE_KEY = 'veritas_session_history_v1';
const MAX_HISTORY_ITEMS = 6;

interface InspectorWorkspaceProps {
  fileInputRef: RefObject<HTMLInputElement | null>;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function createSmallThumbnail(dataUrl: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 96;
      canvas.height = 96;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(dataUrl);
        return;
      }
      const size = Math.min(img.width, img.height);
      const sx = (img.width - size) / 2;
      const sy = (img.height - size) / 2;
      ctx.drawImage(img, sx, sy, size, size, 0, 0, 96, 96);
      resolve(canvas.toDataURL('image/jpeg', 0.75));
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}

export function InspectorWorkspace({ fileInputRef }: InspectorWorkspaceProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [signals, setSignals] = useState<ComputedImageSignals | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('original');
  const [engine, setEngine] = useState<DetectionEngine>('pytorch');
  const [models, setModels] = useState<ModelCheckpointInfo[]>([
    {
      name: 'model_combined.pth',
      label: 'Combined EfficientNet-B0',
      note: 'CIFAKE + Faces · 0.992 / 0.972 AUC',
      sizeBytes: 0,
      recommended: true,
    },
    {
      name: 'model_cifake.pth',
      label: 'CIFAKE EfficientNet-B0',
      note: 'Objects only · 0.983 CIFAKE / 0.460 Faces AUC',
      sizeBytes: 0,
      recommended: false,
    },
    {
      name: 'model_faces.pth',
      label: 'Faces EfficientNet-B0',
      note: 'Portraits only · 0.493 CIFAKE / 0.880 Faces AUC',
      sizeBytes: 0,
      recommended: false,
    },
  ]);
  const [selectedModel, setSelectedModel] = useState<string>('model_combined.pth');
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [brokenSamples, setBrokenSamples] = useState<Record<string, boolean>>({});
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      const saved = localStorage.getItem(HISTORY_STORAGE_KEY);
      return saved ? (JSON.parse(saved) as HistoryEntry[]) : [];
    } catch {
      return [];
    }
  });

  const activeRequestRef = useRef<number>(0);
  const sessionImageCacheRef = useRef<Map<string, { previewUrl: string; uploadBase64: string }>>(
    new Map()
  );

  useEffect(() => {
    let cancelled = false;
    fetch('/api/models')
      .then((res) => (res.ok ? res.json() : null))
      .then((data: { models?: ModelCheckpointInfo[] } | null) => {
        if (!cancelled && data?.models && data.models.length > 0) {
          setModels(data.models);
          setSelectedModel((prev) =>
            data.models!.some((m) => m.name === prev) ? prev : data.models![0].name
          );
        }
      })
      .catch(() => {
        // Keep default checkpoint list if endpoint unreachable
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      // Ignore storage quota issues
    }
  }, [history]);

  const runServerDetection = useCallback(
    async (
      computedSignals: ComputedImageSignals,
      imgPreviewUrl: string,
      saveToHistory = true,
      targetEngine: DetectionEngine = engine,
      targetModel: string = selectedModel
    ) => {
      const requestId = ++activeRequestRef.current;
      setIsAnalyzing(true);
      setErrorMsg(null);
      setResult(null);

      try {
        const res = await fetch('/api/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            imageBase64: computedSignals.uploadBase64,
            mimeType: computedSignals.uploadMimeType,
            fileName: computedSignals.fileName,
            engine: targetEngine,
            modelName: targetModel,
          }),
        });

        const payload = (await res.json()) as DetectionResult & { error?: string };

        if (requestId !== activeRequestRef.current) return;

        if (!res.ok) {
          throw new Error(payload.error || `Server returned status ${res.status}`);
        }

        setResult(payload);

        if (saveToHistory) {
          const thumb = await createSmallThumbnail(imgPreviewUrl);
          if (requestId !== activeRequestRef.current) return;
          const { uploadBase64: _ignored, ...lightSignals } = computedSignals;
          const entryId = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
          sessionImageCacheRef.current.set(entryId, {
            previewUrl: imgPreviewUrl,
            uploadBase64: computedSignals.uploadBase64,
          });
          const entry: HistoryEntry = {
            id: entryId,
            fileName: computedSignals.fileName,
            thumbnailDataUrl: thumb,
            verdict: payload.verdict,
            fakeProbability: payload.fakeProbability,
            realProbability: payload.realProbability,
            width: computedSignals.width,
            height: computedSignals.height,
            analyzedAt: payload.analyzedAt,
            result: payload,
            signals: lightSignals,
            previewUrl: thumb,
          };
          setHistory((prev) => [entry, ...prev].slice(0, MAX_HISTORY_ITEMS));
        }
      } catch (err) {
        if (requestId !== activeRequestRef.current) return;
        setErrorMsg(
          err instanceof Error ? err.message : 'Could not complete image authenticity check.'
        );
      } finally {
        if (requestId === activeRequestRef.current) {
          setIsAnalyzing(false);
        }
      }
    },
    [engine, selectedModel]
  );

  const handleImageInput = useCallback(
    async (source: File | { url: string; name: string }) => {
      setErrorMsg(null);
      setViewMode('original');

      if (source instanceof File) {
        if (!source.type.startsWith('image/')) {
          setErrorMsg('Please select a valid image file (JPEG, PNG, or WebP).');
          return;
        }
        if (source.size > 15 * 1024 * 1024) {
          setErrorMsg('Image exceeds the 15 MB upload limit.');
          return;
        }
      }

      try {
        setIsAnalyzing(true);
        const { previewUrl: decodedUrl, signals: computed } = await analyzeImagePixels(source);
        setPreviewUrl(decodedUrl);
        setSignals(computed);
        await runServerDetection(computed, decodedUrl, true);
      } catch (err) {
        setIsAnalyzing(false);
        setErrorMsg(err instanceof Error ? err.message : 'Unable to process image.');
      }
    },
    [runServerDetection]
  );

  // Support Ctrl+V / Cmd+V image paste
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const items = event.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            void handleImageInput(file);
            break;
          }
        }
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [handleImageInput]);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      void handleImageInput(file);
    }
  };

  const handleClear = () => {
    activeRequestRef.current++;
    setPreviewUrl(null);
    setSignals(null);
    setResult(null);
    setErrorMsg(null);
    setIsAnalyzing(false);
    setViewMode('original');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSelectHistory = (entry: HistoryEntry) => {
    activeRequestRef.current++;
    setIsAnalyzing(false);
    setErrorMsg(null);
    const cached = sessionImageCacheRef.current.get(entry.id);
    const restoredPreview = cached?.previewUrl || entry.previewUrl;
    const restoredBase64 =
      cached?.uploadBase64 ||
      (entry.previewUrl.includes(',') ? entry.previewUrl.split(',')[1] || '' : '');
    setPreviewUrl(restoredPreview);
    setSignals({ ...entry.signals, uploadBase64: restoredBase64 });
    setResult(entry.result);
    if (entry.result.engineMode) {
      setEngine(entry.result.engineMode);
    }
    if (entry.result.modelName) {
      setSelectedModel(entry.result.modelName);
    }
    setViewMode('original');
  };

  const handleEngineSwitch = (nextEngine: DetectionEngine) => {
    if (nextEngine === engine) return;
    setEngine(nextEngine);
    if (signals?.uploadBase64 && previewUrl) {
      void runServerDetection(signals, previewUrl, false, nextEngine, selectedModel);
    }
  };

  const handleModelChange = (nextModel: string) => {
    if (nextModel === selectedModel) return;
    setSelectedModel(nextModel);
    if (engine === 'pytorch' && signals?.uploadBase64 && previewUrl) {
      void runServerDetection(signals, previewUrl, false, 'pytorch', nextModel);
    }
  };

  const activeImageSrc =
    viewMode === 'tensor128' && signals
      ? signals.resized128DataUrl
      : viewMode === 'edges' && signals
        ? signals.edgeMapDataUrl
        : previewUrl;

  const activeModelMeta = models.find((m) => m.name === selectedModel) || models[0];

  return (
    <section id="detector" className="mx-auto max-w-[1360px] px-6 py-8 md:py-10 space-y-8">
      {/* Page Lead */}
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-2xl space-y-2">
          <p className="text-xs font-medium text-slate-500">
            Binary Image Authenticity Classifier · Class 0 Fake / Class 1 Real
          </p>
          <h1 className="font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl text-balance">
            Detect AI-Generated & Deepfake Photos
          </h1>
          <p className="text-sm leading-relaxed text-slate-600">
            Upload, drop, or paste any photograph to evaluate it with your{' '}
            <code className="font-mono text-xs text-slate-800">models/*.pth</code> PyTorch
            EfficientNet-B0 checkpoints or switch to Gemini Vision forensics.
          </p>
        </div>

        <div className="text-xs text-slate-500 font-mono shrink-0">
          <span>
            Active Engine:{' '}
            {engine === 'pytorch'
              ? `${selectedModel} (EfficientNet-B0)`
              : 'Gemini Vision (gemini-4-hydrogen)'}
          </span>
        </div>
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/gif"
        className="sr-only"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleImageInput(file);
        }}
      />

      {/* Asymmetric Split Console */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-stretch">
        {/* Left Viewport / Upload Stage (7 cols) */}
        <div className="lg:col-span-7 flex flex-col rounded-xl border border-slate-200 bg-white overflow-hidden">
          {/* Viewport Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-3 bg-slate-50/60">
            <div className="flex items-center gap-1 rounded-lg bg-slate-200/70 p-1">
              <button
                type="button"
                disabled={!previewUrl}
                onClick={() => setViewMode('original')}
                className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed ${
                  viewMode === 'original'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Eye className="h-3.5 w-3.5" />
                Original
              </button>
              <button
                type="button"
                disabled={!signals}
                onClick={() => setViewMode('tensor128')}
                className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed ${
                  viewMode === 'tensor128'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Layers className="h-3.5 w-3.5" />
                128×128 Input
              </button>
              <button
                type="button"
                disabled={!signals}
                onClick={() => setViewMode('edges')}
                className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed ${
                  viewMode === 'edges'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <ImageIcon className="h-3.5 w-3.5" />
                Edge Residual
              </button>
            </div>

            <div className="flex items-center gap-2">
              {previewUrl && (
                <>
                  <button
                    type="button"
                    disabled={isAnalyzing || !signals?.uploadBase64}
                    onClick={() => {
                      if (signals && previewUrl) {
                        void runServerDetection(signals, previewUrl, false);
                      }
                    }}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${isAnalyzing ? 'animate-spin' : ''}`} />
                    Re-analyze
                  </button>
                  <button
                    type="button"
                    onClick={handleClear}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Clear
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Main Image Canvas / Dropzone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`relative flex-1 min-h-[380px] md:min-h-[440px] flex items-center justify-center p-6 lab-grid-surface transition-colors ${
              isDragging ? 'bg-blue-50/70 ring-2 ring-inset ring-blue-500' : ''
            }`}
          >
            {!previewUrl ? (
              <div className="w-full max-w-md text-center space-y-6">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl border border-slate-200 bg-white text-blue-600 shadow-xs">
                  <Upload className="h-6 w-6" />
                </div>
                <div className="space-y-2">
                  <h2 className="font-display text-lg font-bold text-slate-900">
                    Drop a photo here to inspect
                  </h2>
                  <p className="text-sm text-slate-600">
                    Supports JPEG, PNG, and WebP up to 15 MB. You can also paste an image directly
                    from your clipboard.
                  </p>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-3">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 cursor-pointer"
                  >
                    Choose Image File
                  </button>
                </div>

                {/* Built-in Sample Test Images */}
                <div className="pt-4 border-t border-slate-200/80 space-y-3">
                  <p className="text-xs font-medium text-slate-500">
                    Or run a quick check on a sample image
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    {SAMPLE_IMAGES.map((sample) => (
                      <button
                        key={sample.id}
                        type="button"
                        onClick={() =>
                          void handleImageInput({ url: sample.src, name: `${sample.title}.jpg` })
                        }
                        className="group text-left rounded-lg border border-slate-200 bg-white p-1.5 hover:border-blue-400 transition-colors cursor-pointer"
                      >
                        <div className="aspect-square w-full overflow-hidden rounded-md bg-slate-100 flex items-center justify-center">
                          {brokenSamples[sample.id] ? (
                            <ImageIcon className="h-5 w-5 text-slate-400" />
                          ) : (
                            <img
                              src={sample.src}
                              alt={sample.title}
                              referrerPolicy="no-referrer"
                              onError={() =>
                                setBrokenSamples((prev) => ({ ...prev, [sample.id]: true }))
                              }
                              className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105"
                            />
                          )}
                        </div>
                        <div className="mt-1.5 px-1">
                          <div className="truncate text-xs font-medium text-slate-800">
                            {sample.title}
                          </div>
                          <div className="truncate text-[11px] text-slate-500">
                            {sample.category}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="relative flex flex-col items-center justify-center w-full h-full">
                <img
                  src={activeImageSrc || previewUrl}
                  alt={signals?.fileName || 'Uploaded photo preview'}
                  referrerPolicy="no-referrer"
                  className={`max-h-[400px] w-auto max-w-full rounded-lg border border-slate-200 bg-white object-contain shadow-xs ${
                    viewMode === 'tensor128' ? 'w-64 h-64 [image-rendering:pixelated]' : ''
                  }`}
                />
                {viewMode === 'tensor128' && (
                  <p className="mt-3 text-xs font-mono text-slate-600">
                    Nearest-canvas view of 128×128 spatial input passed to EfficientNet-B0
                  </p>
                )}
                {viewMode === 'edges' && (
                  <p className="mt-3 text-xs font-mono text-slate-600">
                    Sobel gradient magnitude map highlighting high-frequency edge consistency
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Bottom Pixel Telemetry Strip */}
          <div className="border-t border-slate-200 bg-white px-4 py-3">
            {signals ? (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-mono tabular-nums text-slate-600">
                <span className="font-medium text-slate-900 truncate max-w-[200px]">
                  {signals.fileName}
                </span>
                <span aria-hidden="true">·</span>
                <span>
                  {signals.width}×{signals.height} px ({signals.aspectRatio})
                </span>
                <span aria-hidden="true">·</span>
                <span>{formatBytes(signals.fileSizeBytes)}</span>
                <span aria-hidden="true">·</span>
                <span>
                  RGB μ [{signals.rgbMeans.map((v) => v.toFixed(2)).join(', ')}]
                </span>
                <span aria-hidden="true">·</span>
                <span>Laplacian Var {signals.laplacianVariance.toFixed(4)}</span>
              </div>
            ) : (
              <div className="text-xs text-slate-500">
                No image loaded · Select or drop a file to compute tensor & frequency diagnostics
              </div>
            )}
          </div>
        </div>

        {/* Right Diagnostic Verdict Panel (5 cols) */}
        <div className="lg:col-span-5 flex flex-col rounded-xl border border-slate-200 bg-white p-6 justify-between space-y-6">
          {/* Evaluation Engine Switch & Checkpoint Selector */}
          <div className="space-y-3 border-b border-slate-200 pb-5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Evaluation Engine
              </span>
              <span className="text-[11px] font-mono text-slate-500">
                {engine === 'pytorch' ? 'Local .pth Weights' : 'Multimodal Vision'}
              </span>
            </div>

            <div
              role="radiogroup"
              aria-label="Evaluation engine"
              className="grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1"
            >
              <button
                type="button"
                role="radio"
                aria-checked={engine === 'pytorch'}
                onClick={() => handleEngineSwitch('pytorch')}
                className={`rounded-md px-3 py-2 text-xs font-semibold transition-colors whitespace-nowrap cursor-pointer ${
                  engine === 'pytorch'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                PyTorch Model (.pth)
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={engine === 'gemini'}
                onClick={() => handleEngineSwitch('gemini')}
                className={`rounded-md px-3 py-2 text-xs font-semibold transition-colors whitespace-nowrap cursor-pointer ${
                  engine === 'gemini'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Gemini Vision
              </button>
            </div>

            {engine === 'pytorch' && (
              <div className="space-y-1.5 pt-1">
                <label
                  htmlFor="checkpoint-select"
                  className="block text-xs font-medium text-slate-600"
                >
                  Checkpoint Weights (<code className="font-mono">models/</code>)
                </label>
                <select
                  id="checkpoint-select"
                  value={selectedModel}
                  onChange={(e) => handleModelChange(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-mono text-slate-900 focus:border-blue-500 focus:bg-white focus:outline-none cursor-pointer"
                >
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name} — {m.label}
                    </option>
                  ))}
                </select>
                {activeModelMeta && (
                  <p className="text-[11px] text-slate-500 font-mono">{activeModelMeta.note}</p>
                )}
              </div>
            )}
          </div>

          {!previewUrl && !isAnalyzing && (
            <div className="flex-1 flex flex-col justify-center space-y-5 py-4">
              <div className="space-y-2">
                <h2 className="font-display text-lg font-bold text-slate-900">
                  Authenticity Verdict
                </h2>
                <p className="text-sm leading-relaxed text-slate-600">
                  Upload a photo on the left to inspect whether it was captured by an optical
                  camera or synthesized by an AI model.
                </p>
              </div>

              <div className="border-t border-slate-200 pt-5 space-y-3 text-xs text-slate-600">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Binary Output Schema</span>
                  <span className="font-mono text-slate-800">{'{ fake: 0..1, real: 0..1 }'}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Target Preprocessing</span>
                  <span className="font-mono text-slate-800">128×128 RGB · ImageNet Norm</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Selected Engine</span>
                  <span className="font-mono text-slate-800">
                    {engine === 'pytorch' ? `models/${selectedModel}` : 'gemini-4-hydrogen'}
                  </span>
                </div>
              </div>
            </div>
          )}

          {isAnalyzing && (
            <div className="flex-1 flex flex-col items-center justify-center text-center py-12 space-y-4">
              <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
              <div className="space-y-1">
                <p className="font-display text-base font-semibold text-slate-900">
                  Analyzing Photo Authenticity
                </p>
                <p className="text-xs text-slate-500">
                  Evaluating texture coherence, lighting geometry, and synthetic artifacts...
                </p>
              </div>
            </div>
          )}

          {!isAnalyzing && errorMsg && (
            <div className="flex-1 flex flex-col justify-center space-y-4 py-6">
              <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 space-y-2">
                <div className="flex items-center gap-2 text-sm font-semibold text-rose-800">
                  <AlertTriangle className="h-4 w-4 shrink-0" />
                  <span>Analysis Failed</span>
                </div>
                <p className="text-xs leading-relaxed text-rose-700">{errorMsg}</p>
              </div>
              {signals?.uploadBase64 && previewUrl && (
                <button
                  type="button"
                  onClick={() => void runServerDetection(signals, previewUrl, true)}
                  className="self-start rounded-lg bg-slate-900 px-4 py-2 text-xs font-semibold text-white hover:bg-slate-800 transition-colors cursor-pointer"
                >
                  Retry Analysis
                </button>
              )}
            </div>
          )}

          {!isAnalyzing && !errorMsg && result && (
            <div className="space-y-6">
              {/* Verdict Header */}
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-xs font-medium text-slate-500">Classification Result</span>
                  <span className="text-xs text-slate-500 font-mono">
                    {result.domainCategory} · {result.confidenceLevel} Confidence
                  </span>
                </div>

                {result.verdict === 'ai_generated' ? (
                  <div className="flex items-center gap-3 rounded-lg border border-rose-200 bg-rose-50/70 px-4 py-3.5">
                    <AlertTriangle className="h-6 w-6 text-rose-600 shrink-0" />
                    <div>
                      <div className="font-display text-xl font-bold text-rose-900">
                        AI-Generated (Fake)
                      </div>
                      <div className="text-xs text-rose-700">
                        Synthetic visual artifacts detected
                      </div>
                    </div>
                  </div>
                ) : result.verdict === 'authentic' ? (
                  <div className="flex items-center gap-3 rounded-lg border border-emerald-200 bg-emerald-50/70 px-4 py-3.5">
                    <CheckCircle2 className="h-6 w-6 text-emerald-600 shrink-0" />
                    <div>
                      <div className="font-display text-xl font-bold text-emerald-900">
                        Authentic Photo (Real)
                      </div>
                      <div className="text-xs text-emerald-700">
                        Consistent with real camera capture
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center gap-3 rounded-lg border border-amber-200 bg-amber-50/70 px-4 py-3.5">
                    <HelpCircle className="h-6 w-6 text-amber-600 shrink-0" />
                    <div>
                      <div className="font-display text-xl font-bold text-amber-900">
                        Inconclusive Signals
                      </div>
                      <div className="text-xs text-amber-700">
                        Mixed optical and synthetic indicators
                      </div>
                    </div>
                  </div>
                )}

                <p className="text-sm leading-relaxed text-slate-700">{result.summary}</p>
              </div>

              {/* Repo-Aligned Probability Split */}
              <div className="space-y-3 border-t border-slate-200 pt-5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">Class Probability Distribution</span>
                  <span className="font-mono text-slate-500">
                    {'{ fake: '}
                    {result.fakeProbability.toFixed(2)}
                    {', real: '}
                    {result.realProbability.toFixed(2)}
                    {' }'}
                  </span>
                </div>

                <div className="space-y-2.5">
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-medium text-slate-800">Class 0 · AI / Fake</span>
                      <span className="font-mono tabular-nums font-semibold text-slate-900">
                        {(result.fakeProbability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2.5 w-full rounded-xs bg-slate-100 overflow-hidden">
                      <div
                        className="h-full bg-rose-600 transition-all duration-300"
                        style={{ width: `${Math.round(result.fakeProbability * 100)}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-medium text-slate-800">Class 1 · Real Photo</span>
                      <span className="font-mono tabular-nums font-semibold text-slate-900">
                        {(result.realProbability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2.5 w-full rounded-xs bg-slate-100 overflow-hidden">
                      <div
                        className="h-full bg-emerald-600 transition-all duration-300"
                        style={{ width: `${Math.round(result.realProbability * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Observed Indicators / Model Telemetry */}
              <div className="space-y-3 border-t border-slate-200 pt-5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  {result.engineMode === 'pytorch'
                    ? 'PyTorch Model Telemetry'
                    : 'Observed Visual Indicators'}
                </h3>
                <ul className="divide-y divide-slate-100">
                  {result.indicators.map((ind, idx) => (
                    <li key={`${ind.category}-${idx}`} className="py-2.5 first:pt-0 last:pb-0">
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <span className="font-semibold text-slate-900">{ind.category}</span>
                        <span
                          className={`font-medium ${
                            ind.signal === 'synthetic'
                              ? 'text-rose-700'
                              : ind.signal === 'authentic'
                                ? 'text-emerald-700'
                                : 'text-slate-500'
                          }`}
                        >
                          {ind.signal === 'synthetic'
                            ? 'Synthetic cue'
                            : ind.signal === 'authentic'
                              ? 'Authentic cue'
                              : 'Pipeline info'}
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-slate-600">
                        {ind.observation}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Active Engine Footer Note */}
          <div className="border-t border-slate-100 pt-4 text-[11px] leading-relaxed text-slate-500">
            {engine === 'pytorch' ? (
              <span>
                Running local PyTorch inference via{' '}
                <code className="font-mono">models/{selectedModel}</code> (
                <code className="font-mono">deepfake_detection.utils</code>). Switch above to
                compare with Gemini Vision.
              </span>
            ) : (
              <span>
                Running server-side multimodal forensics via{' '}
                <code className="font-mono">gemini-4-hydrogen</code>. Switch above to evaluate with
                your local <code className="font-mono">.pth</code> checkpoints.
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Session History Strip */}
      {history.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-900">
              Recent Session Inspections ({history.length})
            </h2>
            <button
              type="button"
              onClick={() => setHistory([])}
              className="text-xs font-medium text-slate-500 hover:text-slate-900 transition-colors cursor-pointer"
            >
              Clear History
            </button>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-6">
            {history.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => handleSelectHistory(item)}
                className="group text-left rounded-lg border border-slate-200 bg-slate-50 p-2 hover:border-blue-400 transition-colors cursor-pointer"
              >
                <div className="aspect-square w-full overflow-hidden rounded-md bg-slate-200">
                  <img
                    src={item.thumbnailDataUrl}
                    alt={item.fileName}
                    referrerPolicy="no-referrer"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="mt-2 space-y-0.5">
                  <div className="truncate text-xs font-medium text-slate-900">
                    {item.fileName}
                  </div>
                  <div
                    className={`text-[11px] font-mono tabular-nums font-medium ${
                      item.verdict === 'ai_generated'
                        ? 'text-rose-700'
                        : item.verdict === 'authentic'
                          ? 'text-emerald-700'
                          : 'text-amber-700'
                    }`}
                  >
                    {item.verdict === 'ai_generated'
                      ? `AI · ${(item.fakeProbability * 100).toFixed(0)}%`
                      : item.verdict === 'authentic'
                        ? `Real · ${(item.realProbability * 100).toFixed(0)}%`
                        : 'Mixed'}
                  </div>
                  <div className="truncate text-[10px] font-mono text-slate-500">
                    {item.result.engineMode === 'gemini'
                      ? 'Gemini Vision'
                      : item.result.modelName || 'PyTorch'}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
