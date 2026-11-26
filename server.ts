import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const MODELS_DIR = path.resolve(__dirname, 'models');
const WORKER_SCRIPT = path.resolve(__dirname, 'scripts', 'inference_worker.py');
const VENV_PYTHON = path.resolve(__dirname, '.venv', 'bin', 'python');

const ALLOWED_MIME_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/gif',
]);

const KNOWN_MODEL_META: Record<string, { label: string; note: string; recommended?: boolean }> = {
  'model_combined.pth': {
    label: 'Combined EfficientNet-B0',
    note: 'CIFAKE + Faces · 0.992 / 0.972 AUC',
    recommended: true,
  },
  'model_cifake.pth': {
    label: 'CIFAKE EfficientNet-B0',
    note: 'Objects only · 0.983 CIFAKE / 0.460 Faces AUC',
  },
  'model_faces.pth': {
    label: 'Faces EfficientNet-B0',
    note: 'Portraits only · 0.493 CIFAKE / 0.880 Faces AUC',
  },
};

interface DetectRequestBody {
  imageBase64?: unknown;
  mimeType?: unknown;
  fileName?: unknown;
  engine?: unknown;
  modelName?: unknown;
}

interface RawIndicator {
  category?: unknown;
  observation?: unknown;
  signal?: unknown;
}

interface RawDetectionOutput {
  verdict?: unknown;
  fakeProbability?: unknown;
  realProbability?: unknown;
  confidenceLevel?: unknown;
  domainCategory?: unknown;
  summary?: unknown;
  indicators?: unknown;
}

interface WorkerResponse {
  id?: number;
  ready?: boolean;
  ok?: boolean;
  error?: string;
  modelName?: string;
  arch?: string;
  device?: string;
  fake?: number;
  real?: number;
  logits?: [number, number];
  latencyMs?: number;
}

interface PendingWorkerRequest {
  resolve: (value: WorkerResponse) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

class PyTorchInferenceClient {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private readyPromise: Promise<void> | null = null;
  private nextId = 1;
  private pending = new Map<number, PendingWorkerRequest>();

  private getPythonBinary(): string {
    if (fs.existsSync(VENV_PYTHON)) {
      return VENV_PYTHON;
    }
    return 'python3';
  }

  private ensureWorker(): Promise<void> {
    if (this.proc && !this.proc.killed && this.readyPromise) {
      return this.readyPromise;
    }

    const pythonBin = this.getPythonBinary();
    const proc = spawn(pythonBin, [WORKER_SCRIPT], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.proc = proc;

    const rl = readline.createInterface({ input: proc.stdout });
    let stderrBuf = '';

    proc.stderr.on('data', (chunk: Buffer) => {
      stderrBuf += chunk.toString();
      if (stderrBuf.length > 4000) {
        stderrBuf = stderrBuf.slice(-4000);
      }
    });

    this.readyPromise = new Promise<void>((resolveReady, rejectReady) => {
      let settledReady = false;
      const readyTimeout = setTimeout(() => {
        if (!settledReady) {
          settledReady = true;
          rejectReady(new Error('PyTorch worker startup timed out.'));
        }
      }, 30000);

      rl.on('line', (line) => {
        if (!line.trim()) return;
        let msg: WorkerResponse;
        try {
          msg = JSON.parse(line) as WorkerResponse;
        } catch {
          return;
        }

        if (msg.ready && !settledReady) {
          settledReady = true;
          clearTimeout(readyTimeout);
          resolveReady();
          return;
        }

        if (typeof msg.id === 'number') {
          const entry = this.pending.get(msg.id);
          if (entry) {
            clearTimeout(entry.timer);
            this.pending.delete(msg.id);
            if (msg.ok) {
              entry.resolve(msg);
            } else {
              entry.reject(new Error(msg.error || 'PyTorch inference failed.'));
            }
          }
        }
      });

      proc.on('exit', (code) => {
        this.proc = null;
        this.readyPromise = null;
        if (!settledReady) {
          settledReady = true;
          clearTimeout(readyTimeout);
          rejectReady(
            new Error(
              `PyTorch worker exited with code ${code}${stderrBuf ? `: ${stderrBuf.trim()}` : ''}`
            )
          );
        }
        for (const [id, entry] of this.pending.entries()) {
          clearTimeout(entry.timer);
          entry.reject(new Error('PyTorch worker terminated unexpectedly.'));
          this.pending.delete(id);
        }
      });
    });

    return this.readyPromise;
  }

  async predict(modelName: string, imageBase64: string): Promise<WorkerResponse> {
    await this.ensureWorker();
    if (!this.proc) {
      throw new Error('PyTorch worker is not running.');
    }

    const id = this.nextId++;
    return new Promise<WorkerResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('PyTorch inference timed out.'));
      }, 30000);

      this.pending.set(id, { resolve, reject, timer });
      const payload = JSON.stringify({ id, modelName, imageBase64 }) + '\n';
      this.proc!.stdin.write(payload);
    });
  }
}

const pytorchWorker = new PyTorchInferenceClient();

function listAvailableModels() {
  if (!fs.existsSync(MODELS_DIR)) {
    return [];
  }
  const files = fs
    .readdirSync(MODELS_DIR)
    .filter((f) => f.endsWith('.pth') && fs.statSync(path.join(MODELS_DIR, f)).isFile());

  const list = files.map((name) => {
    const stat = fs.statSync(path.join(MODELS_DIR, name));
    const meta = KNOWN_MODEL_META[name];
    return {
      name,
      label: meta?.label || name.replace(/\.pth$/, ''),
      note: meta?.note || 'Custom PyTorch EfficientNet-B0 checkpoint',
      sizeBytes: stat.size,
      recommended: Boolean(meta?.recommended),
    };
  });

  list.sort((a, b) => {
    if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  return list;
}

function getGenAIClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured on the server.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function extractReadableError(err: unknown): { statusCode: number; message: string } {
  const raw = err instanceof Error ? err.message : 'Unexpected error analyzing image.';
  let parsedMessage = raw;
  let parsedCode: number | undefined;

  try {
    const obj = JSON.parse(raw) as { error?: { code?: number; message?: string; status?: string } };
    if (obj?.error?.message) {
      parsedMessage = obj.error.message;
      parsedCode = obj.error.code;
    }
  } catch {
    // Keep raw message if not JSON
  }

  if (
    parsedMessage.includes('API_KEY_INVALID') ||
    parsedMessage.includes('PERMISSION_DENIED') ||
    parsedCode === 403
  ) {
    return { statusCode: 403, message: parsedMessage };
  }
  if (parsedCode === 429 || parsedMessage.includes('RESOURCE_EXHAUSTED')) {
    return { statusCode: 429, message: parsedMessage };
  }
  if (
    parsedCode === 503 ||
    parsedMessage.includes('UNAVAILABLE') ||
    parsedMessage.includes('high demand')
  ) {
    return {
      statusCode: 503,
      message: 'The Gemini model is temporarily busy due to high demand. Please retry in a few seconds.',
    };
  }
  return { statusCode: 500, message: parsedMessage };
}

async function generateWithRetry<T>(fn: () => Promise<T>, maxAttempts = 3): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const msg = err instanceof Error ? err.message : '';
      const isTransient =
        msg.includes('503') ||
        msg.includes('UNAVAILABLE') ||
        msg.includes('high demand') ||
        msg.includes('429');
      if (!isTransient || attempt === maxAttempts) {
        throw err;
      }
      await new Promise((resolve) => setTimeout(resolve, attempt * 1000));
    }
  }
  throw lastError;
}

async function startServer() {
  const app = express();

  app.use(express.json({ limit: '16mb' }));

  app.get('/api/health', (_req, res) => {
    const models = listAvailableModels();
    res.json({
      ok: true,
      hasApiKey: Boolean(process.env.GEMINI_API_KEY),
      pythonAvailable: fs.existsSync(VENV_PYTHON),
      modelsCount: models.length,
    });
  });

  app.get('/api/models', (_req, res) => {
    const models = listAvailableModels();
    res.json({
      models,
      pythonAvailable: fs.existsSync(VENV_PYTHON),
    });
  });

  app.post('/api/detect', async (req, res) => {
    try {
      const body = req.body as DetectRequestBody;
      const imageBase64 = typeof body.imageBase64 === 'string' ? body.imageBase64.trim() : '';
      const mimeType = typeof body.mimeType === 'string' ? body.mimeType.trim() : 'image/jpeg';
      const fileName = typeof body.fileName === 'string' ? body.fileName.trim() : 'uploaded-image';
      const requestedEngine = body.engine === 'gemini' ? 'gemini' : 'pytorch';
      const requestedModel =
        typeof body.modelName === 'string' && body.modelName.trim()
          ? body.modelName.trim()
          : 'model_combined.pth';

      if (!imageBase64) {
        res.status(400).json({ error: 'No image data provided.' });
        return;
      }

      if (!ALLOWED_MIME_TYPES.has(mimeType)) {
        res.status(400).json({
          error: `Unsupported image format "${mimeType}". Please upload a JPEG, PNG, or WebP photo.`,
        });
        return;
      }

      if (requestedEngine === 'pytorch') {
        const available = listAvailableModels();
        if (available.length === 0) {
          res.status(404).json({
            error: 'No .pth model files were found in the models/ directory.',
          });
          return;
        }
        const chosen = available.find((m) => m.name === requestedModel) || available[0];
        const workerRes = await pytorchWorker.predict(chosen.name, imageBase64);

        const rawFake = typeof workerRes.fake === 'number' ? workerRes.fake : 0.5;
        const rawReal = typeof workerRes.real === 'number' ? workerRes.real : 1 - rawFake;
        const totalProb = rawFake + rawReal;
        const fakeProb =
          totalProb > 0
            ? Number(Math.min(1, Math.max(0, rawFake / totalProb)).toFixed(4))
            : 0.5;
        const realProb = Number((1 - fakeProb).toFixed(4));

        const verdict: 'ai_generated' | 'authentic' | 'inconclusive' =
          fakeProb >= 0.5 ? 'ai_generated' : 'authentic';

        const maxProb = Math.max(fakeProb, realProb);
        const confidenceLevel: 'High' | 'Moderate' | 'Low' =
          maxProb >= 0.85 ? 'High' : maxProb >= 0.65 ? 'Moderate' : 'Low';

        const logitPair = Array.isArray(workerRes.logits)
          ? `[${workerRes.logits[0].toFixed(3)}, ${workerRes.logits[1].toFixed(3)}]`
          : 'N/A';

        const pctWinner = (maxProb * 100).toFixed(1);
        const summary =
          verdict === 'ai_generated'
            ? `${chosen.name} classified this image as AI-generated (Class 0) with ${pctWinner}% softmax probability.`
            : `${chosen.name} classified this image as an authentic photograph (Class 1) with ${pctWinner}% softmax probability.`;

        res.json({
          fileName,
          verdict,
          fakeProbability: fakeProb,
          realProbability: realProb,
          confidenceLevel,
          domainCategory: chosen.label,
          summary,
          indicators: [
            {
              category: 'PyTorch Checkpoint',
              observation: `Evaluated directly with models/${chosen.name} (${workerRes.arch || 'efficientnet_b0'} on ${workerRes.device || 'cpu'}).`,
              signal: verdict === 'ai_generated' ? 'synthetic' : 'authentic',
            },
            {
              category: 'Tensor Preprocessing',
              observation:
                'Resized to 128×128 RGB and normalized with ImageNet channel mean/std per deepfake_detection/utils.py.',
              signal: 'neutral',
            },
            {
              category: 'Softmax & Logits',
              observation: `Output logits ${logitPair} mapped to Class 0 Fake (${(fakeProb * 100).toFixed(2)}%) and Class 1 Real (${(realProb * 100).toFixed(2)}%).`,
              signal: verdict === 'ai_generated' ? 'synthetic' : 'authentic',
            },
            {
              category: 'Inference Latency',
              observation: `Forward pass completed in ${workerRes.latencyMs ?? 0} ms via persistent PyTorch worker.`,
              signal: 'neutral',
            },
          ],
          analyzedAt: new Date().toISOString(),
          engine: `${chosen.name} (${workerRes.arch || 'efficientnet_b0'})`,
          engineMode: 'pytorch',
          modelName: chosen.name,
          latencyMs: workerRes.latencyMs,
        });
        return;
      }

      // Gemini Vision evaluation mode
      if (!process.env.GEMINI_API_KEY) {
        res.status(503).json({
          error: 'Gemini evaluation requires GEMINI_API_KEY in server configuration.',
        });
        return;
      }

      const ai = getGenAIClient();

      const prompt = [
        'Examine this uploaded image carefully to determine whether it is AI-generated (synthetic/deepfake) or an authentic real-world photograph.',
        'Inspect fine texture details (skin pores, hair strands, fabric weave, natural sensor grain vs over-smoothed diffusion sheen), lighting and shadow geometry, reflections, anatomical/structural consistency (hands, eyes, teeth, text, straight architectural lines), and background coherence.',
        'Provide calibrated probabilities matching the binary convention fakeProbability (0.0 to 1.0, likelihood image is AI-generated) and realProbability (0.0 to 1.0, likelihood image is an authentic photograph) summing to 1.0.',
        'Be objective: if the image looks genuinely photographed with natural optical imperfections, score realProbability higher; if it exhibits synthetic generation, diffusion, or GAN traits, score fakeProbability higher.',
      ].join(' ');

      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: {
            parts: [
              {
                inlineData: {
                  mimeType,
                  data: imageBase64,
                },
              },
              { text: prompt },
            ],
          },
          config: {
            systemInstruction:
              'You are an objective digital media forensics analyst evaluating whether an image is AI-generated or an authentic photograph. Output strictly valid JSON conforming to the requested schema. Cite only visual evidence actually observable in the submitted image.',
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                verdict: {
                  type: Type.STRING,
                  description: 'Classification verdict: ai_generated, authentic, or inconclusive.',
                },
                fakeProbability: {
                  type: Type.NUMBER,
                  description: 'Calibrated probability from 0.0 to 1.0 that the image is AI-generated.',
                },
                realProbability: {
                  type: Type.NUMBER,
                  description: 'Calibrated probability from 0.0 to 1.0 that the image is an authentic photograph.',
                },
                confidenceLevel: {
                  type: Type.STRING,
                  description: 'Confidence tier: High, Moderate, or Low.',
                },
                domainCategory: {
                  type: Type.STRING,
                  description:
                    'Primary visual domain: Portrait / Human Face, Object / Scene, Graphic / Illustration, or Composite / Edited.',
                },
                summary: {
                  type: Type.STRING,
                  description: 'Concise 1-2 sentence forensic summary explaining the verdict.',
                },
                indicators: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      category: {
                        type: Type.STRING,
                        description:
                          'Short label such as Lighting & Shadows, Surface Texture, Geometry & Edges, or Lens & Grain.',
                      },
                      observation: {
                        type: Type.STRING,
                        description: 'Specific visual observation from the uploaded image.',
                      },
                      signal: {
                        type: Type.STRING,
                        description: 'Whether this cue points toward synthetic, authentic, or neutral.',
                      },
                    },
                    required: ['category', 'observation', 'signal'],
                  },
                  description: '3 to 5 concrete visual indicators observed in the image.',
                },
              },
              required: [
                'verdict',
                'fakeProbability',
                'realProbability',
                'confidenceLevel',
                'domainCategory',
                'summary',
                'indicators',
              ],
            },
          },
        })
      );

      const rawText = (response.text ?? '{}').trim();
      const parsed = JSON.parse(rawText) as RawDetectionOutput;

      let fakeProb = typeof parsed.fakeProbability === 'number' ? parsed.fakeProbability : 0.5;
      let realProb = typeof parsed.realProbability === 'number' ? parsed.realProbability : 1 - fakeProb;

      fakeProb = Math.min(1, Math.max(0, fakeProb));
      realProb = Math.min(1, Math.max(0, realProb));
      const sum = fakeProb + realProb;
      if (sum > 0) {
        fakeProb = Number((fakeProb / sum).toFixed(4));
        realProb = Number((1 - fakeProb).toFixed(4));
      } else {
        fakeProb = 0.5;
        realProb = 0.5;
      }

      const rawVerdict = String(parsed.verdict ?? '').toLowerCase();
      let verdict: 'ai_generated' | 'authentic' | 'inconclusive' = 'inconclusive';
      if (rawVerdict === 'ai_generated' || (rawVerdict !== 'authentic' && fakeProb >= 0.6)) {
        verdict = 'ai_generated';
      } else if (rawVerdict === 'authentic' || realProb >= 0.6) {
        verdict = 'authentic';
      }

      const rawConfidence = String(parsed.confidenceLevel ?? '');
      const confidenceLevel: 'High' | 'Moderate' | 'Low' =
        rawConfidence === 'High' || rawConfidence === 'Moderate' || rawConfidence === 'Low'
          ? rawConfidence
          : Math.max(fakeProb, realProb) >= 0.8
            ? 'High'
            : Math.max(fakeProb, realProb) >= 0.62
              ? 'Moderate'
              : 'Low';

      const indicators = Array.isArray(parsed.indicators)
        ? (parsed.indicators as RawIndicator[])
            .slice(0, 5)
            .map((item) => {
              const rawSig = String(item.signal ?? '').toLowerCase();
              const signal: 'synthetic' | 'authentic' | 'neutral' =
                rawSig === 'synthetic' || rawSig === 'authentic' ? rawSig : 'neutral';
              return {
                category: String(item.category ?? 'Visual Cue'),
                observation: String(item.observation ?? ''),
                signal,
              };
            })
            .filter((item) => item.observation.length > 0)
        : [];

      res.json({
        fileName,
        verdict,
        fakeProbability: fakeProb,
        realProbability: realProb,
        confidenceLevel,
        domainCategory: String(parsed.domainCategory ?? 'Object / Scene'),
        summary: String(parsed.summary ?? 'Forensic visual inspection completed.'),
        indicators,
        analyzedAt: new Date().toISOString(),
        engine: 'gemini-4-hydrogen',
        engineMode: 'gemini',
      });
    } catch (error) {
      const { statusCode, message } = extractReadableError(error);
      console.error('[/api/detect] Error:', message);
      res.status(statusCode).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('/{*splat}', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Veritas server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
