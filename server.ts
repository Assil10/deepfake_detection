import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const ALLOWED_MIME_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/gif',
]);

interface DetectRequestBody {
  imageBase64?: unknown;
  mimeType?: unknown;
  fileName?: unknown;
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

  if (parsedMessage.includes('API_KEY_INVALID') || parsedMessage.includes('PERMISSION_DENIED') || parsedCode === 403) {
    return { statusCode: 403, message: parsedMessage };
  }
  if (parsedCode === 429 || parsedMessage.includes('RESOURCE_EXHAUSTED')) {
    return { statusCode: 429, message: parsedMessage };
  }
  if (parsedCode === 503 || parsedMessage.includes('UNAVAILABLE') || parsedMessage.includes('high demand')) {
    return {
      statusCode: 503,
      message: 'The detection model is temporarily busy due to high demand. Please retry in a few seconds.',
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
    res.json({
      ok: true,
      hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    });
  });

  app.post('/api/detect', async (req, res) => {
    try {
      const body = req.body as DetectRequestBody;
      const imageBase64 = typeof body.imageBase64 === 'string' ? body.imageBase64.trim() : '';
      const mimeType = typeof body.mimeType === 'string' ? body.mimeType.trim() : 'image/jpeg';
      const fileName = typeof body.fileName === 'string' ? body.fileName.trim() : 'uploaded-image';

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

      if (!process.env.GEMINI_API_KEY) {
        res.status(503).json({
          error: 'Detection service is missing GEMINI_API_KEY in server configuration.',
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
