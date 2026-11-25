export interface ComputedImageSignals {
  width: number;
  height: number;
  aspectRatio: string;
  fileSizeBytes: number;
  mimeType: string;
  fileName: string;
  rgbMeans: [number, number, number];
  rgbStds: [number, number, number];
  edgeDensity: number;
  laplacianVariance: number;
  resized128DataUrl: string;
  edgeMapDataUrl: string;
  uploadBase64: string;
  uploadMimeType: string;
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to decode image file.'));
    img.src = src;
  });
}

export async function analyzeImagePixels(
  fileOrUrl: File | { url: string; name: string }
): Promise<{ previewUrl: string; signals: ComputedImageSignals }> {
  let previewUrl = '';
  let fileName = '';
  let fileSizeBytes = 0;
  let mimeType = 'image/jpeg';

  if (fileOrUrl instanceof File) {
    fileName = fileOrUrl.name;
    fileSizeBytes = fileOrUrl.size;
    mimeType = fileOrUrl.type || 'image/jpeg';
    previewUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error('Could not read uploaded file.'));
      reader.readAsDataURL(fileOrUrl);
    });
  } else {
    fileName = fileOrUrl.name;
    const response = await fetch(fileOrUrl.url);
    if (!response.ok) {
      throw new Error('Could not load sample image.');
    }
    const blob = await response.blob();
    fileSizeBytes = blob.size;
    mimeType = blob.type || 'image/jpeg';
    previewUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error('Could not decode sample image.'));
      reader.readAsDataURL(blob);
    });
  }

  const img = await loadImageElement(previewUrl);
  const width = img.naturalWidth;
  const height = img.naturalHeight;
  const divisor = gcd(width, height) || 1;
  const aspectRatio = `${Math.round(width / divisor)}:${Math.round(height / divisor)}`;

  // 1. Generate 128x128 resize preview matching deepfake_detection/utils.py T.Resize((128, 128))
  const canvas128 = document.createElement('canvas');
  canvas128.width = 128;
  canvas128.height = 128;
  const ctx128 = canvas128.getContext('2d');
  if (!ctx128) {
    throw new Error('Canvas 2D context unavailable.');
  }
  ctx128.drawImage(img, 0, 0, 128, 128);
  const resized128DataUrl = canvas128.toDataURL('image/png');

  // 2. Compute RGB channel stats & Sobel/Laplacian high-frequency signal on analysis canvas (max 384px)
  const maxAnalysisDim = 384;
  const scale = Math.min(1, maxAnalysisDim / Math.max(width, height));
  const aw = Math.max(32, Math.round(width * scale));
  const ah = Math.max(32, Math.round(height * scale));

  const analysisCanvas = document.createElement('canvas');
  analysisCanvas.width = aw;
  analysisCanvas.height = ah;
  const actx = analysisCanvas.getContext('2d');
  if (!actx) {
    throw new Error('Canvas 2D context unavailable.');
  }
  actx.drawImage(img, 0, 0, aw, ah);
  const imgData = actx.getImageData(0, 0, aw, ah);
  const data = imgData.data;
  const nPixels = aw * ah;

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  const gray = new Float32Array(nPixels);

  for (let i = 0; i < nPixels; i++) {
    const idx = i * 4;
    const r = data[idx] / 255;
    const g = data[idx + 1] / 255;
    const b = data[idx + 2] / 255;
    sumR += r;
    sumG += g;
    sumB += b;
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  const meanR = sumR / nPixels;
  const meanG = sumG / nPixels;
  const meanB = sumB / nPixels;

  let varR = 0;
  let varG = 0;
  let varB = 0;
  for (let i = 0; i < nPixels; i++) {
    const idx = i * 4;
    varR += Math.pow(data[idx] / 255 - meanR, 2);
    varG += Math.pow(data[idx + 1] / 255 - meanG, 2);
    varB += Math.pow(data[idx + 2] / 255 - meanB, 2);
  }

  // Sobel edge visualization & Laplacian variance
  const edgeCanvas = document.createElement('canvas');
  edgeCanvas.width = aw;
  edgeCanvas.height = ah;
  const ectx = edgeCanvas.getContext('2d');
  if (!ectx) {
    throw new Error('Canvas 2D context unavailable.');
  }
  const edgeImageData = ectx.createImageData(aw, ah);
  const ed = edgeImageData.data;

  let edgeSum = 0;
  let lapSum = 0;
  let lapSqSum = 0;
  let interiorCount = 0;

  for (let y = 1; y < ah - 1; y++) {
    for (let x = 1; x < aw - 1; x++) {
      const c = y * aw + x;
      const gx =
        -gray[c - aw - 1] +
        gray[c - aw + 1] -
        2 * gray[c - 1] +
        2 * gray[c + 1] -
        gray[c + aw - 1] +
        gray[c + aw + 1];
      const gy =
        -gray[c - aw - 1] -
        2 * gray[c - aw] -
        gray[c - aw + 1] +
        gray[c + aw - 1] +
        2 * gray[c + aw] +
        gray[c + aw + 1];

      const mag = Math.min(1, Math.sqrt(gx * gx + gy * gy) * 1.4);
      edgeSum += mag;

      const lap =
        gray[c - aw] + gray[c + aw] + gray[c - 1] + gray[c + 1] - 4 * gray[c];
      lapSum += lap;
      lapSqSum += lap * lap;
      interiorCount++;

      // Map magnitude into crisp blue-white lab luminance scale on dark slate
      const o = c * 4;
      const intensity = Math.round(mag * 255);
      ed[o] = Math.round(intensity * 0.25);
      ed[o + 1] = Math.round(intensity * 0.68);
      ed[o + 2] = Math.min(255, Math.round(intensity * 1.15 + 16));
      ed[o + 3] = 255;
    }
  }
  ectx.putImageData(edgeImageData, 0, 0);
  const edgeMapDataUrl = edgeCanvas.toDataURL('image/png');

  const lapMean = interiorCount > 0 ? lapSum / interiorCount : 0;
  const laplacianVariance =
    interiorCount > 0 ? Math.max(0, lapSqSum / interiorCount - lapMean * lapMean) : 0;
  const edgeDensity = interiorCount > 0 ? edgeSum / interiorCount : 0;

  // 3. Prepare bounded upload image (<= 1400px max dimension) for reliable server transfer
  const maxUploadDim = 1400;
  let uploadBase64 = previewUrl.split(',')[1] || '';
  let uploadMimeType = mimeType === 'image/png' || mimeType === 'image/webp' ? mimeType : 'image/jpeg';

  if (Math.max(width, height) > maxUploadDim || fileSizeBytes > 3.5 * 1024 * 1024) {
    const upScale = Math.min(1, maxUploadDim / Math.max(width, height));
    const upCanvas = document.createElement('canvas');
    upCanvas.width = Math.round(width * upScale);
    upCanvas.height = Math.round(height * upScale);
    const uctx = upCanvas.getContext('2d');
    if (uctx) {
      uctx.drawImage(img, 0, 0, upCanvas.width, upCanvas.height);
      const compressed = upCanvas.toDataURL('image/jpeg', 0.9);
      uploadBase64 = compressed.split(',')[1] || uploadBase64;
      uploadMimeType = 'image/jpeg';
    }
  }

  return {
    previewUrl,
    signals: {
      width,
      height,
      aspectRatio,
      fileSizeBytes,
      mimeType,
      fileName,
      rgbMeans: [meanR, meanG, meanB],
      rgbStds: [Math.sqrt(varR / nPixels), Math.sqrt(varG / nPixels), Math.sqrt(varB / nPixels)],
      edgeDensity,
      laplacianVariance,
      resized128DataUrl,
      edgeMapDataUrl,
      uploadBase64,
      uploadMimeType,
    },
  };
}
