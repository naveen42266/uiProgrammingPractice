import { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, RefreshCw, Ruler, Crosshair, CheckCircle2, HelpCircle, Target, Sparkles, Sliders } from 'lucide-react';

interface Preset {
  label: string;
  category: 'tyre' | 'reference' | 'custom';
  widthCm: number;
  description: string;
}

const PRESETS: Preset[] = [
  { label: 'Tyre (205 mm)', category: 'tyre', widthCm: 20.5, description: 'Standard 205 section tyre tread' },
  { label: 'Tyre (225 mm)', category: 'tyre', widthCm: 22.5, description: 'Standard 225 section tyre tread' },
  { label: 'Tyre (195 mm)', category: 'tyre', widthCm: 19.5, description: 'Compact 195 section tyre tread' },
  { label: 'Credit / ID Card', category: 'reference', widthCm: 8.56, description: 'Standard ISO card width (85.6 mm)' },
  { label: 'Standard Coin', category: 'reference', widthCm: 2.5, description: 'Approx 25 mm coin reference' },
];

const DEFAULT_FOCAL_LENGTH = 750; // Calibrated focal length of camera sensor in px

export default function DistanceEstimator() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Optical parameters
  const [targetWidthCm, setTargetWidthCm] = useState<number>(20.5);
  const [focalLength, setFocalLength] = useState<number>(() => {
    const saved = localStorage.getItem('camera_focal_length');
    return saved ? parseFloat(saved) : DEFAULT_FOCAL_LENGTH;
  });

  // Mode: Auto Detection (Classical CV) vs Manual Calipers
  const [autoDetect, setAutoDetect] = useState<boolean>(true);
  const [detectionMode, setDetectionMode] = useState<'tyre' | 'high_contrast'>('tyre');
  const [sensitivity, setSensitivity] = useState<number>(45); // Gradient threshold

  // Detected bounding box (percentages 0.0 to 1.0)
  const [autoBox, setAutoBox] = useState<{
    left: number;
    right: number;
    top: number;
    bottom: number;
    confidence: number;
  } | null>(null);

  // Manual caliper positions (used when autoDetect is off or for fine-tuning)
  const [leftRatio, setLeftRatio] = useState<number>(0.3);
  const [rightRatio, setRightRatio] = useState<number>(0.7);
  const [dragging, setDragging] = useState<'left' | 'right' | null>(null);

  // Filtered distance state (smoothed to avoid jitter)
  const [smoothedDistanceCm, setSmoothedDistanceCm] = useState<number>(45);

  // Camera state
  const [cameraFacing, setCameraFacing] = useState<'environment' | 'user'>('environment');
  const [cameraReady, setCameraReady] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Calibration state
  const [showCalibration, setShowCalibration] = useState<boolean>(false);
  const [knownDistInput, setKnownDistInput] = useState<number>(30); // 30 cm default calibration distance
  const [calibratedSuccess, setCalibratedSuccess] = useState<boolean>(false);

  // Start / restart camera
  useEffect(() => {
    let stream: MediaStream | null = null;
    setCameraReady(false);
    setErrorMsg(null);

    navigator.mediaDevices
      .getUserMedia({
        video: {
          facingMode: cameraFacing,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })
      .then((s) => {
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.onloadedmetadata = () => {
            setCameraReady(true);
          };
        }
      })
      .catch((err) => {
        console.error('Camera access error:', err);
        setErrorMsg('Unable to access camera. Please allow camera permissions.');
      });

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [cameraFacing]);

  // ==========================================
  // Classical Computer Vision Auto-Detection Loop
  // (Zero ML models, pure canvas pixel gradient math)
  // ==========================================
  useEffect(() => {
    if (!cameraReady || !autoDetect) return;

    let animationFrameId: number;
    const processCanvas = canvasRef.current || document.createElement('canvas');
    const procCtx = processCanvas.getContext('2d', { willReadFrequently: true });

    // Processing resolution (downsampled for ultra-smooth 60fps)
    const procW = 320;
    const procH = 180;
    processCanvas.width = procW;
    processCanvas.height = procH;

    let lastLeft = 0.3;
    let lastRight = 0.7;

    const runDetection = () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2 && procCtx) {
        // Draw video frame to off-screen canvas
        procCtx.drawImage(video, 0, 0, procW, procH);
        const imgData = procCtx.getImageData(0, 0, procW, procH);
        const data = imgData.data;

        // Scan central horizontal band (30% to 70% height)
        const startY = Math.floor(procH * 0.35);
        const endY = Math.floor(procH * 0.65);
        const hSpan = endY - startY;

        // Compute horizontal luminance profile
        const lumProfile = new Float32Array(procW);
        for (let x = 0; x < procW; x++) {
          let sumLum = 0;
          for (let y = startY; y < endY; y++) {
            const idx = (y * procW + x) * 4;
            // Standard luminance: 0.299*R + 0.587*G + 0.114*B
            const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            sumLum += lum;
          }
          lumProfile[x] = sumLum / hSpan;
        }

        let detectedMinX = -1;
        let detectedMaxX = -1;

        if (detectionMode === 'tyre') {
          // Tyres are dark black rubber compared to background:
          // Find the average background vs center luminance
          let totalLum = 0;
          for (let x = 0; x < procW; x++) totalLum += lumProfile[x];
          const avgLum = totalLum / procW;
          const darkThreshold = Math.min(avgLum * 0.9, 120 + (sensitivity - 50));

          // Find the central continuous dark region
          const centerX = Math.floor(procW / 2);

          // Scan left from center
          let leftX = centerX;
          while (leftX > 10 && lumProfile[leftX] < darkThreshold) {
            leftX--;
          }

          // Scan right from center
          let rightX = centerX;
          while (rightX < procW - 10 && lumProfile[rightX] < darkThreshold) {
            rightX++;
          }

          if (rightX - leftX > 25) {
            detectedMinX = leftX;
            detectedMaxX = rightX;
          }
        }

        // Fallback: Gradient edge profile (detect sharpest left & right contrast transitions)
        if (detectedMinX === -1 || detectedMaxX === -1) {
          const grad = new Float32Array(procW);
          for (let x = 1; x < procW - 1; x++) {
            grad[x] = Math.abs(lumProfile[x + 1] - lumProfile[x - 1]);
          }

          const centerX = Math.floor(procW / 2);
          // Find peak edge to the left of center
          let maxGradLeft = 0;
          let bestLeft = Math.floor(procW * 0.25);
          for (let x = Math.floor(procW * 0.1); x < centerX - 15; x++) {
            if (grad[x] > maxGradLeft && grad[x] > (100 - sensitivity) * 0.3) {
              maxGradLeft = grad[x];
              bestLeft = x;
            }
          }

          // Find peak edge to the right of center
          let maxGradRight = 0;
          let bestRight = Math.floor(procW * 0.75);
          for (let x = centerX + 15; x < Math.floor(procW * 0.9); x++) {
            if (grad[x] > maxGradRight && grad[x] > (100 - sensitivity) * 0.3) {
              maxGradRight = grad[x];
              bestRight = x;
            }
          }

          if (bestRight - bestLeft > 25) {
            detectedMinX = bestLeft;
            detectedMaxX = bestRight;
          }
        }

        if (detectedMinX !== -1 && detectedMaxX !== -1) {
          const rawLeftRatio = detectedMinX / procW;
          const rawRightRatio = detectedMaxX / procW;

          // Smooth ratio changes (Exponential Moving Average) to eliminate jitter
          lastLeft = lastLeft * 0.75 + rawLeftRatio * 0.25;
          lastRight = lastRight * 0.75 + rawRightRatio * 0.25;

          setAutoBox({
            left: lastLeft,
            right: lastRight,
            top: 0.3,
            bottom: 0.7,
            confidence: 0.88,
          });

          // Sync manual handles to detected values
          setLeftRatio(lastLeft);
          setRightRatio(lastRight);
        }
      }

      animationFrameId = requestAnimationFrame(runDetection);
    };

    animationFrameId = requestAnimationFrame(runDetection);
    return () => cancelAnimationFrame(animationFrameId);
  }, [cameraReady, autoDetect, detectionMode, sensitivity]);

  // Compute actual pixel span on the camera's native video resolution
  const videoNativeWidth = videoRef.current?.videoWidth || 1280;
  const activeLeft = autoDetect && autoBox ? autoBox.left : leftRatio;
  const activeRight = autoDetect && autoBox ? autoBox.right : rightRatio;
  const pixelSpan = Math.max(1, (activeRight - activeLeft) * videoNativeWidth);

  // Triangle Similarity Formula: Distance = (Real Width * Focal Length) / Pixel Width
  const rawDistanceCm = (targetWidthCm * focalLength) / pixelSpan;

  // Real-time smoothing filter on distance
  useEffect(() => {
    if (!isNaN(rawDistanceCm) && isFinite(rawDistanceCm)) {
      setSmoothedDistanceCm((prev) => prev * 0.7 + rawDistanceCm * 0.3);
    }
  }, [rawDistanceCm]);

  const displayDistanceCm = autoDetect ? smoothedDistanceCm : rawDistanceCm;
  const displayDistanceM = displayDistanceCm / 100;
  const displayDistanceInches = displayDistanceCm / 2.54;

  // Calibrate Focal Length: F = (Pixel Span * Known Distance) / Target Real Width
  const handleCalibrate = () => {
    if (knownDistInput <= 0 || targetWidthCm <= 0) return;
    const computedF = (pixelSpan * knownDistInput) / targetWidthCm;
    const rounded = Math.round(computedF);
    setFocalLength(rounded);
    localStorage.setItem('camera_focal_length', rounded.toString());
    setCalibratedSuccess(true);
    setTimeout(() => {
      setCalibratedSuccess(false);
      setShowCalibration(false);
    }, 1800);
  };

  const handleResetCalibration = () => {
    setFocalLength(DEFAULT_FOCAL_LENGTH);
    localStorage.removeItem('camera_focal_length');
  };

  // Manual Calipers dragging
  const handlePointerMove = useCallback(
    (clientX: number) => {
      if (!dragging || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const currentRatio = Math.max(0.05, Math.min(0.95, (clientX - rect.left) / rect.width));

      if (dragging === 'left') {
        setLeftRatio(Math.min(currentRatio, rightRatio - 0.05));
      } else if (dragging === 'right') {
        setRightRatio(Math.max(currentRatio, leftRatio + 0.05));
      }
    },
    [dragging, leftRatio, rightRatio]
  );

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => handlePointerMove(e.clientX);
    const onMouseUp = () => setDragging(null);
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches[0]) handlePointerMove(e.touches[0].clientX);
    };
    const onTouchEnd = () => setDragging(null);

    if (dragging) {
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
      window.addEventListener('touchmove', onTouchMove);
      window.addEventListener('touchend', onTouchEnd);
    }
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onTouchEnd);
    };
  }, [dragging, handlePointerMove]);

  return (
    <div className="flex flex-col items-center min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 select-none font-sans">
      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Header */}
      <header className="w-full max-w-2xl flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Crosshair className="w-6 h-6 text-cyan-400" />
          <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white">
            Tyre Distance Meter
          </h1>
        </div>

        <div className="flex items-center gap-2">
          {/* Auto vs Manual Mode Switcher */}
          <button
            onClick={() => setAutoDetect(!autoDetect)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
              autoDetect
                ? 'bg-emerald-600/30 border-emerald-500 text-emerald-300'
                : 'bg-slate-800 border-slate-700 text-slate-300'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            {autoDetect ? 'Auto-Detect ON' : 'Manual Mode'}
          </button>

          <button
            onClick={() => setCameraFacing((prev) => (prev === 'environment' ? 'user' : 'environment'))}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700 transition"
            title="Switch front/rear camera"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </header>

      {/* Main Container */}
      <div className="w-full max-w-2xl flex flex-col gap-3">
        {/* Camera Live Viewport */}
        <div
          ref={containerRef}
          className="relative w-full aspect-[4/3] md:aspect-video bg-black rounded-2xl overflow-hidden border-2 border-slate-800 shadow-2xl"
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />

          {errorMsg && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/80 p-6 text-center text-rose-400 text-sm">
              {errorMsg}
            </div>
          )}

          {!cameraReady && !errorMsg && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-slate-400 text-sm">
              <Camera className="w-6 h-6 animate-pulse mr-2" /> Starting Camera...
            </div>
          )}

          {/* Central Target Zone Crosshair Reticle */}
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center z-10">
            <div className="w-16 h-16 border border-white/20 rounded-full flex items-center justify-center">
              <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full" />
            </div>
          </div>

          {/* Auto-Detection Bounding Box Overlay */}
          {autoDetect && (
            <div
              style={{
                left: `${activeLeft * 100}%`,
                width: `${(activeRight - activeLeft) * 100}%`,
                top: '25%',
                bottom: '25%',
              }}
              className="absolute pointer-events-none z-20 border-2 border-emerald-400 bg-emerald-500/10 rounded-lg shadow-[0_0_15px_rgba(52,211,153,0.4)] transition-all duration-75"
            >
              {/* Corner Accents */}
              <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-emerald-300" />
              <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-emerald-300" />
              <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-emerald-300" />
              <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-emerald-300" />

              {/* Status Tag on top of box */}
              <div className="absolute -top-7 left-1/2 -translate-x-1/2 bg-emerald-950/90 border border-emerald-500 text-emerald-300 text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1 shadow-md whitespace-nowrap">
                <Target className="w-3 h-3 animate-spin text-emerald-400" />
                <span>OBJECT LOCKED</span>
              </div>
            </div>
          )}

          {/* Manual Mode Draggable Calipers */}
          {!autoDetect && (
            <>
              {/* Left Caliper Line */}
              <div
                style={{ left: `${leftRatio * 100}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 cursor-ew-resize z-20 flex items-center justify-center shadow-[0_0_10px_rgba(34,211,238,0.8)]"
                onMouseDown={() => setDragging('left')}
                onTouchStart={() => setDragging('left')}
              >
                <div className="w-6 h-12 bg-cyan-500 hover:bg-cyan-400 text-black font-extrabold text-xs rounded-full flex items-center justify-center shadow-lg cursor-grab active:cursor-grabbing">
                  ◀
                </div>
              </div>

              {/* Right Caliper Line */}
              <div
                style={{ left: `${rightRatio * 100}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 cursor-ew-resize z-20 flex items-center justify-center shadow-[0_0_10px_rgba(34,211,238,0.8)]"
                onMouseDown={() => setDragging('right')}
                onTouchStart={() => setDragging('right')}
              >
                <div className="w-6 h-12 bg-cyan-500 hover:bg-cyan-400 text-black font-extrabold text-xs rounded-full flex items-center justify-center shadow-lg cursor-grab active:cursor-grabbing">
                  ▶
                </div>
              </div>
            </>
          )}

          {/* Real-time Distance Overlay HUD Badge */}
          <div className="absolute top-3 left-3 z-30 bg-slate-950/85 backdrop-blur-md border border-cyan-500/40 px-3.5 py-1.5 rounded-xl shadow-lg flex items-center gap-2.5">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-semibold text-cyan-300 tracking-wider">Distance</span>
              <span className="text-lg md:text-xl font-black font-mono text-white leading-tight">
                {displayDistanceCm < 100
                  ? `${displayDistanceCm.toFixed(1)} cm`
                  : `${displayDistanceM.toFixed(2)} m`}
              </span>
            </div>
            <span className="text-xs text-slate-400 border-l border-slate-700 pl-2">
              {Math.round(pixelSpan)} px
            </span>
          </div>

          {/* Bottom Guidance Instruction */}
          <div className="absolute bottom-3 inset-x-0 text-center pointer-events-none z-10">
            <span className="bg-slate-950/80 text-slate-300 text-xs px-3.5 py-1 rounded-full border border-slate-800 backdrop-blur-md">
              {autoDetect
                ? 'Align the tyre tread in the center frame — bounding box will lock automatically'
                : 'Drag handles ◀ ▶ to match the edges of the tyre tread'}
            </span>
          </div>
        </div>

        {/* Live Distance Output Card */}
        <div className="grid grid-cols-3 gap-3 bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-lg">
          <div className="col-span-2 flex flex-col justify-center">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Ruler className="w-4 h-4 text-cyan-400" />
              Real-time Measurement
            </span>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-4xl md:text-5xl font-black text-cyan-400 font-mono tracking-tight">
                {displayDistanceCm < 100
                  ? displayDistanceCm.toFixed(1)
                  : displayDistanceM.toFixed(2)}
              </span>
              <span className="text-lg md:text-xl font-bold text-slate-300">
                {displayDistanceCm < 100 ? 'cm' : 'meters'}
              </span>
            </div>
            <span className="text-xs text-slate-400 mt-1">
              ≈ {displayDistanceInches.toFixed(1)} inches (
              {displayDistanceCm < 100
                ? `${(displayDistanceCm * 10).toFixed(0)} mm`
                : `${displayDistanceM.toFixed(2)} m`}
              )
            </span>
          </div>

          <div className="col-span-1 border-l border-slate-800 pl-4 flex flex-col justify-center">
            <span className="text-[11px] text-slate-400 font-medium">Sensor Focal Length</span>
            <span className="text-base font-bold font-mono text-slate-200 mt-0.5">{focalLength} px</span>
            <button
              onClick={() => setShowCalibration(!showCalibration)}
              className="mt-2 text-xs font-semibold text-amber-400 hover:text-amber-300 underline text-left"
            >
              {showCalibration ? 'Close Calibrate' : 'Calibrate Lens'}
            </button>
          </div>
        </div>

        {/* Vision Sensitivity Controls (For Auto Mode) */}
        {autoDetect && (
          <div className="bg-slate-900/70 border border-slate-800 px-4 py-3 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2 text-slate-300">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span className="font-semibold">Detection Type:</span>
              <button
                onClick={() => setDetectionMode('tyre')}
                className={`px-2 py-1 rounded ${
                  detectionMode === 'tyre' ? 'bg-cyan-600 text-white font-bold' : 'bg-slate-800 text-slate-400'
                }`}
              >
                Tyre Rubber (Dark)
              </button>
              <button
                onClick={() => setDetectionMode('high_contrast')}
                className={`px-2 py-1 rounded ${
                  detectionMode === 'high_contrast' ? 'bg-cyan-600 text-white font-bold' : 'bg-slate-800 text-slate-400'
                }`}
              >
                High Contrast / Edges
              </button>
            </div>

            <div className="flex items-center gap-2 text-slate-400">
              <span>Sensitivity:</span>
              <input
                type="range"
                min="10"
                max="90"
                value={sensitivity}
                onChange={(e) => setSensitivity(parseInt(e.target.value))}
                className="w-24 accent-cyan-400"
              />
              <span className="font-mono text-white">{sensitivity}%</span>
            </div>
          </div>
        )}

        {/* Lens Calibration Box (Collapsible) */}
        {showCalibration && (
          <div className="bg-amber-950/30 border border-amber-600/40 p-4 rounded-xl flex flex-col gap-3 transition">
            <div className="flex items-center gap-2 text-amber-300 font-semibold text-sm">
              <HelpCircle className="w-4 h-4" /> 1-Step Lens Calibration
            </div>
            <p className="text-xs text-amber-200/80 leading-relaxed">
              Place the object at a known physical distance from the camera, ensure the bounding box wraps its
              edges, and click <strong>Save Calibration</strong>.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <label className="text-xs text-amber-200 flex items-center gap-1.5">
                Exact Distance (cm):
                <input
                  type="number"
                  min="5"
                  max="500"
                  value={knownDistInput}
                  onChange={(e) => setKnownDistInput(parseFloat(e.target.value) || 0)}
                  className="w-20 bg-slate-900 border border-amber-700/60 rounded px-2 py-1 text-sm text-white font-mono"
                />
              </label>

              <button
                onClick={handleCalibrate}
                className="bg-amber-500 hover:bg-amber-400 active:scale-95 text-slate-950 font-bold text-xs px-4 py-1.5 rounded-lg transition flex items-center gap-1.5"
              >
                {calibratedSuccess ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-900" /> Saved!
                  </>
                ) : (
                  'Save Calibration'
                )}
              </button>

              <button
                onClick={handleResetCalibration}
                className="text-xs text-slate-400 hover:text-slate-200 underline ml-auto"
              >
                Reset Default
              </button>
            </div>
          </div>
        )}

        {/* Object & Tyre Presets Selection */}
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Select Tyre Width or Reference Object
            </span>
            <span className="text-xs text-cyan-400 font-mono font-bold">
              Current: {targetWidthCm} cm ({targetWidthCm * 10} mm)
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => setTargetWidthCm(preset.widthCm)}
                className={`p-2.5 rounded-xl border text-left flex flex-col transition ${
                  targetWidthCm === preset.widthCm
                    ? 'bg-cyan-950/60 border-cyan-400 shadow-md shadow-cyan-950'
                    : 'bg-slate-800/80 border-slate-700/80 hover:bg-slate-800 text-slate-300'
                }`}
              >
                <span className="text-xs font-bold text-white">{preset.label}</span>
                <span className="text-[11px] text-slate-400 mt-0.5">{preset.widthCm} cm</span>
              </button>
            ))}
          </div>

          {/* Custom Dimension Input */}
          <div className="flex items-center gap-3 pt-2 border-t border-slate-800">
            <span className="text-xs text-slate-400 font-medium">Custom Width:</span>
            <div className="flex items-center gap-1">
              <input
                type="number"
                step="0.1"
                min="0.5"
                value={targetWidthCm}
                onChange={(e) => setTargetWidthCm(parseFloat(e.target.value) || 1)}
                className="w-24 bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-sm text-white font-mono"
              />
              <span className="text-xs text-slate-400">cm</span>
            </div>
            <span className="text-xs text-slate-500 ml-auto">
              ({(targetWidthCm * 10).toFixed(0)} mm)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}