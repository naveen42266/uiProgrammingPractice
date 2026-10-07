import { useRef, useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Camera, RefreshCw, Ruler, Crosshair, CheckCircle2, HelpCircle, Cpu, Box } from 'lucide-react';

interface Preset {
  label: string;
  category: 'tyre' | 'reference' | 'gadget';
  widthCm: number;
  description: string;
}

const PRESETS: Preset[] = [
  { label: 'Tyre Tread (205 mm)', category: 'tyre', widthCm: 20.5, description: '205 mm section tyre contact width' },
  { label: 'Tyre Tread (225 mm)', category: 'tyre', widthCm: 22.5, description: '225 mm section tyre contact width' },
  { label: 'Tyre Tread (195 mm)', category: 'tyre', widthCm: 19.5, description: '195 mm section tyre contact width' },
  { label: 'Credit / ID Card', category: 'reference', widthCm: 8.56, description: 'Standard ISO card width (85.6 mm)' },
  { label: 'Smartphone', category: 'gadget', widthCm: 7.5, description: 'Average phone width (~7.5 cm)' },
  { label: 'Water Bottle', category: 'gadget', widthCm: 7.0, description: 'Standard drink bottle diameter (~7 cm)' },
];

export default function DistanceEstimator() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Model & Camera states
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [modelLoading, setModelLoading] = useState<boolean>(true);
  const [cameraFacing, setCameraFacing] = useState<'environment' | 'user'>('environment');
  const [cameraReady, setCameraReady] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Detection states
  const [predictions, setPredictions] = useState<cocoSsd.DetectedObject[]>([]);
  const [selectedObjIndex, setSelectedObjIndex] = useState<number>(0);

  // Target object real dimensions
  const [targetWidthCm, setTargetWidthCm] = useState<number>(20.5);

  // Focal length (px)
  const [focalLength, setFocalLength] = useState<number>(() => {
    const saved = localStorage.getItem('ai_camera_focal_length');
    return saved ? parseFloat(saved) : 600;
  });

  // Smoothed distance state
  const [smoothedDistanceCm, setSmoothedDistanceCm] = useState<number>(40);

  // Calibration tool state
  const [showCalibration, setShowCalibration] = useState<boolean>(false);
  const [knownDistInput, setKnownDistInput] = useState<number>(30); // 30 cm default calibration distance
  const [calibratedSuccess, setCalibratedSuccess] = useState<boolean>(false);

  // 1. Initialize TensorFlow.js and Load COCO-SSD Model (Client-Side)
  useEffect(() => {
    let isMounted = true;
    async function loadModel() {
      try {
        setModelLoading(true);
        await tf.ready();
        // Load in-browser quantized MobileNet COCO-SSD
        const loadedModel = await cocoSsd.load({ base: 'mobilenet_v2' });
        if (isMounted) {
          setModel(loadedModel);
          setModelLoading(false);
        }
      } catch (err) {
        console.error('Failed to load in-browser AI model:', err);
        if (isMounted) {
          setErrorMsg('Failed to load TensorFlow.js model in browser.');
          setModelLoading(false);
        }
      }
    }
    loadModel();
    return () => {
      isMounted = false;
    };
  }, []);

  // 2. Initialize Camera Stream
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
            const v = videoRef.current;
            if (v) {
              setCameraReady(true);
              // Auto-calibrate default focal length to native camera resolution
              // Using standard mobile/webcam ~70° HFOV: F = width / (2 * tan(35°)) ≈ width * 0.714
              const savedF = localStorage.getItem('ai_camera_focal_length');
              if (!savedF && v.videoWidth) {
                const autoF = Math.round(v.videoWidth * 0.714);
                setFocalLength(autoF);
              }
            }
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

  // 3. In-Browser Real-Time AI Detection Loop
  useEffect(() => {
    if (!model || !cameraReady) return;

    let animFrameId: number;
    let isDetecting = false;

    const detectFrame = async () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2 && !isDetecting) {
        isDetecting = true;
        try {
          // Detect objects directly on the HTMLVideoElement
          const results = await model.detect(video, 5, 0.3);
          setPredictions(results);
        } catch (err) {
          console.warn('Detection frame drop:', err);
        } finally {
          isDetecting = false;
        }
      }
      animFrameId = requestAnimationFrame(detectFrame);
    };

    animFrameId = requestAnimationFrame(detectFrame);
    return () => cancelAnimationFrame(animFrameId);
  }, [model, cameraReady]);

  // Selected object tracking
  const activePrediction = predictions[selectedObjIndex] || predictions[0] || null;

  // Compute detected pixel width on the camera sensor
  const detectedPixelWidth = activePrediction ? activePrediction.bbox[2] : 0;

  // Triangle Similarity Distance Calculation: Distance = (Real Width * Focal Length) / Pixel Width
  useEffect(() => {
    if (detectedPixelWidth > 5 && targetWidthCm > 0) {
      const calculated = (targetWidthCm * focalLength) / detectedPixelWidth;
      if (!isNaN(calculated) && isFinite(calculated)) {
        // Exponential Moving Average filter to smooth distance readings
        setSmoothedDistanceCm((prev) => prev * 0.7 + calculated * 0.3);
      }
    }
  }, [detectedPixelWidth, targetWidthCm, focalLength]);

  // 1-Click Calibration at known distance (e.g., 30cm)
  const handleCalibrate = useCallback(() => {
    if (!detectedPixelWidth || knownDistInput <= 0 || targetWidthCm <= 0) return;
    // F = (Pixel Width * Known Distance) / Real Target Width
    const calculatedF = Math.round((detectedPixelWidth * knownDistInput) / targetWidthCm);
    setFocalLength(calculatedF);
    localStorage.setItem('ai_camera_focal_length', calculatedF.toString());
    setCalibratedSuccess(true);
    setTimeout(() => {
      setCalibratedSuccess(false);
      setShowCalibration(false);
    }, 1800);
  }, [detectedPixelWidth, knownDistInput, targetWidthCm]);

  const handleResetCalibration = () => {
    const defaultF = videoRef.current?.videoWidth ? Math.round(videoRef.current.videoWidth * 0.714) : 600;
    setFocalLength(defaultF);
    localStorage.removeItem('ai_camera_focal_length');
  };

  const videoNativeW = videoRef.current?.videoWidth || 1280;
  const videoNativeH = videoRef.current?.videoHeight || 720;

  return (
    <div className="flex flex-col items-center min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 select-none font-sans">
      {/* Header */}
      <header className="w-full max-w-2xl flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Cpu className="w-6 h-6 text-emerald-400" />
          <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white">
            AI Object Distance Meter
          </h1>
        </div>

        <div className="flex items-center gap-2">
          {modelLoading ? (
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-amber-950/60 border border-amber-500/50 text-amber-300">
              <Cpu className="w-3.5 h-3.5 animate-spin" /> Loading AI Model...
            </span>
          ) : (
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-emerald-950/60 border border-emerald-500/50 text-emerald-300">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" /> AI Online
            </span>
          )}

          <button
            onClick={() => setCameraFacing((prev) => (prev === 'environment' ? 'user' : 'environment'))}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700 transition"
            title="Switch front/rear camera"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </header>

      {/* Main Viewport Container */}
      <div className="w-full max-w-2xl flex flex-col gap-3">
        {/* Camera Live Viewport + AI Bounding Box Overlay */}
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
              <Camera className="w-6 h-6 animate-pulse mr-2" /> Initializing Camera Stream...
            </div>
          )}

          {/* Render All AI Detected Bounding Boxes */}
          {predictions.map((pred, idx) => {
            const [x, y, width, height] = pred.bbox;
            // Normalize to percentages (0 to 100%)
            const leftPct = (x / videoNativeW) * 100;
            const topPct = (y / videoNativeH) * 100;
            const widthPct = (width / videoNativeW) * 100;
            const heightPct = (height / videoNativeH) * 100;
            const isSelected = idx === selectedObjIndex || (predictions.length === 1 && idx === 0);

            return (
              <div
                key={`${pred.class}-${idx}`}
                onClick={() => setSelectedObjIndex(idx)}
                style={{
                  left: `${leftPct}%`,
                  top: `${topPct}%`,
                  width: `${widthPct}%`,
                  height: `${heightPct}%`,
                }}
                className={`absolute cursor-pointer pointer-events-auto transition-all duration-75 rounded-lg ${
                  isSelected
                    ? 'border-2 border-emerald-400 bg-emerald-500/15 shadow-[0_0_15px_rgba(52,211,153,0.5)] z-20'
                    : 'border border-cyan-400/60 bg-cyan-500/10 z-10'
                }`}
              >
                {/* Object Tag Badge */}
                <div
                  className={`absolute -top-6 left-0 px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase flex items-center gap-1 shadow-md whitespace-nowrap ${
                    isSelected
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-900/90 border border-cyan-500/50 text-cyan-300'
                  }`}
                >
                  <Box className="w-3 h-3" />
                  <span>
                    {pred.class} ({Math.round(pred.score * 100)}%)
                  </span>
                  {isSelected && <span className="ml-1 text-[9px] bg-emerald-800 px-1 rounded">LOCKED</span>}
                </div>

                {/* Corner reticles for selected box */}
                {isSelected && (
                  <>
                    <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-emerald-300" />
                    <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-emerald-300" />
                    <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-emerald-300" />
                    <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-emerald-300" />
                  </>
                )}
              </div>
            );
          })}

          {/* Central Aiming Reticle */}
          {predictions.length === 0 && cameraReady && !modelLoading && (
            <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center z-10">
              <Crosshair className="w-12 h-12 text-slate-500/50 animate-pulse" />
              <span className="text-xs text-slate-400 bg-black/60 px-3 py-1 rounded-full mt-2">
                Point camera at tyre, car, or target object
              </span>
            </div>
          )}

          {/* Top Real-time Distance HUD Badge */}
          <div className="absolute top-3 left-3 z-30 bg-slate-950/90 backdrop-blur-md border border-emerald-500/50 px-3.5 py-1.5 rounded-xl shadow-lg flex items-center gap-2.5">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-semibold text-emerald-300 tracking-wider">
                {activePrediction ? `Distance to ${activePrediction.class}` : 'Distance'}
              </span>
              <span className="text-lg md:text-xl font-black font-mono text-white leading-tight">
                {detectedPixelWidth > 0 ? (
                  smoothedDistanceCm < 100 ? (
                    `${smoothedDistanceCm.toFixed(1)} cm`
                  ) : (
                    `${(smoothedDistanceCm / 100).toFixed(2)} m`
                  )
                ) : (
                  '--'
                )}
              </span>
            </div>
            {detectedPixelWidth > 0 && (
              <span className="text-xs text-slate-400 border-l border-slate-700 pl-2">
                {Math.round(detectedPixelWidth)} px
              </span>
            )}
          </div>
        </div>

        {/* Live Distance Output Card */}
        <div className="grid grid-cols-3 gap-3 bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-lg">
          <div className="col-span-2 flex flex-col justify-center">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Ruler className="w-4 h-4 text-emerald-400" />
              Real-time Measurement
            </span>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-4xl md:text-5xl font-black text-emerald-400 font-mono tracking-tight">
                {detectedPixelWidth > 0 ? (
                  smoothedDistanceCm < 100 ? (
                    smoothedDistanceCm.toFixed(1)
                  ) : (
                    (smoothedDistanceCm / 100).toFixed(2)
                  )
                ) : (
                  '--'
                )}
              </span>
              <span className="text-lg md:text-xl font-bold text-slate-300">
                {detectedPixelWidth > 0 ? (smoothedDistanceCm < 100 ? 'cm' : 'meters') : ''}
              </span>
            </div>
            <span className="text-xs text-slate-400 mt-1">
              {detectedPixelWidth > 0 ? (
                <>
                  ≈ {(smoothedDistanceCm / 2.54).toFixed(1)} inches (
                  {smoothedDistanceCm < 100
                    ? `${(smoothedDistanceCm * 10).toFixed(0)} mm`
                    : `${(smoothedDistanceCm / 100).toFixed(2)} m`}
                  )
                </>
              ) : (
                'Waiting for object to be detected...'
              )}
            </span>
          </div>

          <div className="col-span-1 border-l border-slate-800 pl-4 flex flex-col justify-center">
            <span className="text-[11px] text-slate-400 font-medium">Calibrated Focal Length</span>
            <span className="text-base font-bold font-mono text-slate-200 mt-0.5">{focalLength} px</span>
            <button
              onClick={() => setShowCalibration(!showCalibration)}
              className="mt-2 text-xs font-semibold text-amber-400 hover:text-amber-300 underline text-left"
            >
              {showCalibration ? 'Close Calibrate' : 'Calibrate Lens'}
            </button>
          </div>
        </div>

        {/* Lens Calibration Box (Collapsible) */}
        {showCalibration && (
          <div className="bg-amber-950/30 border border-amber-600/40 p-4 rounded-xl flex flex-col gap-3 transition">
            <div className="flex items-center gap-2 text-amber-300 font-semibold text-sm">
              <HelpCircle className="w-4 h-4" /> 1-Step Accuracy Calibration
            </div>
            <p className="text-xs text-amber-200/80 leading-relaxed">
              Why was distance previously doubled? Every camera lens has a different focal length.
              Place the object at an exact known distance (e.g., 30 cm) from the camera, verify the AI detects it,
              and click <strong>Save Calibration</strong>.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <label className="text-xs text-amber-200 flex items-center gap-1.5">
                Known Distance (cm):
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
                disabled={!detectedPixelWidth}
                className={`font-bold text-xs px-4 py-1.5 rounded-lg transition flex items-center gap-1.5 ${
                  detectedPixelWidth
                    ? 'bg-amber-500 hover:bg-amber-400 text-slate-950 active:scale-95'
                    : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                }`}
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
              Target Real Width Configuration
            </span>
            <span className="text-xs text-emerald-400 font-mono font-bold">
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
                    ? 'bg-emerald-950/60 border-emerald-400 shadow-md shadow-emerald-950'
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
            <span className="text-xs text-slate-400 font-medium">Custom Real Width:</span>
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