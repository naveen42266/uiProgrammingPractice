import { useRef, useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import {
  Camera,
  RefreshCw,
  Ruler,
  Crosshair,
  CheckCircle2,
  Cpu,
  Box,
  ShieldCheck,
  RotateCcw,
  Target,
} from 'lucide-react';

interface Preset {
  label: string;
  category: 'tyre' | 'reference' | 'gadget';
  widthCm: number;
  description: string;
}

interface TwoPointCalibration {
  slopeA: number; // A = (D2 - D1) * P1 * P2 / (P1 - P2)
  offsetB: number; // B = D1 - (A / P1)
  calibWidthCm: number; // Object real width used during calibration
  p1: number;
  d1: number;
  p2: number;
  d2: number;
  timestamp: string;
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
  const pixelHistoryRef = useRef<number[]>([]);

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

  // 2-Point Precision Calibration State
  const [twoPointCalib, setTwoPointCalib] = useState<TwoPointCalibration | null>(() => {
    const saved = localStorage.getItem('two_point_camera_calib');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        return null;
      }
    }
    return null;
  });

  // Fallback single focal length (used if 2-point not calibrated yet)
  const [fallbackFocalLength, setFallbackFocalLength] = useState<number>(() => {
    const saved = localStorage.getItem('ai_camera_focal_length');
    return saved ? parseFloat(saved) : 600;
  });

  // Wizard state for 2-point calibration
  const [showWizard, setShowWizard] = useState<boolean>(false);
  const [wizardStep, setWizardStep] = useState<1 | 2>(1);
  const [dist1Input, setDist1Input] = useState<number>(20); // Point 1: 20 cm
  const [dist2Input, setDist2Input] = useState<number>(50); // Point 2: 50 cm
  const [point1Data, setPoint1Data] = useState<{ p: number; d: number } | null>(null);
  const [point2Data, setPoint2Data] = useState<{ p: number; d: number } | null>(null);
  const [isSampling, setIsSampling] = useState<boolean>(false);
  const [wizardSuccess, setWizardSuccess] = useState<boolean>(false);

  // Distance output
  const [currentDistanceCm, setCurrentDistanceCm] = useState<number | null>(null);
  const [stablePixelWidth, setStablePixelWidth] = useState<number>(0);

  // 1. Initialize TensorFlow.js and Load COCO-SSD Model (Client-Side)
  useEffect(() => {
    let isMounted = true;
    async function loadModel() {
      try {
        setModelLoading(true);
        await tf.ready();
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
              const savedF = localStorage.getItem('ai_camera_focal_length');
              if (!savedF && v.videoWidth) {
                const autoF = Math.round(v.videoWidth * 0.714);
                setFallbackFocalLength(autoF);
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

  // 3. Real-Time AI Detection Loop
  useEffect(() => {
    if (!model || !cameraReady) return;

    let animFrameId: number;
    let isDetecting = false;

    const detectFrame = async () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2 && !isDetecting) {
        isDetecting = true;
        try {
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
  const rawPixelWidth = activePrediction ? activePrediction.bbox[2] : 0;

  // 4. Temporal Median Filter (Eliminates Jitter & Spikes)
  useEffect(() => {
    if (rawPixelWidth > 5) {
      const history = pixelHistoryRef.current;
      history.push(rawPixelWidth);
      if (history.length > 15) history.shift();

      // Compute median of recent frames
      const sorted = [...history].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      const median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

      setStablePixelWidth(median);

      // Distance calculation with 2-Point Calibration vs 1-Point Fallback
      if (twoPointCalib) {
        // D = (A * (W_target / W_calib)) / P + B
        const effectiveA = twoPointCalib.slopeA * (targetWidthCm / twoPointCalib.calibWidthCm);
        const calculatedD = effectiveA / median + twoPointCalib.offsetB;
        if (!isNaN(calculatedD) && isFinite(calculatedD)) {
          setCurrentDistanceCm((prev) => (prev !== null ? prev * 0.75 + calculatedD * 0.25 : calculatedD));
        }
      } else {
        // Fallback: 1-point model D = (W * F) / P
        const calculatedD = (targetWidthCm * fallbackFocalLength) / median;
        if (!isNaN(calculatedD) && isFinite(calculatedD)) {
          setCurrentDistanceCm((prev) => (prev !== null ? prev * 0.75 + calculatedD * 0.25 : calculatedD));
        }
      }
    } else {
      pixelHistoryRef.current = [];
    }
  }, [rawPixelWidth, targetWidthCm, twoPointCalib, fallbackFocalLength]);

  // Helper: Sample 10 frames to capture a rock-solid point
  const sampleFrames = useCallback(async (): Promise<number> => {
    setIsSampling(true);
    const samples: number[] = [];
    for (let i = 0; i < 12; i++) {
      if (pixelHistoryRef.current.length > 0) {
        const last = pixelHistoryRef.current[pixelHistoryRef.current.length - 1];
        samples.push(last);
      }
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
    setIsSampling(false);

    if (samples.length === 0) return 0;
    const sorted = [...samples].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }, []);

  // Wizard Handler: Capture Point 1
  const handleCapturePoint1 = async () => {
    if (!rawPixelWidth) {
      alert('Please ensure the object is detected in the camera frame first.');
      return;
    }
    const sampledP = await sampleFrames();
    if (sampledP > 5) {
      setPoint1Data({ p: Math.round(sampledP), d: dist1Input });
      setWizardStep(2);
    }
  };

  // Wizard Handler: Capture Point 2 and Calculate Calibration
  const handleCapturePoint2 = async () => {
    if (!point1Data || !rawPixelWidth) {
      alert('Please ensure Point 1 is locked and object is currently detected.');
      return;
    }
    const sampledP = await sampleFrames();
    if (sampledP <= 5) return;

    const p1 = point1Data.p;
    const d1 = point1Data.d;
    const p2 = Math.round(sampledP);
    const d2 = dist2Input;

    if (p1 <= p2) {
      alert('Invalid calibration: Point 2 must be farther away (smaller pixel width) than Point 1.');
      return;
    }

    // Solve 2-Point Linear Regression:
    // A = (D2 - D1) * P1 * P2 / (P1 - P2)
    // B = D1 - (A / P1)
    const slopeA = ((d2 - d1) * p1 * p2) / (p1 - p2);
    const offsetB = d1 - slopeA / p1;

    const calibResult: TwoPointCalibration = {
      slopeA,
      offsetB,
      calibWidthCm: targetWidthCm,
      p1,
      d1,
      p2,
      d2,
      timestamp: new Date().toLocaleTimeString(),
    };

    setPoint2Data({ p: p2, d: d2 });
    setTwoPointCalib(calibResult);
    localStorage.setItem('two_point_camera_calib', JSON.stringify(calibResult));
    setWizardSuccess(true);

    setTimeout(() => {
      setWizardSuccess(false);
      setShowWizard(false);
      setWizardStep(1);
    }, 2200);
  };

  const handleResetCalibration = () => {
    setTwoPointCalib(null);
    setPoint1Data(null);
    setPoint2Data(null);
    localStorage.removeItem('two_point_camera_calib');
    setWizardStep(1);
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
            Precision Distance Meter
          </h1>
        </div>

        <div className="flex items-center gap-2">
          {twoPointCalib ? (
            <span className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs bg-emerald-950/80 border border-emerald-500/60 text-emerald-300 font-semibold shadow-sm">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> 2-Point Calibrated
            </span>
          ) : (
            <span className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs bg-amber-950/60 border border-amber-500/50 text-amber-300">
              Uncalibrated (Estimated)
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
                Point camera at tyre, card, or target object
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
                {currentDistanceCm !== null ? (
                  currentDistanceCm < 100 ? (
                    `${currentDistanceCm.toFixed(1)} cm`
                  ) : (
                    `${(currentDistanceCm / 100).toFixed(2)} m`
                  )
                ) : (
                  '--'
                )}
              </span>
            </div>
            {stablePixelWidth > 0 && (
              <span className="text-xs text-slate-400 border-l border-slate-700 pl-2">
                {Math.round(stablePixelWidth)} px
              </span>
            )}
          </div>
        </div>

        {/* Live Distance Output Card */}
        <div className="grid grid-cols-3 gap-3 bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-lg">
          <div className="col-span-2 flex flex-col justify-center">
            <div className="flex items-center gap-2">
              <Ruler className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Measured Distance
              </span>
              {twoPointCalib && (
                <span className="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-700 px-1.5 py-0.2 rounded font-bold">
                  2-POINT TUNED
                </span>
              )}
            </div>

            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-4xl md:text-5xl font-black text-emerald-400 font-mono tracking-tight">
                {currentDistanceCm !== null ? (
                  currentDistanceCm < 100 ? (
                    currentDistanceCm.toFixed(1)
                  ) : (
                    (currentDistanceCm / 100).toFixed(2)
                  )
                ) : (
                  '--'
                )}
              </span>
              <span className="text-lg md:text-xl font-bold text-slate-300">
                {currentDistanceCm !== null ? (currentDistanceCm < 100 ? 'cm' : 'meters') : ''}
              </span>
            </div>
            <span className="text-xs text-slate-400 mt-1">
              {currentDistanceCm !== null ? (
                <>
                  ≈ {(currentDistanceCm / 2.54).toFixed(1)} inches (
                  {currentDistanceCm < 100
                    ? `${(currentDistanceCm * 10).toFixed(0)} mm`
                    : `${(currentDistanceCm / 100).toFixed(2)} m`}
                  )
                </>
              ) : (
                'Waiting for object to be detected...'
              )}
            </span>
          </div>

          <div className="col-span-1 border-l border-slate-800 pl-4 flex flex-col justify-center">
            <span className="text-[11px] text-slate-400 font-medium">Precision State</span>
            <span className="text-xs font-bold font-mono text-slate-200 mt-0.5">
              {twoPointCalib ? 'Calibrated (Linear)' : 'Uncalibrated (±30%)'}
            </span>
            <button
              onClick={() => setShowWizard(!showWizard)}
              className="mt-2 text-xs font-bold text-cyan-400 hover:text-cyan-300 underline text-left flex items-center gap-1"
            >
              <Target className="w-3.5 h-3.5" />
              {showWizard ? 'Close Wizard' : '2-Point Calibrate'}
            </button>
          </div>
        </div>

        {/* 2-POINT PRECISION CALIBRATION WIZARD MODAL / ACCORDION */}
        {showWizard && (
          <div className="bg-slate-900/95 border-2 border-cyan-500/50 p-4 md:p-5 rounded-2xl flex flex-col gap-4 shadow-2xl backdrop-blur-md">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2 text-cyan-300 font-bold text-sm">
                <Target className="w-4 h-4" /> 2-Point Precision Calibration Wizard
              </div>
              <span className="text-xs text-slate-400">Step {wizardStep} of 2</span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              By sampling at <strong>two exact distances</strong> (Close & Far), we calculate both your camera
              lens slope and sensor depth offset:
              <br />
              <code className="text-cyan-300 font-mono mt-1 block">
                Distance = (Slope / PixelWidth) + SensorOffset
              </code>
            </p>

            {/* STEP 1: NEAR POINT */}
            <div
              className={`p-3.5 rounded-xl border transition ${
                wizardStep === 1
                  ? 'bg-cyan-950/30 border-cyan-500'
                  : 'bg-slate-950/40 border-slate-800 opacity-60'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-white flex items-center gap-1.5">
                  <span className="w-5 h-5 rounded-full bg-cyan-600 text-black flex items-center justify-center text-[11px] font-black">
                    1
                  </span>
                  Point 1: Close Distance (e.g., 20 cm)
                </span>
                {point1Data && (
                  <span className="text-[11px] text-emerald-400 font-mono flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Locked: {point1Data.p} px @ {point1Data.d} cm
                  </span>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <label className="text-xs text-slate-300 flex items-center gap-1.5">
                  Known Distance (cm):
                  <input
                    type="number"
                    min="5"
                    max="100"
                    value={dist1Input}
                    onChange={(e) => setDist1Input(parseFloat(e.target.value) || 20)}
                    disabled={wizardStep !== 1}
                    className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                  />
                </label>

                {wizardStep === 1 && (
                  <button
                    onClick={handleCapturePoint1}
                    disabled={isSampling || !rawPixelWidth}
                    className={`ml-auto font-bold text-xs px-4 py-2 rounded-lg transition flex items-center gap-1.5 ${
                      rawPixelWidth && !isSampling
                        ? 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 active:scale-95 shadow-md shadow-cyan-900'
                        : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    }`}
                  >
                    {isSampling ? 'Sampling Frames...' : 'Capture Point 1'}
                  </button>
                )}
              </div>
            </div>

            {/* STEP 2: FAR POINT */}
            <div
              className={`p-3.5 rounded-xl border transition ${
                wizardStep === 2
                  ? 'bg-cyan-950/30 border-cyan-500'
                  : 'bg-slate-950/40 border-slate-800 opacity-60'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-white flex items-center gap-1.5">
                  <span className="w-5 h-5 rounded-full bg-cyan-600 text-black flex items-center justify-center text-[11px] font-black">
                    2
                  </span>
                  Point 2: Far Distance (e.g., 50 cm or 60 cm)
                </span>
                {point2Data && (
                  <span className="text-[11px] text-emerald-400 font-mono flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Locked: {point2Data.p} px @ {point2Data.d} cm
                  </span>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <label className="text-xs text-slate-300 flex items-center gap-1.5">
                  Known Distance (cm):
                  <input
                    type="number"
                    min="25"
                    max="300"
                    value={dist2Input}
                    onChange={(e) => setDist2Input(parseFloat(e.target.value) || 50)}
                    disabled={wizardStep !== 2}
                    className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                  />
                </label>

                {wizardStep === 2 && (
                  <div className="ml-auto flex items-center gap-2">
                    <button
                      onClick={() => setWizardStep(1)}
                      className="text-xs text-slate-400 hover:text-white px-2 py-1 underline"
                    >
                      Back to Step 1
                    </button>
                    <button
                      onClick={handleCapturePoint2}
                      disabled={isSampling || !rawPixelWidth}
                      className={`font-bold text-xs px-4 py-2 rounded-lg transition flex items-center gap-1.5 ${
                        rawPixelWidth && !isSampling
                          ? 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 active:scale-95 shadow-md shadow-emerald-900'
                          : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                      }`}
                    >
                      {wizardSuccess ? (
                        <>
                          <CheckCircle2 className="w-4 h-4 text-emerald-950" /> Done!
                        </>
                      ) : isSampling ? (
                        'Sampling Frames...'
                      ) : (
                        'Capture & Finish'
                      )}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Calibration Summary & Reset */}
            {twoPointCalib && (
              <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                <span className="font-mono text-[11px]">
                  Slope: {Math.round(twoPointCalib.slopeA)} | Offset: {twoPointCalib.offsetB.toFixed(1)} cm
                </span>
                <button
                  onClick={handleResetCalibration}
                  className="text-rose-400 hover:text-rose-300 flex items-center gap-1 underline text-xs"
                >
                  <RotateCcw className="w-3.5 h-3.5" /> Reset to Factory Default
                </button>
              </div>
            )}
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