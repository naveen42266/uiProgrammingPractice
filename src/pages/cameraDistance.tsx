import { useRef, useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import {
  Camera,
  RefreshCw,
  Volume2,
  VolumeX,
  Play,
  Square,
  ShieldCheck,
  Activity,
  RotateCcw,
  Target,
  Box,
} from 'lucide-react';

interface Preset {
  label: string;
  category: 'tyre' | 'reference' | 'gadget';
  widthCm: number;
  description: string;
}

interface TwoPointCalibration {
  slopeA: number;
  offsetB: number;
  calibWidthCm: number;
  p1: number;
  d1: number;
  p2: number;
  d2: number;
  timestamp: string;
}

type GuidanceZone = 'TOO_CLOSE' | 'IN_RANGE' | 'TOO_FAR' | 'NO_TARGET';

const PRESETS: Preset[] = [
  { label: 'Tyre Tread (205 mm)', category: 'tyre', widthCm: 20.5, description: 'Standard 205 mm contact tread' },
  { label: 'Tyre Tread (225 mm)', category: 'tyre', widthCm: 22.5, description: 'Standard 225 mm contact tread' },
  { label: 'Tyre Tread (195 mm)', category: 'tyre', widthCm: 19.5, description: 'Standard 195 mm contact tread' },
  { label: 'Credit / ID Card', category: 'reference', widthCm: 8.56, description: 'Standard ISO card width (85.6 mm)' },
  { label: 'Smartphone', category: 'gadget', widthCm: 7.5, description: 'Average phone width (~7.5 cm)' },
];

export default function DistanceEstimator() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const pixelHistoryRef = useRef<number[]>([]);
  const lastToneTimeRef = useRef<number>(0);

  // Optical parameters
  const [targetWidthCm, setTargetWidthCm] = useState<number>(20.5);
  const [focalLength, setFocalLength] = useState<number>(() => {
    const saved = localStorage.getItem('ai_camera_focal_length');
    return saved ? parseFloat(saved) : 650;
  });

  // 2-Point Calibration (if available)
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

  // Camera & Model state
  const [cameraFacing, setCameraFacing] = useState<'environment' | 'user'>('environment');
  const [cameraReady, setCameraReady] = useState<boolean>(false);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [modelLoading, setModelLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Real-time Detection & Metrics
  const [predictions, setPredictions] = useState<cocoSsd.DetectedObject[]>([]);
  const [currentDistanceCm, setCurrentDistanceCm] = useState<number | null>(null);
  const [stablePixelWidth, setStablePixelWidth] = useState<number>(0);
  const [sharpnessScore, setSharpnessScore] = useState<number>(0);

  // Guidance System Controls
  const [isAudioEnabled, setIsAudioEnabled] = useState<boolean>(true);
  const [isSessionActive, setIsSessionActive] = useState<boolean>(false);
  const [validRecordedSeconds, setValidRecordedSeconds] = useState<number>(0);
  const [sessionFrameCount, setSessionFrameCount] = useState<number>(0);

  // Quick 2-Point Calibration Wizard state
  const [showWizard, setShowWizard] = useState<boolean>(false);
  const [wizardStep, setWizardStep] = useState<1 | 2>(1);
  const [dist1Input, setDist1Input] = useState<number>(15); // Point 1: 15 cm
  const [dist2Input, setDist2Input] = useState<number>(25); // Point 2: 25 cm
  const [point1Data, setPoint1Data] = useState<{ p: number; d: number } | null>(null);

  // 1. Initialize TensorFlow.js and COCO-SSD
  useEffect(() => {
    let mounted = true;
    async function initAI() {
      try {
        setModelLoading(true);
        await tf.ready();
        const loaded = await cocoSsd.load({ base: 'mobilenet_v2' });
        if (mounted) {
          setModel(loaded);
          setModelLoading(false);
        }
      } catch (err) {
        console.error('TFJS error:', err);
        if (mounted) {
          setErrorMsg('Failed to load AI model.');
          setModelLoading(false);
        }
      }
    }
    initAI();
    return () => {
      mounted = false;
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
        console.error('Camera error:', err);
        setErrorMsg('Camera access denied. Please grant camera permission.');
      });

    return () => {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [cameraFacing]);

  // 3. Audio Guidance Synthesizer (Web Audio API)
  const playGuidanceTone = useCallback(
    (zone: GuidanceZone) => {
      if (!isAudioEnabled) return;
      const now = performance.now();
      // Throttle audio beeps to avoid harsh sound
      const interval = zone === 'IN_RANGE' ? 220 : 380;
      if (now - lastToneTimeRef.current < interval) return;
      lastToneTimeRef.current = now;

      try {
        if (!audioCtxRef.current) {
          audioCtxRef.current = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
        }
        const ctx = audioCtxRef.current;
        if (ctx.state === 'suspended') {
          ctx.resume();
        }

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        if (zone === 'IN_RANGE') {
          // Harmonious high-pitch chime (Target Locked)
          osc.type = 'sine';
          osc.frequency.setValueAtTime(880, ctx.currentTime);
          gain.gain.setValueAtTime(0.06, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
          osc.start();
          osc.stop(ctx.currentTime + 0.12);
        } else if (zone === 'TOO_CLOSE') {
          // Low warning buzz (Move back)
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(260, ctx.currentTime);
          gain.gain.setValueAtTime(0.06, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
          osc.start();
          osc.stop(ctx.currentTime + 0.15);
        } else if (zone === 'TOO_FAR') {
          // Medium prompting chirp (Move closer)
          osc.type = 'sine';
          osc.frequency.setValueAtTime(520, ctx.currentTime);
          gain.gain.setValueAtTime(0.04, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);
          osc.start();
          osc.stop(ctx.currentTime + 0.08);
        }
      } catch {
        // Audio policy ignore
      }
    },
    [isAudioEnabled]
  );

  // 4. Real-time Laplacian Sharpness & Object Detection Engine
  useEffect(() => {
    if (!cameraReady) return;

    let animId: number;
    let isDetecting = false;
    const canvas = analysisCanvasRef.current || document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 90;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const processFrame = async () => {
      const v = videoRef.current;
      if (v && v.readyState >= 2 && ctx) {
        // Fast Sharpness calculation via Laplacian kernel on off-screen canvas
        ctx.drawImage(v, 0, 0, 160, 90);
        const imgData = ctx.getImageData(0, 0, 160, 90);
        const d = imgData.data;

        let sumLap = 0;
        let sumLapSq = 0;
        let count = 0;

        for (let y = 1; y < 89; y += 2) {
          for (let x = 1; x < 159; x += 2) {
            const idx = (y * 160 + x) * 4;
            const c = d[idx];
            const up = d[((y - 1) * 160 + x) * 4];
            const down = d[((y + 1) * 160 + x) * 4];
            const left = d[(y * 160 + (x - 1)) * 4];
            const right = d[(y * 160 + (x + 1)) * 4];
            const lap = Math.abs(up + down + left + right - 4 * c);
            sumLap += lap;
            sumLapSq += lap * lap;
            count++;
          }
        }
        if (count > 0) {
          const mean = sumLap / count;
          const variance = sumLapSq / count - mean * mean;
          const score = Math.min(100, Math.round(Math.sqrt(Math.max(0, variance)) * 4.2));
          setSharpnessScore(score);
        }

        // AI Detection
        if (model && !isDetecting) {
          isDetecting = true;
          try {
            const results = await model.detect(v, 4, 0.25);
            setPredictions(results);
          } catch (e) {
            console.warn(e);
          } finally {
            isDetecting = false;
          }
        }
      }
      animId = requestAnimationFrame(processFrame);
    };

    animId = requestAnimationFrame(processFrame);
    return () => cancelAnimationFrame(animId);
  }, [cameraReady, model]);

  // Active object width tracking
  const activePred = predictions[0] || null;
  const rawPixelWidth = activePred ? activePred.bbox[2] : 0;

  // 5. Distance Filtering & State Calculation
  useEffect(() => {
    if (rawPixelWidth > 5) {
      const hist = pixelHistoryRef.current;
      hist.push(rawPixelWidth);
      if (hist.length > 12) hist.shift();

      const sorted = [...hist].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      setStablePixelWidth(median);

      let dist = 0;
      if (twoPointCalib) {
        const effectiveA = twoPointCalib.slopeA * (targetWidthCm / twoPointCalib.calibWidthCm);
        dist = effectiveA / median + twoPointCalib.offsetB;
      } else {
        dist = (targetWidthCm * focalLength) / median;
      }

      if (!isNaN(dist) && isFinite(dist)) {
        setCurrentDistanceCm((prev) => (prev !== null ? prev * 0.75 + dist * 0.25 : dist));
      }
    } else {
      pixelHistoryRef.current = [];
    }
  }, [rawPixelWidth, targetWidthCm, twoPointCalib, focalLength]);

  // Compute Current Guidance Zone (15 to 20 cm TARGET)
  const currentZone: GuidanceZone =
    currentDistanceCm === null
      ? 'NO_TARGET'
      : currentDistanceCm < 15.0
      ? 'TOO_CLOSE'
      : currentDistanceCm <= 20.0
      ? 'IN_RANGE'
      : 'TOO_FAR';

  // Trigger Guidance Audio and Mobile Haptics
  useEffect(() => {
    if (currentZone !== 'NO_TARGET') {
      playGuidanceTone(currentZone);
      if (currentZone === 'IN_RANGE' && navigator.vibrate) {
        navigator.vibrate(25);
      }
    }
  }, [currentZone, playGuidanceTone]);

  // 6. Smart Auto-Record Gating (Accumulate valid frames only when in 15-20cm & sharp)
  useEffect(() => {
    if (!isSessionActive) return;

    const timer = setInterval(() => {
      // Only accumulate when in the green zone (15 - 20 cm) and not severely blurred
      if (currentZone === 'IN_RANGE' && sharpnessScore >= 25) {
        setValidRecordedSeconds((prev) => +(prev + 0.1).toFixed(1));
        setSessionFrameCount((prev) => prev + 3);
      }
    }, 100);

    return () => clearInterval(timer);
  }, [isSessionActive, currentZone, sharpnessScore]);

  // Expected pixel width at optimal 17.5 cm target distance for on-screen corridor brackets
  const expectedPixelsAt17cm = Math.round((targetWidthCm * focalLength) / 17.5);
  const corridorWidthPct = Math.min(85, Math.max(30, (expectedPixelsAt17cm / 1280) * 100));

  // Quick 2-Point Calibration handlers
  const handleCapturePoint1 = () => {
    if (!stablePixelWidth) return;
    setPoint1Data({ p: Math.round(stablePixelWidth), d: dist1Input });
    setWizardStep(2);
  };

  const handleCapturePoint2 = () => {
    if (!point1Data || !stablePixelWidth) return;
    const p1 = point1Data.p;
    const d1 = point1Data.d;
    const p2 = Math.round(stablePixelWidth);
    const d2 = dist2Input;

    if (p1 <= p2) {
      alert('Point 2 must be farther away (smaller pixel width) than Point 1.');
      return;
    }

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

    setTwoPointCalib(calibResult);
    localStorage.setItem('two_point_camera_calib', JSON.stringify(calibResult));
    setShowWizard(false);
    setWizardStep(1);
  };

  const handleResetCalibration = () => {
    setTwoPointCalib(null);
    setPoint1Data(null);
    localStorage.removeItem('two_point_camera_calib');
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 select-none font-sans">
      <canvas ref={analysisCanvasRef} className="hidden" />

      {/* Header */}
      <header className="w-full max-w-2xl flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Target className="w-6 h-6 text-emerald-400 animate-spin-slow" />
          <div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              Tyre Distance Guidance HUD
            </h1>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-emerald-400 font-semibold uppercase tracking-wider">
                Enforcing 15 – 20 cm Scan Window
              </span>
              {modelLoading && (
                <span className="text-[10px] text-amber-300 font-semibold animate-pulse">
                  (Loading AI...)
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Audio toggle */}
          <button
            onClick={() => setIsAudioEnabled(!isAudioEnabled)}
            className={`p-2 rounded-lg border text-xs transition ${
              isAudioEnabled
                ? 'bg-emerald-950/80 border-emerald-500/60 text-emerald-300'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
            title={isAudioEnabled ? 'Guidance Audio Enabled' : 'Audio Muted'}
          >
            {isAudioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>

          {/* Camera flip */}
          <button
            onClick={() => setCameraFacing((p) => (p === 'environment' ? 'user' : 'environment'))}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
            title="Flip camera"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Guidance Viewport */}
      <div className="w-full max-w-2xl flex flex-col gap-3">
        {/* REAL-TIME DISTANCE BAR GAUGE (Traffic Light Scale) */}
        <div className="bg-slate-900 border border-slate-800 p-3 rounded-2xl flex flex-col gap-2 shadow-lg">
          <div className="flex items-center justify-between text-xs font-bold">
            <span className="text-amber-400">TOO CLOSE (&lt;15 cm)</span>
            <span className="text-emerald-400 font-extrabold flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5" /> GREEN ZONE (15 – 20 cm)
            </span>
            <span className="text-sky-400">TOO FAR (&gt;20 cm)</span>
          </div>

          {/* Graphical Distance Slider Bar */}
          <div className="relative h-6 bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex">
            {/* Too Close Zone (0 to 15cm) */}
            <div className="w-[37.5%] h-full bg-amber-500/20 border-r border-amber-500/50 flex items-center justify-center text-[10px] text-amber-300/80 font-mono">
              0 - 15 cm
            </div>

            {/* Target 15-20cm Zone */}
            <div className="w-[12.5%] h-full bg-emerald-500/35 border-r border-emerald-400 flex items-center justify-center text-[10px] text-emerald-200 font-extrabold font-mono shadow-[inset_0_0_12px_rgba(52,211,153,0.3)]">
              ★ 15-20
            </div>

            {/* Too Far Zone (20cm to 40cm+) */}
            <div className="w-[50%] h-full bg-sky-500/20 flex items-center justify-center text-[10px] text-sky-300/80 font-mono">
              20 - 40+ cm
            </div>

            {/* Live Indicator Pin */}
            {currentDistanceCm !== null && (
              <div
                style={{
                  left: `${Math.min(98, Math.max(2, (currentDistanceCm / 40) * 100))}%`,
                }}
                className={`absolute top-0 bottom-0 w-3 -ml-1.5 rounded-full shadow-lg transition-all duration-75 flex items-center justify-center ${
                  currentZone === 'IN_RANGE'
                    ? 'bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,1)] scale-110'
                    : currentZone === 'TOO_CLOSE'
                    ? 'bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.8)]'
                    : 'bg-sky-400 shadow-[0_0_10px_rgba(56,189,248,0.8)]'
                }`}
              >
                <div className="w-1 h-3 bg-black rounded-full" />
              </div>
            )}
          </div>
        </div>

        {/* Live Camera Viewport with Guidance Reticle */}
        <div
          ref={containerRef}
          className={`relative w-full aspect-[4/3] md:aspect-video bg-black rounded-2xl overflow-hidden border-2 transition-all duration-200 shadow-2xl ${
            currentZone === 'IN_RANGE'
              ? 'border-emerald-400 shadow-[0_0_25px_rgba(52,211,153,0.35)]'
              : currentZone === 'TOO_CLOSE'
              ? 'border-amber-500 shadow-[0_0_20px_rgba(245,158,11,0.25)]'
              : 'border-slate-800'
          }`}
        >
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />

          {errorMsg && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/85 p-6 text-center text-rose-400 text-sm">
              {errorMsg}
            </div>
          )}

          {!cameraReady && !errorMsg && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/75 text-slate-400 text-sm">
              <Camera className="w-6 h-6 animate-pulse mr-2" /> Initializing Live Camera...
            </div>
          )}

          {/* ON-SCREEN GEOMETRIC GUIDANCE CORRIDOR (17.5 cm Framing Target) */}
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center z-10">
            <div
              style={{ width: `${corridorWidthPct}%`, height: '55%' }}
              className={`border-2 border-dashed rounded-2xl flex flex-col justify-between p-2 transition-all duration-150 ${
                currentZone === 'IN_RANGE'
                  ? 'border-emerald-400 bg-emerald-500/10 shadow-[0_0_20px_rgba(52,211,153,0.3)]'
                  : 'border-white/30 bg-white/5'
              }`}
            >
              <div className="flex justify-between text-[10px] font-mono font-bold text-white/70">
                <span>[ CORRIDOR TARGET ]</span>
                <span>17.5 cm GUIDE</span>
              </div>
              <div className="text-center text-[11px] font-semibold text-white/80">
                Fit tyre tread between dashed boundaries
              </div>
            </div>
          </div>

          {/* AI Detected Object Box */}
          {predictions.map((pred, i) => {
            const [x, y, w, h] = pred.bbox;
            const leftPct = (x / (videoRef.current?.videoWidth || 1280)) * 100;
            const topPct = (y / (videoRef.current?.videoHeight || 720)) * 100;
            const wPct = (w / (videoRef.current?.videoWidth || 1280)) * 100;
            const hPct = (h / (videoRef.current?.videoHeight || 720)) * 100;

            return (
              <div
                key={i}
                style={{ left: `${leftPct}%`, top: `${topPct}%`, width: `${wPct}%`, height: `${hPct}%` }}
                className={`absolute pointer-events-none z-20 border-2 rounded-lg transition-all ${
                  currentZone === 'IN_RANGE'
                    ? 'border-emerald-400 bg-emerald-500/15'
                    : 'border-cyan-400/80 bg-cyan-500/10'
                }`}
              >
                <span className="absolute -top-5 left-0 text-[10px] bg-black/80 px-1.5 py-0.5 rounded text-white font-mono flex items-center gap-1">
                  <Box className="w-2.5 h-2.5" /> {pred.class}
                </span>
              </div>
            );
          })}

          {/* TOP REAL-TIME PROMPT BANNER */}
          <div className="absolute top-3 inset-x-3 z-30 flex items-center justify-between">
            {/* Status Pill */}
            <div
              className={`px-3.5 py-1.5 rounded-xl border backdrop-blur-md flex items-center gap-2 font-black text-xs md:text-sm shadow-xl transition-all ${
                currentZone === 'IN_RANGE'
                  ? 'bg-emerald-950/90 border-emerald-400 text-emerald-300'
                  : currentZone === 'TOO_CLOSE'
                  ? 'bg-amber-950/90 border-amber-400 text-amber-300'
                  : currentZone === 'TOO_FAR'
                  ? 'bg-sky-950/90 border-sky-400 text-sky-300'
                  : 'bg-slate-900/90 border-slate-700 text-slate-400'
              }`}
            >
              <span className="w-2.5 h-2.5 rounded-full animate-ping bg-current" />
              <span>
                {currentZone === 'IN_RANGE'
                  ? '🟢 PERFECT DISTANCE — IN RANGE'
                  : currentZone === 'TOO_CLOSE'
                  ? '⬅ MOVE BACK (TOO CLOSE)'
                  : currentZone === 'TOO_FAR'
                  ? '➡ MOVE CLOSER (TOO FAR)'
                  : 'AIM AT TYRE TREAD'}
              </span>
            </div>

            {/* Live Distance Value */}
            <div className="bg-slate-950/90 border border-slate-800 px-3 py-1 rounded-xl text-right backdrop-blur-md">
              <span className="text-[10px] text-slate-400 block font-semibold">LIVE DISTANCE</span>
              <span className="text-base md:text-lg font-black font-mono text-cyan-300">
                {currentDistanceCm !== null ? `${currentDistanceCm.toFixed(1)} cm` : '--'}
              </span>
            </div>
          </div>

          {/* BOTTOM METRICS HUD (Sharpness + Gating Status) */}
          <div className="absolute bottom-3 inset-x-3 z-30 flex items-center justify-between pointer-events-none">
            {/* Sharpness (Laplacian) */}
            <div className="bg-slate-950/80 border border-slate-800 px-2.5 py-1 rounded-lg flex items-center gap-1.5 text-xs">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400 text-[11px]">Sharpness:</span>
              <span
                className={`font-mono font-bold ${
                  sharpnessScore > 35 ? 'text-emerald-400' : 'text-amber-400'
                }`}
              >
                {sharpnessScore}%
              </span>
            </div>

            {/* Auto-Record Gating Indicator */}
            {isSessionActive && (
              <div
                className={`px-3 py-1 rounded-lg text-xs font-bold border flex items-center gap-1.5 shadow-md ${
                  currentZone === 'IN_RANGE'
                    ? 'bg-emerald-950 border-emerald-400 text-emerald-300 animate-pulse'
                    : 'bg-rose-950/80 border-rose-500 text-rose-300'
                }`}
              >
                <div className="w-2 h-2 rounded-full bg-current" />
                <span>
                  {currentZone === 'IN_RANGE' ? 'CAPTURING VALID FRAMES' : 'RECORDING PAUSED (OUT OF RANGE)'}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* SMART RECORDING CONTROLS & COVERAGE LOG */}
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl flex flex-wrap items-center justify-between gap-3 shadow-lg">
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                if (!isSessionActive) {
                  setValidRecordedSeconds(0);
                  setSessionFrameCount(0);
                }
                setIsSessionActive(!isSessionActive);
              }}
              className={`px-4 py-2 rounded-xl font-bold text-xs flex items-center gap-2 transition shadow-md ${
                isSessionActive
                  ? 'bg-rose-600 hover:bg-rose-500 text-white'
                  : 'bg-emerald-600 hover:bg-emerald-500 text-white'
              }`}
            >
              {isSessionActive ? (
                <>
                  <Square className="w-4 h-4" /> Stop Tyre Scan Session
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" /> Start Tyre Scan Session
                </>
              )}
            </button>

            <div className="flex flex-col">
              <span className="text-xs text-slate-400 font-semibold">Valid 15–20 cm Footage:</span>
              <span className="text-sm font-bold font-mono text-emerald-400">
                {validRecordedSeconds}s ({sessionFrameCount} frames)
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowWizard(!showWizard)}
              className="text-xs text-cyan-400 hover:text-cyan-300 underline font-semibold flex items-center gap-1"
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              {twoPointCalib ? 'Calibrated (15 & 25 cm)' : 'Calibrate 15 & 25 cm'}
            </button>
          </div>
        </div>

        {/* 2-POINT QUICK CALIBRATION WIZARD (Collapsible) */}
        {showWizard && (
          <div className="bg-slate-900/95 border-2 border-cyan-500/50 p-4 rounded-2xl flex flex-col gap-3 shadow-xl">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-cyan-300 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4" /> Fast 15 & 25 cm Lens Calibration
              </span>
              <span className="text-xs text-slate-400">Step {wizardStep} of 2</span>
            </div>

            <p className="text-xs text-slate-300">
              Hold the object at <strong>15 cm</strong> and click Capture Point 1, then hold at{' '}
              <strong>25 cm</strong> and click Capture Point 2.
            </p>

            <div className="flex flex-wrap items-center gap-3">
              {wizardStep === 1 ? (
                <div className="flex items-center gap-2 w-full">
                  <span className="text-xs text-slate-300">Point 1 (15 cm):</span>
                  <input
                    type="number"
                    value={dist1Input}
                    onChange={(e) => setDist1Input(parseFloat(e.target.value) || 15)}
                    className="w-16 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                  />
                  <button
                    onClick={handleCapturePoint1}
                    disabled={!stablePixelWidth}
                    className="ml-auto bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs px-3 py-1.5 rounded-lg"
                  >
                    Capture 15 cm
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2 w-full">
                  <span className="text-xs text-slate-300">Point 2 (25 cm):</span>
                  <input
                    type="number"
                    value={dist2Input}
                    onChange={(e) => setDist2Input(parseFloat(e.target.value) || 25)}
                    className="w-16 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                  />
                  <button
                    onClick={handleCapturePoint2}
                    disabled={!stablePixelWidth}
                    className="ml-auto bg-emerald-500 hover:bg-emerald-400 text-black font-bold text-xs px-3 py-1.5 rounded-lg"
                  >
                    Finish Calibration
                  </button>
                </div>
              )}
            </div>

            {twoPointCalib && (
              <div className="pt-2 border-t border-slate-800 flex justify-end">
                <button
                  onClick={handleResetCalibration}
                  className="text-xs text-rose-400 hover:text-rose-300 flex items-center gap-1 underline"
                >
                  <RotateCcw className="w-3.5 h-3.5" /> Reset Calibration
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tyre Section Width Presets */}
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Tyre Nominal Width Setting
            </span>
            <span className="text-xs text-emerald-400 font-mono font-bold">
              Current: {targetWidthCm} cm ({targetWidthCm * 10} mm)
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setTargetWidthCm(p.widthCm)}
                className={`p-2 rounded-xl border text-left flex flex-col transition ${
                  targetWidthCm === p.widthCm
                    ? 'bg-emerald-950/60 border-emerald-400 shadow-md shadow-emerald-950 text-white'
                    : 'bg-slate-800/80 border-slate-700/80 hover:bg-slate-800 text-slate-300'
                }`}
              >
                <span className="text-xs font-bold">{p.label}</span>
                <span className="text-[11px] text-slate-400 mt-0.5">{p.widthCm} cm</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}