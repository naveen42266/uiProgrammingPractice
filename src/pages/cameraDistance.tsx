import { useRef, useState, useEffect, useCallback } from 'react';
import {
  Camera,
  RefreshCw,
  Volume2,
  VolumeX,
  Play,
  Square,
  ShieldCheck,
  Activity,
  Target,
  Sliders,
  CheckCircle2,
  RotateCcw,
} from 'lucide-react';

interface TyrePreset {
  label: string;
  widthCm: number;
}

type GuidanceZone = 'TOO_CLOSE' | 'IN_RANGE' | 'TOO_FAR' | 'NO_TARGET';

const TYRE_PRESETS: TyrePreset[] = [
  { label: '195 mm Tread', widthCm: 19.5 },
  { label: '205 mm Tread', widthCm: 20.5 },
  { label: '225 mm Tread', widthCm: 22.5 },
  { label: '245 mm Tread', widthCm: 24.5 },
];

export default function TyreGuidanceScanner() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const pixelHistoryRef = useRef<number[]>([]);
  const lastToneTimeRef = useRef<number>(0);

  // Tyre & Optical Parameters
  const [targetWidthCm, setTargetWidthCm] = useState<number>(20.5);
  const [focalLength, setFocalLength] = useState<number>(() => {
    const saved = localStorage.getItem('tyre_camera_focal_length');
    return saved ? parseFloat(saved) : 650;
  });
  const [isCalibrated, setIsCalibrated] = useState<boolean>(() => {
    return !!localStorage.getItem('tyre_camera_focal_length');
  });

  // Edge Marker Positions (0.0 to 1.0 fraction of viewport width)
  const [leftEdgeRatio, setLeftEdgeRatio] = useState<number>(0.25);
  const [rightEdgeRatio, setRightEdgeRatio] = useState<number>(0.75);
  const [autoEdgeTracking, setAutoEdgeTracking] = useState<boolean>(true);
  const [draggingHandle, setDraggingHandle] = useState<'left' | 'right' | null>(null);

  // Calibration Form State
  const [calibrationDistInput, setCalibrationDistInput] = useState<number>(18); // 18 cm sweet spot
  const [showCalibrationDrawer, setShowCalibrationDrawer] = useState<boolean>(false);
  const [calibrationSuccess, setCalibrationSuccess] = useState<boolean>(false);

  // Camera state
  const [cameraFacing, setCameraFacing] = useState<'environment' | 'user'>('environment');
  const [cameraReady, setCameraReady] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Real-time Metrics
  const [currentDistanceCm, setCurrentDistanceCm] = useState<number | null>(null);
  const [sharpnessScore, setSharpnessScore] = useState<number>(0);

  // Guidance System Controls
  const [isAudioEnabled, setIsAudioEnabled] = useState<boolean>(true);
  const [isSessionActive, setIsSessionActive] = useState<boolean>(false);
  const [validRecordedSeconds, setValidRecordedSeconds] = useState<number>(0);
  const [sessionFrameCount, setSessionFrameCount] = useState<number>(0);

  // 1. Initialize Camera Stream
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
              const savedF = localStorage.getItem('tyre_camera_focal_length');
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

  // 2. Multi-channel Guidance Audio (Web Audio API)
  const playGuidanceTone = useCallback(
    (zone: GuidanceZone) => {
      if (!isAudioEnabled) return;
      const now = performance.now();
      const interval = zone === 'IN_RANGE' ? 220 : 380;
      if (now - lastToneTimeRef.current < interval) return;
      lastToneTimeRef.current = now;

      try {
        if (!audioCtxRef.current) {
          audioCtxRef.current = new (window.AudioContext ||
            (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
        }
        const ctx = audioCtxRef.current;
        if (ctx.state === 'suspended') ctx.resume();

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        if (zone === 'IN_RANGE') {
          // Harmonious High Tone (880 Hz)
          osc.type = 'sine';
          osc.frequency.setValueAtTime(880, ctx.currentTime);
          gain.gain.setValueAtTime(0.06, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
          osc.start();
          osc.stop(ctx.currentTime + 0.12);
        } else if (zone === 'TOO_CLOSE') {
          // Low Alert Tone (260 Hz)
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(260, ctx.currentTime);
          gain.gain.setValueAtTime(0.06, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
          osc.start();
          osc.stop(ctx.currentTime + 0.15);
        } else if (zone === 'TOO_FAR') {
          // Medium Chirp (520 Hz)
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

  // 3. Real-Time Laplacian Sharpness & Edge Detection Loop
  useEffect(() => {
    if (!cameraReady) return;

    let animId: number;
    const canvas = analysisCanvasRef.current || document.createElement('canvas');
    const procW = 240;
    const procH = 135;
    canvas.width = procW;
    canvas.height = procH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const processFrame = () => {
      const v = videoRef.current;
      if (v && v.readyState >= 2 && ctx) {
        ctx.drawImage(v, 0, 0, procW, procH);
        const imgData = ctx.getImageData(0, 0, procW, procH);
        const d = imgData.data;

        // Laplacian Focus/Sharpness Variance Calculation
        let sumLap = 0;
        let sumLapSq = 0;
        let count = 0;

        for (let y = 2; y < procH - 2; y += 2) {
          for (let x = 2; x < procW - 2; x += 2) {
            const idx = (y * procW + x) * 4;
            const c = d[idx];
            const up = d[((y - 1) * procW + x) * 4];
            const down = d[((y + 1) * procW + x) * 4];
            const left = d[(y * procW + (x - 1)) * 4];
            const right = d[(y * procW + (x + 1)) * 4];
            const lap = Math.abs(up + down + left + right - 4 * c);
            sumLap += lap;
            sumLapSq += lap * lap;
            count++;
          }
        }

        if (count > 0) {
          const mean = sumLap / count;
          const variance = sumLapSq / count - mean * mean;
          const score = Math.min(100, Math.round(Math.sqrt(Math.max(0, variance)) * 3.8));
          setSharpnessScore(score);
        }

        // Automatic Edge Detection (only if auto tracking enabled and not user-dragging)
        if (autoEdgeTracking && !draggingHandle) {
          const startY = Math.floor(procH * 0.35);
          const endY = Math.floor(procH * 0.65);
          const hSpan = endY - startY;

          const lum = new Float32Array(procW);
          for (let x = 0; x < procW; x++) {
            let colSum = 0;
            for (let y = startY; y < endY; y++) {
              const idx = (y * procW + x) * 4;
              colSum += 0.299 * d[idx] + 0.587 * d[idx + 1] + 0.114 * d[idx + 2];
            }
            lum[x] = colSum / hSpan;
          }

          const centerX = Math.floor(procW / 2);
          let leftEdge = centerX;
          let rightEdge = centerX;

          let totalLum = 0;
          for (let x = 0; x < procW; x++) totalLum += lum[x];
          const avgLum = totalLum / procW;
          const rubberThreshold = Math.min(avgLum * 0.92, 115);

          while (leftEdge > 12 && lum[leftEdge] < rubberThreshold) leftEdge--;
          while (rightEdge < procW - 12 && lum[rightEdge] < rubberThreshold) rightEdge++;

          const measuredSpanProc = rightEdge - leftEdge;
          if (measuredSpanProc > 30) {
            const rawLeft = leftEdge / procW;
            const rawRight = rightEdge / procW;
            setLeftEdgeRatio((prev) => prev * 0.85 + rawLeft * 0.15);
            setRightEdgeRatio((prev) => prev * 0.85 + rawRight * 0.15);
          }
        }
      }
      animId = requestAnimationFrame(processFrame);
    };

    animId = requestAnimationFrame(processFrame);
    return () => cancelAnimationFrame(animId);
  }, [cameraReady, autoEdgeTracking, draggingHandle]);

  // 4. Calculate Distance from Active Edge Markers (Visual Overlay)
  const videoNativeW = videoRef.current?.videoWidth || 1280;
  const currentPixelSpan = Math.max(10, (rightEdgeRatio - leftEdgeRatio) * videoNativeW);

  useEffect(() => {
    if (currentPixelSpan > 10) {
      // Rolling median filter to reject wobble
      const hist = pixelHistoryRef.current;
      hist.push(currentPixelSpan);
      if (hist.length > 10) hist.shift();

      const sorted = [...hist].sort((a, b) => a - b);
      const medianP = sorted[Math.floor(sorted.length / 2)];

      // Triangle Similarity Formula: Distance = (Real Width * Focal Length) / Pixel Span
      const dist = (targetWidthCm * focalLength) / medianP;
      if (!isNaN(dist) && isFinite(dist)) {
        setCurrentDistanceCm((prev) => (prev !== null ? prev * 0.75 + dist * 0.25 : dist));
      }
    }
  }, [currentPixelSpan, targetWidthCm, focalLength]);

  // 5. Compute Guidance Zone (15 to 20 cm Target Window)
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
        navigator.vibrate(20);
      }
    }
  }, [currentZone, playGuidanceTone]);

  // 6. Smart Auto-Record Gating
  useEffect(() => {
    if (!isSessionActive) return;

    const timer = setInterval(() => {
      if (currentZone === 'IN_RANGE' && sharpnessScore >= 20) {
        setValidRecordedSeconds((prev) => +(prev + 0.1).toFixed(1));
        setSessionFrameCount((prev) => prev + 3);
      }
    }, 100);

    return () => clearInterval(timer);
  }, [isSessionActive, currentZone, sharpnessScore]);

  // Handle Dragging Edge Lines
  const handlePointerMove = useCallback(
    (clientX: number) => {
      if (!draggingHandle || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const currentRatio = Math.max(0.05, Math.min(0.95, (clientX - rect.left) / rect.width));

      if (draggingHandle === 'left') {
        setLeftEdgeRatio(Math.min(currentRatio, rightEdgeRatio - 0.08));
      } else if (draggingHandle === 'right') {
        setRightEdgeRatio(Math.max(currentRatio, leftEdgeRatio + 0.08));
      }
    },
    [draggingHandle, leftEdgeRatio, rightEdgeRatio]
  );

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => handlePointerMove(e.clientX);
    const onMouseUp = () => setDraggingHandle(null);
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches[0]) handlePointerMove(e.touches[0].clientX);
    };
    const onTouchEnd = () => setDraggingHandle(null);

    if (draggingHandle) {
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
  }, [draggingHandle, handlePointerMove]);

  // 1-Click Calibration at physical distance (Eliminates focal length error)
  const handleLockCalibration = () => {
    if (calibrationDistInput <= 0 || targetWidthCm <= 0 || currentPixelSpan <= 0) return;
    // F = (Pixel Span * Known Distance) / Target Real Width
    const exactF = Math.round((currentPixelSpan * calibrationDistInput) / targetWidthCm);
    setFocalLength(exactF);
    localStorage.setItem('tyre_camera_focal_length', exactF.toString());
    setIsCalibrated(true);
    setCalibrationSuccess(true);
    setTimeout(() => {
      setCalibrationSuccess(false);
      setShowCalibrationDrawer(false);
    }, 1800);
  };

  const handleResetCalibration = () => {
    localStorage.removeItem('tyre_camera_focal_length');
    const autoF = videoRef.current?.videoWidth ? Math.round(videoRef.current.videoWidth * 0.714) : 650;
    setFocalLength(autoF);
    setIsCalibrated(false);
  };

  // Expected 17.5 cm corridor guide width
  const expectedPixelsAt17cm = Math.round((targetWidthCm * focalLength) / 17.5);
  const corridorWidthPct = Math.min(85, Math.max(30, (expectedPixelsAt17cm / videoNativeW) * 100));

  return (
    <div className="flex flex-col items-center min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 select-none font-sans">
      <canvas ref={analysisCanvasRef} className="hidden" />

      {/* Header */}
      <header className="w-full max-w-2xl flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Target className="w-6 h-6 text-emerald-400" />
          <div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white">
              Tyre Guidance Scanner
            </h1>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-emerald-400 font-semibold uppercase tracking-wider">
                15 – 20 cm Window Enforcement
              </span>
              {isCalibrated ? (
                <span className="text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-600 px-1.5 py-0.2 rounded font-bold">
                  ✓ Lens Calibrated
                </span>
              ) : (
                <span className="text-[10px] bg-amber-950 text-amber-300 border border-amber-600 px-1.5 py-0.2 rounded font-bold">
                  ⚠ Uncalibrated
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
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
        {/* 1. GREEN ZONE DISTANCE GAUGE (Traffic Light Bar) */}
        <div className="bg-slate-900 border border-slate-800 p-3 rounded-2xl flex flex-col gap-2 shadow-lg">
          <div className="flex items-center justify-between text-xs font-bold">
            <span className="text-amber-400">TOO CLOSE (&lt;15 cm)</span>
            <span className="text-emerald-400 font-extrabold flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5" /> GREEN ZONE (15 – 20 cm)
            </span>
            <span className="text-sky-400">TOO FAR (&gt;20 cm)</span>
          </div>

          {/* Bar Scale with Live Position Needle */}
          <div className="relative h-6 bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex">
            {/* Zone 1: <15cm */}
            <div className="w-[37.5%] h-full bg-amber-500/20 border-r border-amber-500/40 flex items-center justify-center text-[10px] text-amber-300/80 font-mono">
              0 - 15 cm
            </div>
            {/* Zone 2: 15-20cm (Target) */}
            <div className="w-[12.5%] h-full bg-emerald-500/35 border-r border-emerald-400 flex items-center justify-center text-[10px] text-emerald-200 font-extrabold font-mono shadow-[inset_0_0_12px_rgba(52,211,153,0.3)]">
              ★ 15-20
            </div>
            {/* Zone 3: >20cm */}
            <div className="w-[50%] h-full bg-sky-500/20 flex items-center justify-center text-[10px] text-sky-300/80 font-mono">
              20 - 40+ cm
            </div>

            {/* Live Indicator Needle */}
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

        {/* 2. CAMERA FEED + VISUAL EDGE OVERLAY & CORRIDOR */}
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

          {/* VISUAL 17.5 cm CORRIDOR BRACKETS (Dashed) */}
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center z-10">
            <div
              style={{ width: `${corridorWidthPct}%`, height: '55%' }}
              className={`border-2 border-dashed rounded-2xl flex flex-col justify-between p-2.5 transition-all duration-150 ${
                currentZone === 'IN_RANGE'
                  ? 'border-emerald-400 bg-emerald-500/10 shadow-[0_0_20px_rgba(52,211,153,0.3)]'
                  : 'border-white/30 bg-white/5'
              }`}
            >
              <div className="flex justify-between text-[10px] font-mono font-bold text-white/75">
                <span>[ 17.5 cm TARGET ]</span>
                <span>CORRIDOR</span>
              </div>
              <div className="text-center text-[11px] font-semibold text-white/80">
                Fit tyre tread width between brackets
              </div>
            </div>
          </div>

          {/* REAL-TIME DETECTED EDGE MARKERS (Cyan Vertical Guidelines with Handles) */}
          {/* Left Edge Guideline */}
          <div
            style={{ left: `${leftEdgeRatio * 100}%` }}
            className="absolute top-0 bottom-0 w-1 bg-cyan-400 cursor-ew-resize z-20 flex items-center justify-center shadow-[0_0_12px_rgba(34,211,238,0.9)]"
            onMouseDown={() => setDraggingHandle('left')}
            onTouchStart={() => setDraggingHandle('left')}
          >
            <div className="w-6 h-12 bg-cyan-400 hover:bg-cyan-300 text-black font-extrabold text-xs rounded-full flex items-center justify-center shadow-lg select-none">
              ◀
            </div>
          </div>

          {/* Right Edge Guideline */}
          <div
            style={{ left: `${rightEdgeRatio * 100}%` }}
            className="absolute top-0 bottom-0 w-1 bg-cyan-400 cursor-ew-resize z-20 flex items-center justify-center shadow-[0_0_12px_rgba(34,211,238,0.9)]"
            onMouseDown={() => setDraggingHandle('right')}
            onTouchStart={() => setDraggingHandle('right')}
          >
            <div className="w-6 h-12 bg-cyan-400 hover:bg-cyan-300 text-black font-extrabold text-xs rounded-full flex items-center justify-center shadow-lg select-none">
              ▶
            </div>
          </div>

          {/* Active Detected Span Bar */}
          <div
            style={{
              left: `${leftEdgeRatio * 100}%`,
              width: `${(rightEdgeRatio - leftEdgeRatio) * 100}%`,
            }}
            className="absolute top-14 h-8 border-t-2 border-b-2 border-cyan-400/80 bg-cyan-500/20 flex items-center justify-center pointer-events-none z-10 backdrop-blur-[1px]"
          >
            <span className="text-[11px] font-mono font-bold text-cyan-200">
              Active Span: {Math.round(currentPixelSpan)} px
            </span>
          </div>

          {/* REAL-TIME DISTANCE & GUIDANCE STATUS BANNER */}
          <div className="absolute top-3 inset-x-3 z-30 flex items-center justify-between">
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
                  ? '🟢 PERFECT (15–20 cm) — IN RANGE'
                  : currentZone === 'TOO_CLOSE'
                  ? '⬅ MOVE BACK (TOO CLOSE)'
                  : currentZone === 'TOO_FAR'
                  ? '➡ MOVE CLOSER (TOO FAR)'
                  : 'ALIGN TYRE EDGES'}
              </span>
            </div>

            <div className="bg-slate-950/90 border border-slate-800 px-3 py-1 rounded-xl text-right backdrop-blur-md">
              <span className="text-[10px] text-slate-400 block font-semibold">LIVE DISTANCE</span>
              <span className="text-base md:text-lg font-black font-mono text-cyan-300">
                {currentDistanceCm !== null ? `${currentDistanceCm.toFixed(1)} cm` : '--'}
              </span>
            </div>
          </div>

          {/* BOTTOM METRICS HUD */}
          <div className="absolute bottom-3 inset-x-3 z-30 flex items-center justify-between pointer-events-none">
            {/* Focus/Sharpness */}
            <div className="bg-slate-950/85 border border-slate-800 px-2.5 py-1 rounded-lg flex items-center gap-1.5 text-xs backdrop-blur-sm">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400 text-[11px]">Focus/Sharpness:</span>
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
                    : 'bg-rose-950/85 border-rose-500 text-rose-300'
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

        {/* 3. CALIBRATION & EDGE TUNING CONTROLS */}
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-2xl flex flex-wrap items-center justify-between gap-3 shadow-lg">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setAutoEdgeTracking(!autoEdgeTracking)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                autoEdgeTracking
                  ? 'bg-cyan-950/80 border-cyan-400 text-cyan-300'
                  : 'bg-slate-800 border-slate-700 text-slate-300'
              }`}
            >
              {autoEdgeTracking ? 'Auto-Edge Tracking: ON' : 'Manual Calipers: ACTIVE'}
            </button>
            <span className="text-[11px] text-slate-400 hidden sm:inline">
              (Drag ◀ ▶ handles if edges don't match)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowCalibrationDrawer(!showCalibrationDrawer)}
              className="px-3 py-1.5 rounded-lg text-xs font-bold bg-amber-500 hover:bg-amber-400 text-slate-950 transition flex items-center gap-1 shadow-md shadow-amber-950"
            >
              <Sliders className="w-3.5 h-3.5" />
              {showCalibrationDrawer ? 'Hide Calibration' : 'Fix Distance with Ruler'}
            </button>
          </div>
        </div>

        {/* 1-CLICK CALIBRATION DRAWER */}
        {showCalibrationDrawer && (
          <div className="bg-amber-950/30 border-2 border-amber-600/50 p-4 rounded-2xl flex flex-col gap-3 shadow-xl transition">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-amber-300 flex items-center gap-1.5">
                <Sliders className="w-4 h-4" /> 1-Click Lens Calibration (Eliminates Error)
              </span>
              <span className="text-xs font-mono text-amber-200">Current F: {focalLength} px</span>
            </div>

            <p className="text-xs text-amber-200/90 leading-relaxed">
              Place your camera at an exact distance (e.g. 18 cm) from the tyre using a ruler, ensure the ◀ ▶
              handles line up with the tyre edges, and click <strong>Lock Calibration</strong>.
            </p>

            <div className="flex flex-wrap items-center gap-3">
              <label className="text-xs text-amber-200 flex items-center gap-1.5 font-medium">
                Physical Distance from Lens (cm):
                <input
                  type="number"
                  min="5"
                  max="50"
                  value={calibrationDistInput}
                  onChange={(e) => setCalibrationDistInput(parseFloat(e.target.value) || 18)}
                  className="w-20 bg-slate-950 border border-amber-700/60 rounded px-2.5 py-1 text-sm text-white font-mono"
                />
              </label>

              <button
                onClick={handleLockCalibration}
                className="bg-amber-500 hover:bg-amber-400 text-slate-950 font-extrabold text-xs px-4 py-2 rounded-lg transition flex items-center gap-1.5 active:scale-95 shadow-md shadow-amber-950"
              >
                {calibrationSuccess ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-950" /> Calibrated!
                  </>
                ) : (
                  `Lock Calibration at ${calibrationDistInput} cm`
                )}
              </button>

              <button
                onClick={handleResetCalibration}
                className="text-xs text-slate-400 hover:text-white underline ml-auto flex items-center gap-1"
              >
                <RotateCcw className="w-3.5 h-3.5" /> Reset Default
              </button>
            </div>

            {/* Manual Focal Length Tuning Slider */}
            <div className="pt-2 border-t border-amber-800/40 flex items-center gap-3 text-xs text-amber-200/80">
              <span>Fine-Tune Focal Length (px):</span>
              <input
                type="range"
                min="300"
                max="1400"
                value={focalLength}
                onChange={(e) => setFocalLength(parseInt(e.target.value))}
                className="flex-1 accent-amber-400"
              />
              <span className="font-mono text-white font-bold">{focalLength} px</span>
            </div>
          </div>
        )}

        {/* 4. AUTO-RECORDING GATE CONTROLS */}
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

          <span className="text-xs text-slate-400">
            {isSessionActive
              ? currentZone === 'IN_RANGE'
                ? 'Recording active'
                : 'Auto-paused (out of 15-20 cm)'
              : 'Auto-gating ready'}
          </span>
        </div>

        {/* TYRE TREAD WIDTH PRESETS */}
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Tyre Nominal Width Setting
            </span>
            <span className="text-xs text-emerald-400 font-mono font-bold">
              Current: {targetWidthCm} cm ({targetWidthCm * 10} mm)
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {TYRE_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setTargetWidthCm(p.widthCm)}
                className={`p-2 rounded-xl border text-left flex flex-col transition ${
                  targetWidthCm === p.widthCm
                    ? 'bg-emerald-950/60 border-emerald-400 shadow-md shadow-emerald-950 text-white font-bold'
                    : 'bg-slate-800/80 border-slate-700/80 hover:bg-slate-800 text-slate-300'
                }`}
              >
                <span className="text-xs">{p.label}</span>
                <span className="text-[11px] text-slate-400">{p.widthCm} cm</span>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 pt-2 border-t border-slate-800 text-xs text-slate-400">
            <span>Custom Width (cm):</span>
            <input
              type="number"
              step="0.5"
              min="5"
              max="40"
              value={targetWidthCm}
              onChange={(e) => setTargetWidthCm(parseFloat(e.target.value) || 20.5)}
              className="w-20 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
            />
          </div>
        </div>
      </div>
    </div>
  );
}