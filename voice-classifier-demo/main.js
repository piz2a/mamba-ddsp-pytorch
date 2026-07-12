const VOWELS = [
  { id: "ah", display: "ah", hangul: "아", ipa: "/a/", color: "#72d690", proto: [820, 1450] },
  { id: "uh", display: "uh", hangul: "어", ipa: "/ʌ/", color: "#f3c969", proto: [610, 1150] },
  { id: "oh", display: "oh", hangul: "오", ipa: "/o/", color: "#6eb8ff", proto: [500, 850] },
  { id: "woo", display: "woo", hangul: "우", ipa: "/u/", color: "#9fd9ff", proto: [340, 820] },
  { id: "eu", display: "eu", hangul: "으", ipa: "/ɯ/", color: "#bda5ff", proto: [370, 1550] },
  { id: "ee", display: "ee", hangul: "이", ipa: "/i/", color: "#fb847e", proto: [300, 2450] },
];

const ARTICS = [
  { id: "deu", display: "deu", hangul: "드", color: "#72d690" },
  { id: "dng", display: "dng", hangul: "등", color: "#f3c969" },
];

const STORAGE_KEY = "voice-style-classifier.samples.v1";
const FFT_SIZE = 4096;
const ANALYSIS_INTERVAL_MS = 58;

const els = {
  micButton: document.querySelector("#micButton"),
  resetButton: document.querySelector("#resetButton"),
  gateSlider: document.querySelector("#gateSlider"),
  statusLine: document.querySelector("#statusLine"),
  vowelLabel: document.querySelector("#vowelLabel"),
  vowelSub: document.querySelector("#vowelSub"),
  articLabel: document.querySelector("#articLabel"),
  articSub: document.querySelector("#articSub"),
  f0Value: document.querySelector("#f0Value"),
  f1Value: document.querySelector("#f1Value"),
  f2Value: document.querySelector("#f2Value"),
  vowelBars: document.querySelector("#vowelBars"),
  articBars: document.querySelector("#articBars"),
  vowelButtons: document.querySelector("#vowelButtons"),
  articButtons: document.querySelector("#articButtons"),
  vowelModelState: document.querySelector("#vowelModelState"),
  articModelState: document.querySelector("#articModelState"),
  validateButton: document.querySelector("#validateButton"),
  exportButton: document.querySelector("#exportButton"),
  validationOutput: document.querySelector("#validationOutput"),
  featureOutput: document.querySelector("#featureOutput"),
  waveCanvas: document.querySelector("#waveCanvas"),
  spectrumCanvas: document.querySelector("#spectrumCanvas"),
};

const audio = {
  context: null,
  analyser: null,
  source: null,
  stream: null,
  timeData: null,
  freqData: null,
  running: false,
};

const state = {
  gate: Number(els.gateSlider.value),
  lastAnalysisAt: 0,
  lastDrawAt: 0,
  previousFluxVector: null,
  recentFrames: [],
  capture: null,
  samples: loadSamples(),
  smoothed: {
    vowel: uniformScores(VOWELS),
    articulation: uniformScores(ARTICS),
  },
  latestFeature: null,
  barNodes: {
    vowel: new Map(),
    articulation: new Map(),
  },
};

init();

function init() {
  buildBars("vowel", els.vowelBars, VOWELS);
  buildBars("articulation", els.articBars, ARTICS);
  buildCaptureButtons("vowel", els.vowelButtons, VOWELS);
  buildCaptureButtons("articulation", els.articButtons, ARTICS);
  updateSampleState();
  updateBars("vowel", state.smoothed.vowel, VOWELS);
  updateBars("articulation", state.smoothed.articulation, ARTICS);
  drawIdle();

  els.micButton.addEventListener("click", toggleMic);
  els.resetButton.addEventListener("click", resetSamples);
  els.validateButton.addEventListener("click", () => {
    els.validationOutput.textContent = validateSamples();
  });
  els.exportButton.addEventListener("click", exportSamples);
  els.gateSlider.addEventListener("input", () => {
    state.gate = Number(els.gateSlider.value);
  });
}

async function toggleMic() {
  if (audio.running) {
    stopMic();
    return;
  }
  await startMic();
}

async function startMic() {
  try {
    audio.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
      video: false,
    });
  } catch (error) {
    setStatus(`Mic blocked: ${error.message}`);
    throw error;
  }

  audio.context = new AudioContext();
  audio.analyser = audio.context.createAnalyser();
  audio.analyser.fftSize = FFT_SIZE;
  audio.analyser.minDecibels = -100;
  audio.analyser.maxDecibels = -10;
  audio.analyser.smoothingTimeConstant = 0.08;

  audio.source = audio.context.createMediaStreamSource(audio.stream);
  audio.source.connect(audio.analyser);
  audio.timeData = new Float32Array(audio.analyser.fftSize);
  audio.freqData = new Float32Array(audio.analyser.frequencyBinCount);
  audio.running = true;

  els.micButton.textContent = "Stop mic";
  setStatus(`Mic live · ${audio.context.sampleRate} Hz`);
  requestAnimationFrame(tick);
}

function stopMic() {
  if (audio.stream) {
    for (const track of audio.stream.getTracks()) track.stop();
  }
  if (audio.context) audio.context.close();

  audio.context = null;
  audio.analyser = null;
  audio.source = null;
  audio.stream = null;
  audio.timeData = null;
  audio.freqData = null;
  audio.running = false;
  state.previousFluxVector = null;
  state.recentFrames = [];
  state.capture = null;

  els.micButton.textContent = "Start mic";
  els.vowelLabel.textContent = "-";
  els.articLabel.textContent = "-";
  els.vowelSub.textContent = "waiting";
  els.articSub.textContent = "waiting";
  setStatus("Mic idle");
  markCaptureButtons();
  drawIdle();
}

function tick(now) {
  if (!audio.running || !audio.analyser) return;
  requestAnimationFrame(tick);

  audio.analyser.getFloatTimeDomainData(audio.timeData);
  audio.analyser.getFloatFrequencyData(audio.freqData);
  drawScopes(audio.timeData, audio.freqData, state.latestFeature);

  if (now - state.lastAnalysisAt < ANALYSIS_INTERVAL_MS) return;
  state.lastAnalysisAt = now;

  const feature = extractFeatures(audio.timeData, audio.freqData, audio.context.sampleRate, state.gate);
  if (state.previousFluxVector && feature.fluxVector) {
    feature.spectralFlux = spectralFlux(state.previousFluxVector, feature.fluxVector);
  }
  state.previousFluxVector = feature.fluxVector;
  state.latestFeature = feature;

  const frame = compactFrame(feature, now);
  state.recentFrames.push(frame);
  state.recentFrames = state.recentFrames.filter((item) => now - item.t < 1100);

  collectCaptureFrame(frame, now);
  updateLiveUi(now);
}

function updateLiveUi(now) {
  const voicedFrames = state.recentFrames.filter((item) => now - item.t < 360 && item.voiced);
  const activeFrames = state.recentFrames.filter((item) => now - item.t < 760 && item.usable);
  const vowelFeature = aggregateFrames(voicedFrames, "vowel");
  const articulationFeature = aggregateFrames(activeFrames, "articulation");

  const feature = state.latestFeature;
  els.f0Value.textContent = formatHz(feature?.f0);
  els.f1Value.textContent = formatHz(feature?.f1);
  els.f2Value.textContent = formatHz(feature?.f2);
  els.featureOutput.textContent = formatFeatureOutput(feature);

  if (!feature || feature.rms < state.gate) {
    els.vowelLabel.textContent = "-";
    els.vowelSub.textContent = "below gate";
    els.articLabel.textContent = "-";
    els.articSub.textContent = "below gate";
    setStatus(state.capture ? captureStatus(now) : "Listening");
    return;
  }

  if (vowelFeature) {
    const vowelPrediction = classify("vowel", vowelFeature);
    state.smoothed.vowel = smoothScores(state.smoothed.vowel, vowelPrediction.scores, 0.45);
    renderPrediction("vowel", state.smoothed.vowel, VOWELS, vowelPrediction.source);
  }

  if (articulationFeature) {
    const articulationPrediction = classify("articulation", articulationFeature);
    state.smoothed.articulation = smoothScores(state.smoothed.articulation, articulationPrediction.scores, 0.5);
    renderPrediction("articulation", state.smoothed.articulation, ARTICS, articulationPrediction.source);
  }

  updateBars("vowel", state.smoothed.vowel, VOWELS);
  updateBars("articulation", state.smoothed.articulation, ARTICS);
  setStatus(state.capture ? captureStatus(now) : `Listening · RMS ${feature.rms.toFixed(3)}`);
}

function renderPrediction(kind, scores, labels, source) {
  const best = bestScore(scores, labels);
  const confidence = Math.round(best.value * 100);
  if (kind === "vowel") {
    els.vowelLabel.textContent = best.label.display;
    els.vowelSub.textContent = `${best.label.hangul} ${best.label.ipa} · ${confidence}% · ${source}`;
  } else {
    els.articLabel.textContent = best.label.display;
    els.articSub.textContent = `${best.label.hangul} · ${confidence}% · ${source}`;
  }
}

function extractFeatures(timeData, freqData, sampleRate, gate) {
  const rms = rootMeanSquare(timeData);
  const zcr = zeroCrossingRate(timeData);
  const pitch = estimatePitch(timeData, sampleRate, rms, gate);
  const spectral = spectralFeatures(freqData, sampleRate, FFT_SIZE);
  const formants = estimateFormants(timeData, sampleRate, rms, gate) || estimateFormantsFromSpectrum(freqData, sampleRate, FFT_SIZE);
  const f1 = formants?.[0] ?? NaN;
  const f2 = formants?.[1] ?? NaN;
  const f3 = formants?.[2] ?? NaN;
  const nasalIndex = computeNasalIndex({
    f1,
    f2,
    centroid: spectral.centroid,
    lowRatio: spectral.lowRatio,
    midRatio: spectral.midRatio,
    highRatio: spectral.highRatio,
    flatness: spectral.flatness,
  });

  return {
    rms,
    zcr,
    f0: pitch.f0,
    clarity: pitch.clarity,
    voiced: rms > gate && pitch.clarity > 0.23,
    usable: rms > gate * 0.6,
    f1,
    f2,
    f3,
    centroid: spectral.centroid,
    rolloff: spectral.rolloff,
    flatness: spectral.flatness,
    lowRatio: spectral.lowRatio,
    midRatio: spectral.midRatio,
    highRatio: spectral.highRatio,
    nasalIndex,
    spectralFlux: 0,
    fluxVector: spectral.fluxVector,
  };
}

function rootMeanSquare(data) {
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) sum += data[i] * data[i];
  return Math.sqrt(sum / data.length);
}

function zeroCrossingRate(data) {
  let crossings = 0;
  let previous = data[0] >= 0;
  for (let i = 1; i < data.length; i += 1) {
    const current = data[i] >= 0;
    if (current !== previous) crossings += 1;
    previous = current;
  }
  return crossings / data.length;
}

function estimatePitch(data, sampleRate, rms, gate) {
  if (rms < gate * 0.7) return { f0: NaN, clarity: 0 };

  const frameSize = Math.min(3072, data.length);
  let mean = 0;
  for (let i = 0; i < frameSize; i += 1) mean += data[i];
  mean /= frameSize;

  const minLag = Math.max(24, Math.floor(sampleRate / 850));
  const maxLag = Math.min(Math.floor(sampleRate / 60), frameSize - 4);
  let bestLag = -1;
  let bestCorrelation = 0;

  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let sum = 0;
    let energyA = 0;
    let energyB = 0;
    const limit = frameSize - lag;
    for (let i = 0; i < limit; i += 1) {
      const a = data[i] - mean;
      const b = data[i + lag] - mean;
      sum += a * b;
      energyA += a * a;
      energyB += b * b;
    }
    const corr = sum / Math.sqrt(energyA * energyB + 1e-12);
    if (corr > bestCorrelation) {
      bestCorrelation = corr;
      bestLag = lag;
    }
  }

  if (bestLag < 0 || bestCorrelation < 0.18) return { f0: NaN, clarity: Math.max(0, bestCorrelation) };
  return { f0: sampleRate / bestLag, clarity: bestCorrelation };
}

function spectralFeatures(freqData, sampleRate, fftSize) {
  const binHz = sampleRate / fftSize;
  const maxHz = 7000;
  const maxBin = Math.min(freqData.length - 1, Math.floor(maxHz / binHz));
  const fluxVector = new Float32Array(maxBin + 1);
  let total = 0;
  let weighted = 0;
  let low = 0;
  let mid = 0;
  let high = 0;
  let geometric = 0;
  let count = 0;

  for (let i = 1; i <= maxBin; i += 1) {
    const hz = i * binHz;
    const db = Number.isFinite(freqData[i]) ? freqData[i] : -100;
    const power = Math.pow(10, db / 10);
    fluxVector[i] = power;
    total += power;
    weighted += power * hz;
    geometric += Math.log(power + 1e-16);
    count += 1;
    if (hz < 650) low += power;
    else if (hz < 2600) mid += power;
    else high += power;
  }

  const safeTotal = total + 1e-16;
  for (let i = 1; i < fluxVector.length; i += 1) fluxVector[i] /= safeTotal;

  let cumulative = 0;
  let rolloff = 0;
  for (let i = 1; i <= maxBin; i += 1) {
    cumulative += fluxVector[i] * safeTotal;
    if (cumulative >= safeTotal * 0.85) {
      rolloff = i * binHz;
      break;
    }
  }

  const arithmeticMean = safeTotal / Math.max(1, count);
  const flatness = Math.exp(geometric / Math.max(1, count)) / (arithmeticMean + 1e-16);

  return {
    centroid: weighted / safeTotal,
    rolloff,
    flatness: clamp(flatness, 0, 1),
    lowRatio: low / safeTotal,
    midRatio: mid / safeTotal,
    highRatio: high / safeTotal,
    fluxVector,
  };
}

function spectralFlux(previous, current) {
  const length = Math.min(previous.length, current.length);
  let sum = 0;
  for (let i = 1; i < length; i += 1) {
    const diff = current[i] - previous[i];
    if (diff > 0) sum += diff;
  }
  return sum;
}

function estimateFormants(data, sampleRate, rms, gate) {
  if (rms < gate * 0.6) return null;

  const targetRate = 12000;
  const factor = Math.max(1, Math.floor(sampleRate / targetRate));
  const dsRate = sampleRate / factor;
  const sourceLength = Math.min(data.length, 4096);
  const length = Math.floor(sourceLength / factor);
  if (length < 256) return null;

  const frame = new Float64Array(length);
  let previous = 0;
  for (let i = 0; i < length; i += 1) {
    let sum = 0;
    for (let j = 0; j < factor; j += 1) {
      sum += data[i * factor + j] || 0;
    }
    const sample = sum / factor;
    const emphasized = sample - 0.97 * previous;
    previous = sample;
    const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (length - 1));
    frame[i] = emphasized * window;
  }

  const order = 12;
  const autocorr = new Float64Array(order + 1);
  for (let lag = 0; lag <= order; lag += 1) {
    let sum = 0;
    for (let i = 0; i < length - lag; i += 1) sum += frame[i] * frame[i + lag];
    autocorr[lag] = sum;
  }
  if (autocorr[0] <= 1e-9) return null;

  const coeffs = levinsonDurbin(autocorr, order);
  if (!coeffs) return null;
  const roots = durandKernerRoots(coeffs);
  const candidates = [];

  for (const root of roots) {
    if (root.im <= 0.001) continue;
    const radius = Math.hypot(root.re, root.im);
    const angle = Math.atan2(root.im, root.re);
    const freq = (angle * dsRate) / (2 * Math.PI);
    const bandwidth = -(dsRate / Math.PI) * Math.log(Math.max(radius, 1e-6));
    if (freq > 180 && freq < 4200 && bandwidth > 20 && bandwidth < 850) {
      candidates.push({ freq, bandwidth });
    }
  }

  candidates.sort((a, b) => a.freq - b.freq);
  const filtered = [];
  for (const candidate of candidates) {
    const tooClose = filtered.some((item) => Math.abs(item.freq - candidate.freq) < 180);
    if (!tooClose) filtered.push(candidate);
  }

  const f1 = filtered.find((item) => item.freq >= 200 && item.freq <= 1050)?.freq;
  const f2 = filtered.find((item) => item.freq >= Math.max(650, (f1 || 0) + 220) && item.freq <= 3300)?.freq;
  const f3 = filtered.find((item) => item.freq >= Math.max(1400, (f2 || 0) + 220) && item.freq <= 4100)?.freq;

  if (!f1 || !f2) return null;
  return [f1, f2, f3].filter(Boolean);
}

function levinsonDurbin(r, order) {
  const a = new Float64Array(order + 1);
  a[0] = 1;
  let error = r[0];

  for (let i = 1; i <= order; i += 1) {
    let acc = r[i];
    for (let j = 1; j < i; j += 1) acc += a[j] * r[i - j];
    const reflection = -acc / (error + 1e-12);
    if (!Number.isFinite(reflection) || Math.abs(reflection) >= 1) return null;

    const previous = a.slice();
    a[i] = reflection;
    for (let j = 1; j < i; j += 1) a[j] = previous[j] + reflection * previous[i - j];
    error *= 1 - reflection * reflection;
    if (error <= 1e-12) return null;
  }

  return Array.from(a);
}

function durandKernerRoots(coeffs) {
  const degree = coeffs.length - 1;
  const roots = [];
  const radius = 0.88;
  for (let i = 0; i < degree; i += 1) {
    const angle = (2 * Math.PI * i) / degree;
    roots.push({ re: radius * Math.cos(angle), im: radius * Math.sin(angle) });
  }

  for (let iteration = 0; iteration < 64; iteration += 1) {
    let maxDelta = 0;
    for (let i = 0; i < degree; i += 1) {
      let denom = { re: 1, im: 0 };
      for (let j = 0; j < degree; j += 1) {
        if (i === j) continue;
        denom = complexMul(denom, complexSub(roots[i], roots[j]));
      }
      if (complexAbs(denom) < 1e-12) denom = { re: 1e-12, im: 0 };
      const value = evalPolynomial(coeffs, roots[i]);
      const delta = complexDiv(value, denom);
      roots[i] = complexSub(roots[i], delta);
      maxDelta = Math.max(maxDelta, complexAbs(delta));
    }
    if (maxDelta < 1e-8) break;
  }

  return roots;
}

function evalPolynomial(coeffs, z) {
  let result = { re: coeffs[0], im: 0 };
  for (let i = 1; i < coeffs.length; i += 1) {
    result = complexAdd(complexMul(result, z), { re: coeffs[i], im: 0 });
  }
  return result;
}

function complexAdd(a, b) {
  return { re: a.re + b.re, im: a.im + b.im };
}

function complexSub(a, b) {
  return { re: a.re - b.re, im: a.im - b.im };
}

function complexMul(a, b) {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}

function complexDiv(a, b) {
  const denom = b.re * b.re + b.im * b.im + 1e-18;
  return { re: (a.re * b.re + a.im * b.im) / denom, im: (a.im * b.re - a.re * b.im) / denom };
}

function complexAbs(a) {
  return Math.hypot(a.re, a.im);
}

function estimateFormantsFromSpectrum(freqData, sampleRate, fftSize) {
  const binHz = sampleRate / fftSize;
  const smooth = [];
  const maxBin = Math.min(freqData.length - 2, Math.floor(3800 / binHz));
  const radius = 10;

  for (let i = 0; i <= maxBin; i += 1) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(1, i - radius); j <= Math.min(maxBin, i + radius); j += 1) {
      sum += Number.isFinite(freqData[j]) ? freqData[j] : -100;
      count += 1;
    }
    smooth[i] = sum / count;
  }

  const peaks = [];
  for (let i = Math.floor(180 / binHz); i <= maxBin; i += 1) {
    const hz = i * binHz;
    if (smooth[i] > smooth[i - 1] && smooth[i] >= smooth[i + 1] && smooth[i] > -92) {
      peaks.push({ hz, db: smooth[i] });
    }
  }

  peaks.sort((a, b) => b.db - a.db);
  const selected = [];
  for (const peak of peaks) {
    if (selected.every((item) => Math.abs(item.hz - peak.hz) > 260)) selected.push(peak);
    if (selected.length >= 5) break;
  }
  selected.sort((a, b) => a.hz - b.hz);

  const f1 = selected.find((item) => item.hz >= 220 && item.hz <= 1050)?.hz;
  const f2 = selected.find((item) => item.hz >= Math.max(650, (f1 || 0) + 280) && item.hz <= 3300)?.hz;
  const f3 = selected.find((item) => item.hz >= Math.max(1400, (f2 || 0) + 300) && item.hz <= 3800)?.hz;
  if (!f1 || !f2) return null;
  return [f1, f2, f3].filter(Boolean);
}

function computeNasalIndex(feature) {
  const lowDominance = scale(feature.lowRatio - feature.midRatio, -0.2, 0.55);
  const lowCentroid = 1 - scale(feature.centroid, 450, 2100);
  const lowF2 = Number.isFinite(feature.f2) ? 1 - scale(feature.f2, 850, 1750) : 0.45;
  const darkSpectrum = 1 - scale(feature.highRatio, 0.03, 0.18);
  const smoothness = 1 - scale(feature.flatness, 0.05, 0.55);
  return clamp(0.32 * lowDominance + 0.24 * lowCentroid + 0.2 * lowF2 + 0.14 * darkSpectrum + 0.1 * smoothness, 0, 1);
}

function classify(kind, feature) {
  const labels = kind === "vowel" ? VOWELS : ARTICS;
  const heuristic = kind === "vowel" ? heuristicVowel(feature) : heuristicArticulation(feature);
  const samplePrediction = predictFromSamples(kind, feature, state.samples[kind]);
  if (!samplePrediction) return { scores: heuristic, source: "heuristic" };

  const coverage = modelCoverage(kind);
  const weight = coverage >= 1 ? 1 : 0.35 + coverage * 0.5;
  return {
    scores: blendScores(heuristic, samplePrediction, labels, weight),
    source: coverage >= 1 ? "calibrated" : "hybrid",
  };
}

function heuristicVowel(feature) {
  const scores = {};
  const f1 = sane(feature.f1, 520);
  const f2 = sane(feature.f2, 1400);
  const centroid = sane(feature.centroid, 1500);

  for (const label of VOWELS) {
    const [pf1, pf2] = label.proto;
    let d1 = (f1 - pf1) / 155;
    let d2 = (f2 - pf2) / 420;
    if (label.id === "ee") d2 = (f2 - pf2) / 520;
    if (label.id === "ah") d1 = (f1 - pf1) / 190;
    let score = Math.exp(-0.5 * (d1 * d1 + d2 * d2));

    if (label.id === "ee" && f2 > 1900 && f1 < 480) score *= 1.55;
    if (label.id === "ah" && f1 > 680) score *= 1.45;
    if (label.id === "woo" && f1 < 430 && f2 < 1150) score *= 1.35;
    if (label.id === "eu" && f1 < 500 && f2 > 1150 && f2 < 1950) score *= 1.35;
    if (label.id === "oh" && f1 > 410 && f1 < 610 && f2 < 1150) score *= 1.22;
    if (label.id === "uh" && f1 > 500 && f1 < 740 && f2 < 1450) score *= 1.18;
    if (centroid > 2300 && label.id !== "ee") score *= 0.82;
    scores[label.id] = score;
  }

  return normalizeScores(scores, VOWELS);
}

function heuristicArticulation(feature) {
  const nasal = sane(feature.nasalIndex, 0.35);
  const f2 = sane(feature.f2, 1400);
  const flux = sane(feature.spectralFlux, 0);
  const lowRatio = sane(feature.lowRatio, 0.35);
  const centroid = sane(feature.centroid, 1400);

  let dng = 0.55 * nasal + 0.18 * (1 - scale(f2, 850, 1800)) + 0.15 * lowRatio + 0.12 * (1 - scale(centroid, 550, 1900));
  let deu = 1 - dng;

  if (f2 > 1250 && nasal < 0.55) deu += 0.18;
  if (flux > 0.11) {
    deu += 0.04;
    dng += 0.04;
  }
  if (nasal > 0.62) dng += 0.25;

  return normalizeScores({ deu, dng }, ARTICS);
}

function predictFromSamples(kind, feature, samples, omitIndex = -1) {
  const usable = samples
    .map((sample, index) => ({ ...sample, index }))
    .filter((sample) => sample.index !== omitIndex && sample.feature);
  if (usable.length === 0) return null;

  const labels = kind === "vowel" ? VOWELS : ARTICS;
  const vector = featureVector(kind, feature);
  const distances = usable
    .map((sample) => ({
      label: sample.label,
      distance: euclideanDistance(vector, featureVector(kind, sample.feature)),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, Math.min(5, usable.length));

  const scores = Object.fromEntries(labels.map((label) => [label.id, 1e-6]));
  const sigma = kind === "vowel" ? 0.34 : 0.36;
  for (const item of distances) {
    scores[item.label] += Math.exp(-(item.distance * item.distance) / (2 * sigma * sigma));
  }
  return normalizeScores(scores, labels);
}

function featureVector(kind, feature) {
  if (kind === "vowel") {
    return [
      scale(sane(feature.f1, 520), 220, 950),
      scale(sane(feature.f2, 1400), 650, 3100),
      scale(sane(feature.f2, 1400) - sane(feature.f1, 520), 250, 2600),
      scale(sane(feature.centroid, 1500), 500, 3300),
      clamp(sane(feature.lowRatio, 0.3), 0, 1),
      clamp(sane(feature.midRatio, 0.45), 0, 1),
      clamp(sane(feature.highRatio, 0.2), 0, 1),
    ];
  }

  return [
    scale(sane(feature.f1, 500), 180, 950),
    scale(sane(feature.f2, 1300), 550, 2800),
    scale(sane(feature.centroid, 1300), 350, 3600),
    scale(sane(feature.rolloff, 3200), 700, 6500),
    clamp(sane(feature.lowRatio, 0.35), 0, 1),
    clamp(sane(feature.midRatio, 0.45), 0, 1),
    clamp(sane(feature.highRatio, 0.2), 0, 1),
    scale(sane(feature.zcr, 0.05), 0.005, 0.19),
    scale(sane(feature.spectralFlux, 0), 0.015, 0.2),
    clamp(sane(feature.nasalIndex, 0.35), 0, 1),
  ];
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum / a.length);
}

function compactFrame(feature, t) {
  const { fluxVector, ...scalarFeature } = feature;
  return { ...scalarFeature, t };
}

function aggregateFrames(frames, kind) {
  if (!frames.length) return null;
  const usable = frames.filter((frame) => {
    if (kind === "vowel") return frame.voiced && Number.isFinite(frame.f1) && Number.isFinite(frame.f2);
    return frame.usable;
  });
  const source = usable.length ? usable : frames;
  const medianKeys = ["rms", "zcr", "f0", "clarity", "f1", "f2", "f3", "centroid", "rolloff", "flatness", "lowRatio", "midRatio", "highRatio", "nasalIndex"];
  const aggregate = {};
  for (const key of medianKeys) aggregate[key] = median(source.map((frame) => frame[key]).filter(Number.isFinite));
  aggregate.spectralFlux = Math.max(...source.map((frame) => sane(frame.spectralFlux, 0)));
  aggregate.voiced = source.some((frame) => frame.voiced);
  aggregate.usable = source.some((frame) => frame.usable);
  aggregate.frameCount = source.length;
  return aggregate;
}

async function startCapture(kind, label) {
  if (!audio.running) await startMic();
  const duration = kind === "vowel" ? 950 : 1150;
  const now = performance.now();
  state.capture = {
    kind,
    label,
    frames: [],
    startedAt: now,
    endsAt: now + duration,
  };
  markCaptureButtons();
}

function collectCaptureFrame(frame, now) {
  if (!state.capture) return;
  if (frame.usable || frame.rms > state.gate * 0.45) state.capture.frames.push(frame);
  if (now >= state.capture.endsAt) finishCapture();
}

function finishCapture() {
  const capture = state.capture;
  state.capture = null;
  markCaptureButtons();
  if (!capture) return;

  const feature = aggregateFrames(capture.frames, capture.kind);
  if (!feature || feature.frameCount < 4) {
    setStatus(`${capture.label}: too quiet`);
    return;
  }

  state.samples[capture.kind].push({
    label: capture.label,
    feature,
    createdAt: new Date().toISOString(),
  });
  saveSamples();
  updateSampleState();
  els.validationOutput.textContent = validateSamples();
  setStatus(`${capture.label}: saved`);
}

function captureStatus(now) {
  const remaining = Math.max(0, state.capture.endsAt - now);
  return `Recording ${state.capture.label} · ${(remaining / 1000).toFixed(1)}s`;
}

function buildCaptureButtons(kind, container, labels) {
  container.textContent = "";
  for (const label of labels) {
    const button = document.createElement("button");
    button.className = "capture-button";
    button.type = "button";
    button.dataset.kind = kind;
    button.dataset.label = label.id;
    button.innerHTML = `<strong>${label.display}</strong><span>${label.hangul} · 0</span>`;
    button.addEventListener("click", () => startCapture(kind, label.id));
    container.append(button);
  }
}

function markCaptureButtons() {
  for (const button of document.querySelectorAll(".capture-button")) {
    const active =
      state.capture &&
      button.dataset.kind === state.capture.kind &&
      button.dataset.label === state.capture.label;
    button.classList.toggle("recording", Boolean(active));
    button.disabled = Boolean(state.capture && !active);
  }
}

function buildBars(kind, container, labels) {
  container.textContent = "";
  state.barNodes[kind].clear();
  for (const label of labels) {
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `
      <span class="bar__name">${label.display}</span>
      <span class="bar__track"><span class="bar__fill"></span></span>
      <span class="bar__value">0%</span>
    `;
    row.querySelector(".bar__fill").style.background = label.color;
    container.append(row);
    state.barNodes[kind].set(label.id, {
      fill: row.querySelector(".bar__fill"),
      value: row.querySelector(".bar__value"),
    });
  }
}

function updateBars(kind, scores, labels) {
  for (const label of labels) {
    const node = state.barNodes[kind].get(label.id);
    const value = clamp(scores[label.id] || 0, 0, 1);
    node.fill.style.width = `${Math.round(value * 100)}%`;
    node.value.textContent = `${Math.round(value * 100)}%`;
  }
}

function updateSampleState() {
  updateCaptureCounts("vowel", VOWELS);
  updateCaptureCounts("articulation", ARTICS);
  const vowelCounts = countsFor("vowel");
  const articCounts = countsFor("articulation");
  els.vowelModelState.textContent = modelCoverage("vowel") >= 1 ? "calibrated" : countSummary(vowelCounts, VOWELS);
  els.articModelState.textContent = modelCoverage("articulation") >= 1 ? "calibrated" : countSummary(articCounts, ARTICS);
}

function updateCaptureCounts(kind, labels) {
  const counts = countsFor(kind);
  for (const label of labels) {
    const button = document.querySelector(`.capture-button[data-kind="${kind}"][data-label="${label.id}"]`);
    if (button) button.querySelector("span").textContent = `${label.hangul} · ${counts[label.id] || 0}`;
  }
}

function countSummary(counts, labels) {
  const total = labels.reduce((sum, label) => sum + (counts[label.id] || 0), 0);
  if (!total) return "heuristic";
  const ready = labels.filter((label) => counts[label.id] > 0).length;
  return `${ready}/${labels.length} labels`;
}

function countsFor(kind) {
  const labels = kind === "vowel" ? VOWELS : ARTICS;
  const counts = Object.fromEntries(labels.map((label) => [label.id, 0]));
  for (const sample of state.samples[kind]) counts[sample.label] = (counts[sample.label] || 0) + 1;
  return counts;
}

function modelCoverage(kind) {
  const labels = kind === "vowel" ? VOWELS : ARTICS;
  const counts = countsFor(kind);
  return labels.filter((label) => counts[label.id] > 0).length / labels.length;
}

function validateSamples() {
  return [validateKind("vowel", VOWELS), validateKind("articulation", ARTICS)].join("\n\n");
}

function validateKind(kind, labels) {
  const samples = state.samples[kind];
  const counts = countsFor(kind);
  if (!samples.length) return `${kind}: no samples`;

  const matrix = Object.fromEntries(
    labels.map((row) => [row.id, Object.fromEntries(labels.map((col) => [col.id, 0]))]),
  );
  let correct = 0;
  let tested = 0;

  for (let i = 0; i < samples.length; i += 1) {
    if (samples.length < 2) continue;
    const prediction = predictFromSamples(kind, samples[i].feature, samples, i);
    if (!prediction) continue;
    const best = bestScore(prediction, labels).label.id;
    matrix[samples[i].label][best] += 1;
    if (best === samples[i].label) correct += 1;
    tested += 1;
  }

  const header = `${kind}: ${samples.length} samples · ${tested ? `${Math.round((correct / tested) * 100)}% LOO` : "need more"}`;
  const countLine = labels.map((label) => `${label.id}:${counts[label.id] || 0}`).join(" ");
  const rows = labels
    .filter((label) => counts[label.id] > 0)
    .map((label) => {
      const cells = labels.map((col) => `${col.id}=${matrix[label.id][col.id]}`).join(" ");
      return `${label.id.padEnd(5)} ${cells}`;
    });
  return [header, countLine, ...rows].join("\n");
}

function resetSamples() {
  state.samples = { vowel: [], articulation: [] };
  localStorage.removeItem(STORAGE_KEY);
  updateSampleState();
  els.validationOutput.textContent = "No samples yet.";
  setStatus("Calibration reset");
}

function exportSamples() {
  const payload = {
    exportedAt: new Date().toISOString(),
    samples: state.samples,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "voice-classifier-samples.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function loadSamples() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return {
      vowel: Array.isArray(parsed.vowel) ? parsed.vowel : [],
      articulation: Array.isArray(parsed.articulation) ? parsed.articulation : [],
    };
  } catch {
    return { vowel: [], articulation: [] };
  }
}

function saveSamples() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.samples));
}

function uniformScores(labels) {
  const value = 1 / labels.length;
  return Object.fromEntries(labels.map((label) => [label.id, value]));
}

function normalizeScores(scores, labels) {
  const cleaned = {};
  let total = 0;
  for (const label of labels) {
    const value = Math.max(0, sane(scores[label.id], 0));
    cleaned[label.id] = value;
    total += value;
  }
  if (total <= 1e-9) return uniformScores(labels);
  for (const label of labels) cleaned[label.id] /= total;
  return cleaned;
}

function blendScores(base, calibrated, labels, calibratedWeight) {
  const scores = {};
  for (const label of labels) {
    scores[label.id] = base[label.id] * (1 - calibratedWeight) + calibrated[label.id] * calibratedWeight;
  }
  return normalizeScores(scores, labels);
}

function smoothScores(previous, next, alpha) {
  const scores = {};
  for (const key of Object.keys(next)) scores[key] = previous[key] * (1 - alpha) + next[key] * alpha;
  return scores;
}

function bestScore(scores, labels) {
  let best = labels[0];
  let value = -Infinity;
  for (const label of labels) {
    if ((scores[label.id] || 0) > value) {
      best = label;
      value = scores[label.id] || 0;
    }
  }
  return { label: best, value };
}

function median(values) {
  const usable = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!usable.length) return NaN;
  const middle = Math.floor(usable.length / 2);
  if (usable.length % 2) return usable[middle];
  return (usable[middle - 1] + usable[middle]) / 2;
}

function sane(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

function scale(value, min, max) {
  return clamp((value - min) / (max - min), 0, 1);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatHz(value) {
  return Number.isFinite(value) ? `${Math.round(value)} Hz` : "- Hz";
}

function formatFeatureOutput(feature) {
  if (!feature) return "-";
  return [
    `rms           ${feature.rms.toFixed(4)}`,
    `zcr           ${feature.zcr.toFixed(4)}`,
    `f0            ${formatHz(feature.f0)}`,
    `clarity       ${feature.clarity.toFixed(3)}`,
    `f1/f2/f3      ${formatHz(feature.f1)} · ${formatHz(feature.f2)} · ${formatHz(feature.f3)}`,
    `centroid      ${formatHz(feature.centroid)}`,
    `rolloff       ${formatHz(feature.rolloff)}`,
    `low/mid/high  ${feature.lowRatio.toFixed(2)} · ${feature.midRatio.toFixed(2)} · ${feature.highRatio.toFixed(2)}`,
    `nasal         ${feature.nasalIndex.toFixed(2)}`,
    `flux          ${feature.spectralFlux.toFixed(3)}`,
  ].join("\n");
}

function setStatus(text) {
  els.statusLine.textContent = text;
}

function drawIdle() {
  clearCanvas(els.waveCanvas, "waveform");
  clearCanvas(els.spectrumCanvas, "spectrum");
}

function clearCanvas(canvas, label) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#181d22";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#768391";
  ctx.font = "15px ui-sans-serif, system-ui";
  ctx.fillText(label, 18, 28);
}

function drawScopes(timeData, freqData, feature) {
  drawWaveform(timeData);
  drawSpectrum(freqData, feature);
}

function drawWaveform(data) {
  const canvas = els.waveCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#151a1f";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
  ctx.strokeStyle = "#72d690";
  ctx.lineWidth = 2;
  ctx.beginPath();
  const step = data.length / width;
  for (let x = 0; x < width; x += 1) {
    const index = Math.floor(x * step);
    const y = height / 2 + data[index] * height * 0.42;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#768391";
  ctx.font = "15px ui-sans-serif, system-ui";
  ctx.fillText("waveform", 18, 28);
}

function drawSpectrum(freqData, feature) {
  const canvas = els.spectrumCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const sampleRate = audio.context?.sampleRate || 48000;
  const binHz = sampleRate / FFT_SIZE;
  const maxHz = 4200;
  const maxBin = Math.min(freqData.length - 1, Math.floor(maxHz / binHz));
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#151a1f";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i += 1) {
    const x = (i / 4) * width;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  ctx.strokeStyle = "#6eb8ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 1; i <= maxBin; i += 1) {
    const hz = i * binHz;
    const x = (hz / maxHz) * width;
    const db = Number.isFinite(freqData[i]) ? freqData[i] : -100;
    const y = height - scale(db, -100, -18) * height;
    if (i === 1) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  drawFormantLine(ctx, width, height, maxHz, feature?.f1, "F1", "#f3c969");
  drawFormantLine(ctx, width, height, maxHz, feature?.f2, "F2", "#fb847e");
  ctx.fillStyle = "#768391";
  ctx.font = "15px ui-sans-serif, system-ui";
  ctx.fillText("spectrum", 18, 28);
}

function drawFormantLine(ctx, width, height, maxHz, hz, label, color) {
  if (!Number.isFinite(hz) || hz <= 0 || hz > maxHz) return;
  const x = (hz / maxHz) * width;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, height);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = "13px ui-sans-serif, system-ui";
  ctx.fillText(label, Math.min(width - 28, x + 5), 48);
}
