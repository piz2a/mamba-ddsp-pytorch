import { useEffect, useMemo, useRef, useState } from 'react';
import { getSliderState } from './juce';

const STYLES = [
  { name: 'Finger', short: 'FINGER', color: '#66d9a8' },
  { name: 'Muted', short: 'MUTED', color: '#b7bec8' },
  { name: 'Pick', short: 'PICK', color: '#ffcc66' },
  { name: 'Slap Auto', short: 'SLAP AUTO', color: '#ff6b6b' },
  { name: 'Slap Pop', short: 'POP', color: '#ef8cff' },
  { name: 'Slap Thumb', short: 'THUMB', color: '#57c7ff' },
  { name: 'Dead Note', short: 'DEAD', color: '#d5a86e' },
];

const ARTICULATIONS = ['Finger', 'Muted', 'Pick', 'Slap Pop', 'Slap Thumb', 'Dead Note'];
const HISTORY = 150;

type Frame = {
  inputLevel: number;
  f0Hz: number;
  periodicity: number;
  onset: number;
  offset: number;
  gate: number;
  noteAge: number;
  articulation: number;
  inferenceMs: number;
};

const emptyFrame: Frame = {
  inputLevel: 0,
  f0Hz: 0,
  periodicity: 0,
  onset: 0,
  offset: 0,
  gate: 0,
  noteAge: 0,
  articulation: 0,
  inferenceMs: 0,
};

function useStyleParameter() {
  const relay = useMemo(() => getSliderState('style'), []);
  const [value, setValue] = useState(() => Math.round(relay.getScaledValue()));

  useEffect(() => {
    const listener = relay.valueChangedEvent.addListener(() => {
      setValue(Math.round(relay.getScaledValue()));
    });
    return () => relay.valueChangedEvent.removeListener(listener);
  }, [relay]);

  const set = (next: number) => {
    const index = Math.max(0, Math.min(STYLES.length - 1, Math.round(next)));
    relay.sliderDragStarted();
    relay.setNormalisedValue(index / (STYLES.length - 1));
    relay.sliderDragEnded();
    setValue(index);
  };
  return [value, set] as const;
}

function StyleKnob() {
  const [value, setValue] = useStyleParameter();
  const startY = useRef(0);
  const startValue = useRef(value);
  const rotation = -135 + (value / 6) * 270;
  const selected = STYLES[value];

  const beginDrag = (event: React.PointerEvent) => {
    startY.current = event.clientY;
    startValue.current = value;
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const drag = (event: React.PointerEvent) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const steps = Math.round((startY.current - event.clientY) / 18);
    setValue(startValue.current + steps);
  };

  return (
    <section className="knob-section" aria-label="Playing style">
      <div className="knob-label">PLAYING STYLE</div>
      <div
        className="knob"
        role="slider"
        tabIndex={0}
        aria-valuemin={0}
        aria-valuemax={6}
        aria-valuenow={value}
        aria-valuetext={selected.name}
        onPointerDown={beginDrag}
        onPointerMove={drag}
        onWheel={(event) => {
          event.preventDefault();
          setValue(value + (event.deltaY > 0 ? -1 : 1));
        }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowUp' || event.key === 'ArrowRight') setValue(value + 1);
          if (event.key === 'ArrowDown' || event.key === 'ArrowLeft') setValue(value - 1);
        }}
        style={{ '--accent': selected.color } as React.CSSProperties}
      >
        <div className="ticks">
          {STYLES.map((style, index) => (
            <button
              key={style.name}
              className={index === value ? 'tick active' : 'tick'}
              style={{ transform: `rotate(${-135 + index * 45}deg)` }}
              onClick={() => setValue(index)}
              title={style.name}
              aria-label={style.name}
            />
          ))}
        </div>
        <div className="knob-face">
          <div className="indicator" style={{ transform: `rotate(${rotation}deg)` }}>
            <span />
          </div>
          <strong>{selected.short}</strong>
          <small>{value + 1} / 7</small>
        </div>
      </div>
      <div className="style-scale">
        {STYLES.map((style, index) => (
          <button
            key={style.name}
            className={index === value ? 'selected' : ''}
            onClick={() => setValue(index)}
          >
            {style.name}
          </button>
        ))}
      </div>
    </section>
  );
}

function points(values: number[], min: number, max: number) {
  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100;
      const y = 36 - ((Math.max(min, Math.min(max, value)) - min) / (max - min)) * 34;
      return `${x},${y}`;
    })
    .join(' ');
}

function LiveGraph({ frames }: { frames: Frame[] }) {
  const f0 = frames.map((frame) => frame.f0Hz);
  const periodicity = frames.map((frame) => frame.periodicity);
  const gate = frames.map((frame) => frame.gate);
  const latest = frames.at(-1) ?? emptyFrame;

  return (
    <section className="monitor">
      <div className="monitor-header">
        <div>
          <span className={latest.gate ? 'status active' : 'status'} />
          {latest.gate ? 'NOTE ACTIVE' : 'LISTENING'}
        </div>
        <div>{latest.f0Hz > 0 ? `${latest.f0Hz.toFixed(1)} Hz` : '— Hz'}</div>
        <div>{ARTICULATIONS[latest.articulation] ?? 'Finger'}</div>
      </div>
      <div className="lane">
        <span>F0</span>
        <svg viewBox="0 0 100 38" preserveAspectRatio="none">
          <polyline points={points(f0, 30, 330)} className="pitch-line" />
          <polyline points={points(periodicity, 0, 1)} className="periodicity-line" />
        </svg>
      </div>
      <div className="lane gate-lane">
        <span>NOTES</span>
        <svg viewBox="0 0 100 38" preserveAspectRatio="none">
          <polyline points={points(gate, 0, 1)} className="gate-line" />
          {frames.map((frame, index) =>
            frame.onset ? (
              <line key={`on-${index}`} x1={(index / HISTORY) * 100} x2={(index / HISTORY) * 100} y1="2" y2="36" className="onset-mark" />
            ) : frame.offset ? (
              <line key={`off-${index}`} x1={(index / HISTORY) * 100} x2={(index / HISTORY) * 100} y1="2" y2="36" className="offset-mark" />
            ) : null,
          )}
        </svg>
      </div>
      <div className="readouts">
        <div><span>PERIODICITY</span><strong>{latest.periodicity.toFixed(2)}</strong></div>
        <div><span>NOTE AGE</span><strong>{latest.noteAge.toFixed(2)} s</strong></div>
        <div><span>ONNX + DSP</span><strong>{latest.inferenceMs.toFixed(2)} ms</strong></div>
        <div><span>FIXED LATENCY</span><strong>64 ms</strong></div>
      </div>
    </section>
  );
}

export default function App() {
  const [frames, setFrames] = useState<Frame[]>(() => Array(HISTORY).fill(emptyFrame));

  useEffect(() => {
    window.__JUCE__.backend.addEventListener('controlFrame', (frame: Frame) => {
      setFrames((previous) => [...previous.slice(-(HISTORY - 1)), frame]);
    });
  }, []);

  return (
    <main>
      <header>
        <div>
          <p>JIHOAUDIO / BASS-DDSP</p>
          <h1>SCAT TO BASS</h1>
        </div>
        <div className="engine-badge"><span /> NATIVE ENGINE</div>
      </header>
      <div className="workspace">
        <StyleKnob />
        <LiveGraph frames={frames} />
      </div>
      <footer>
        <span>Voice in</span>
        <i />
        <span>TorchCREPE</span>
        <i />
        <span>Aubio Complex</span>
        <i />
        <span>Cached-DCT Bass-DDSP</span>
      </footer>
    </main>
  );
}
