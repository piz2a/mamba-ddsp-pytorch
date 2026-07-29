/**
 * JUCE 8 WebView 프레임워크를 위한 TypeScript 타입 선언 파일
 */

export declare class ListenerList<T = any> {
  private listeners: Map<number, (payload: T) => void>;
  private listenerId: number;

  /** 리스너를 추가하고 고유 ID를 반환합니다. */
  addListener(fn: (payload: T) => void): number;

  /** 고유 ID를 이용해 리스너를 제거합니다. */
  removeListener(id: number): void;

  /** 등록된 모든 리스너를 호출합니다. */
  callListeners(payload?: T): void;
}

export declare interface SliderProperties {
  start: number;
  end: number;
  skew: number;
  name: string;
  label: string;
  numSteps: number;
  interval: number;
  parameterIndex: number;
}

export declare class SliderState {
  readonly name: string;
  readonly identifier: string;
  private scaledValue: number;
  properties: SliderProperties;

  /** 값이 변경될 때 발생하는 이벤트 */
  valueChangedEvent: ListenerList<void>;
  /** 파라미터 속성(범위 등)이 변경될 때 발생하는 이벤트 */
  propertiesChangedEvent: ListenerList<void>;

  constructor(name: string);

  /** [0, 1] 범위의 정규화된 값을 백엔드에 설정합니다. */
  setNormalisedValue(newValue: number): void;

  /** 사용자가 드래그를 시작했음을 알립니다 (Gesture 시작). */
  sliderDragStarted(): void;

  /** 사용자가 드래그를 끝냈음을 알립니다 (Gesture 종료). */
  sliderDragEnded(): void;

  /** C++의 NormalisableRange를 거친 실제 스케일 값을 반환합니다. */
  getScaledValue(): number;

  /** [0, 1] 범위의 정규화된 값을 반환합니다. */
  getNormalisedValue(): number;

  private handleEvent(event: any): void;
  private normalisedToScaledValue(normalisedValue: number): number;
  private snapToLegalValue(value: number): number;
}

export declare interface ToggleProperties {
  name: string;
  parameterIndex: number;
}

export declare class ToggleState {
  readonly name: string;
  readonly identifier: string;
  private value: boolean;
  properties: ToggleProperties;

  valueChangedEvent: ListenerList<void>;
  propertiesChangedEvent: ListenerList<void>;

  constructor(name: string);

  /** 현재 체크 상태를 반환합니다. */
  getValue(): boolean;

  /** 체크 상태를 백엔드에 설정합니다. */
  setValue(newValue: boolean): void;

  private handleEvent(event: any): void;
}

export declare interface ComboBoxProperties {
  name: string;
  parameterIndex: number;
  choices: string[];
}

export declare class ComboBoxState {
  readonly name: string;
  readonly identifier: string;
  private value: number;
  properties: ComboBoxProperties;

  valueChangedEvent: ListenerList<void>;
  propertiesChangedEvent: ListenerList<void>;

  constructor(name: string);

  /** 선택된 아이템의 인덱스를 반환합니다. */
  getChoiceIndex(): number;

  /** 인덱스를 통해 아이템을 선택하고 백엔드에 알립니다. */
  setChoiceIndex(index: number): void;

  private handleEvent(event: any): void;
}

/**
 * C++ 백엔드에 등록된 네이티브 함수를 호출하는 바인딩 함수를 반환합니다.
 * @param name WebBrowserComponent::Options.withNativeFunction()에 등록된 이름
 */
export declare function getNativeFunction(name: string): (...args: any[]) => Promise<any>;

/** 해당 이름의 WebSliderRelay와 연결된 SliderState를 가져옵니다. */
export declare function getSliderState(name: string): SliderState;

/** 해당 이름의 WebToggleRelay와 연결된 ToggleState를 가져옵니다. */
export declare function getToggleState(name: string): ToggleState;

/** 해당 이름의 WebComboBoxRelay와 연결된 ComboBoxState를 가져옵니다. */
export declare function getComboBoxState(name: string): ComboBoxState;

/** 플랫폼별(macOS, Windows 등) 리소스 프로바이더 주소를 반환합니다. */
export declare function getBackendResourceAddress(path: string): string;

/** 마우스 이동에 따라 파라미터 인덱스 어노테이션을 업데이트하는 헬퍼 클래스 */
export declare class ControlParameterIndexUpdater {
  constructor(controlParameterIndexAnnotation: string);
  handleMouseMove(event: MouseEvent | React.MouseEvent): void;
}

// Global Window Extension
declare global {
  interface Window {
    __JUCE__: {
      backend: {
        addEventListener: (name: string, callback: (event: any) => void) => void;
        emitEvent: (name: string, payload: any) => void;
      };
      initialisationData: {
        __juce__functions: string[];
        __juce__sliders: string[];
        __juce__toggles: string[];
        __juce__comboBoxes: string[];
        __juce__platform: string[];
      };
    };
  }
}