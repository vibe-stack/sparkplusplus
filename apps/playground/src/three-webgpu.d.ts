declare module 'three/webgpu' {
  import { WebGLRenderer, type WebGLRendererParameters } from 'three';

  export class WebGPURenderer extends WebGLRenderer {
    constructor(parameters?: WebGLRendererParameters);
    init(): Promise<void>;
  }
}
