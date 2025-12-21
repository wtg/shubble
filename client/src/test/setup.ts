import { expect, afterEach, beforeAll } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

// Mock scrollIntoView which is not supported by jsdom
beforeAll(() => {
  Element.prototype.scrollIntoView = () => {};
  HTMLElement.prototype.scrollIntoView = () => {};

  // Mock DOMPoint for MapKit
  global.DOMPoint = class DOMPoint {
    x: number;
    y: number;
    z: number;
    w: number;
    constructor(x = 0, y = 0, z = 0, w = 1) {
      this.x = x;
      this.y = y;
      this.z = z;
      this.w = w;
    }
  } as any;
});

// Cleanup after each test
afterEach(() => {
  cleanup();
});
