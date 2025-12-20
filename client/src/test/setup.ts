import { expect, afterEach, beforeAll } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

// Mock scrollIntoView which is not supported by jsdom
beforeAll(() => {
  Element.prototype.scrollIntoView = () => {};
  HTMLElement.prototype.scrollIntoView = () => {};
});

// Cleanup after each test
afterEach(() => {
  cleanup();
});
