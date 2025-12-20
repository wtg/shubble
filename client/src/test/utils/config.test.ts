import { describe, it, expect } from 'vitest';
import config from '../../ts/config';

describe('Config', () => {
  it('has required configuration properties', () => {
    expect(config).toBeDefined();
    expect(typeof config.isStaging).toBe('boolean');
  });

  it('isStaging is false by default', () => {
    expect(config.isStaging).toBe(false);
  });
});
