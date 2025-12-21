import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('API Location Fetching', () => {
  beforeEach(() => {
    // Mock fetch
    global.fetch = vi.fn();
  });

  it('fetches locations from API', async () => {
    const mockData = {
      'vehicle_1': {
        name: 'Shuttle 1',
        latitude: 42.7284,
        longitude: -73.6918,
        timestamp: '2024-01-15T12:00:00Z',
        route_name: 'NORTH'
      }
    };

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockData,
    });

    const response = await fetch('/api/locations');
    const data = await response.json();

    expect(data).toEqual(mockData);
    expect(data['vehicle_1'].name).toBe('Shuttle 1');
    expect(data['vehicle_1'].latitude).toBe(42.7284);
  });

  it('handles API errors gracefully', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const response = await fetch('/api/locations');
    expect(response.ok).toBe(false);
    expect(response.status).toBe(500);
  });
});
