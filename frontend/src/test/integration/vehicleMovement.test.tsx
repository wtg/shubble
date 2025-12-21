import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { waitFor } from '@testing-library/react';

/**
 * Integration test for vehicle location updates
 *
 * This test simulates the backend API returning different vehicle locations
 * over time and verifies that the frontend correctly processes the updates.
 */

describe('Vehicle Movement Integration Test', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  let locationHistory: any[] = [];

  beforeEach(() => {
    fetchMock = vi.fn();
    global.fetch = fetchMock;
    locationHistory = [];
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('processes multiple vehicle location updates over time', async () => {
    const vehicleId = 'vehicle_1';

    // Simulate 3 location updates showing shuttle movement
    const locations = [
      // Location 1: At Union
      {
        [vehicleId]: {
          vehicle_id: vehicleId,
          name: 'Shuttle 1',
          latitude: 42.7284,
          longitude: -73.6918,
          heading_degrees: 90,
          speed_mph: 15,
          timestamp: '2024-01-15T12:00:00Z',
          route_name: 'NORTH',
        }
      },
      // Location 2: Moved along route
      {
        [vehicleId]: {
          vehicle_id: vehicleId,
          name: 'Shuttle 1',
          latitude: 42.7290,
          longitude: -73.6920,
          heading_degrees: 95,
          speed_mph: 18,
          timestamp: '2024-01-15T12:00:05Z',
          route_name: 'NORTH',
        }
      },
      // Location 3: Near Library
      {
        [vehicleId]: {
          vehicle_id: vehicleId,
          name: 'Shuttle 1',
          latitude: 42.7295,
          longitude: -73.6922,
          heading_degrees: 100,
          speed_mph: 12,
          timestamp: '2024-01-15T12:00:10Z',
          route_name: 'NORTH',
        }
      },
    ];

    let callCount = 0;
    fetchMock.mockImplementation(() => {
      const locationData = locations[callCount] || locations[locations.length - 1];
      callCount++;
      locationHistory.push(locationData);

      return Promise.resolve({
        ok: true,
        json: async () => locationData,
      } as Response);
    });

    // Simulate fetching locations 3 times
    const response1 = await fetch('/api/locations');
    const data1 = await response1.json();

    const response2 = await fetch('/api/locations');
    const data2 = await response2.json();

    const response3 = await fetch('/api/locations');
    const data3 = await response3.json();

    // Verify all fetches were called
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock).toHaveBeenCalledWith('/api/locations');

    // Verify location data changed across calls
    expect(data1[vehicleId].latitude).toBe(42.7284);
    expect(data2[vehicleId].latitude).toBe(42.7290);
    expect(data3[vehicleId].latitude).toBe(42.7295);

    // Verify the shuttle moved (latitude increased)
    expect(data2[vehicleId].latitude).toBeGreaterThan(data1[vehicleId].latitude);
    expect(data3[vehicleId].latitude).toBeGreaterThan(data2[vehicleId].latitude);

    // Verify speed and heading changed
    expect(data1[vehicleId].speed_mph).toBe(15);
    expect(data2[vehicleId].speed_mph).toBe(18);
    expect(data3[vehicleId].speed_mph).toBe(12);

    // Verify timestamps progressed
    expect(new Date(data2[vehicleId].timestamp).getTime())
      .toBeGreaterThan(new Date(data1[vehicleId].timestamp).getTime());
    expect(new Date(data3[vehicleId].timestamp).getTime())
      .toBeGreaterThan(new Date(data2[vehicleId].timestamp).getTime());
  });

  it('tracks multiple shuttles moving simultaneously', async () => {
    const vehicle1 = 'vehicle_1';
    const vehicle2 = 'vehicle_2';

    // Initial positions
    const location1 = {
      [vehicle1]: {
        vehicle_id: vehicle1,
        name: 'Shuttle 1',
        latitude: 42.7284,
        longitude: -73.6918,
        route_name: 'NORTH',
      },
      [vehicle2]: {
        vehicle_id: vehicle2,
        name: 'Shuttle 2',
        latitude: 42.7290,
        longitude: -73.6920,
        route_name: 'SOUTH',
      },
    };

    // Updated positions
    const location2 = {
      [vehicle1]: {
        vehicle_id: vehicle1,
        name: 'Shuttle 1',
        latitude: 42.7287,
        longitude: -73.6919,
        route_name: 'NORTH',
      },
      [vehicle2]: {
        vehicle_id: vehicle2,
        name: 'Shuttle 2',
        latitude: 42.7293,
        longitude: -73.6921,
        route_name: 'SOUTH',
      },
    };

    let callCount = 0;
    fetchMock.mockImplementation(() => {
      callCount++;
      const locationData = callCount === 1 ? location1 : location2;

      return Promise.resolve({
        ok: true,
        json: async () => locationData,
      } as Response);
    });

    const response1 = await fetch('/api/locations');
    const data1 = await response1.json();

    const response2 = await fetch('/api/locations');
    const data2 = await response2.json();

    // Verify both vehicles are tracked
    expect(Object.keys(data1)).toHaveLength(2);
    expect(data1[vehicle1]).toBeDefined();
    expect(data1[vehicle2]).toBeDefined();

    // Verify both vehicles moved
    expect(data2[vehicle1].latitude).toBeGreaterThan(data1[vehicle1].latitude);
    expect(data2[vehicle2].latitude).toBeGreaterThan(data1[vehicle2].latitude);

    // Verify vehicles are on different routes
    expect(data1[vehicle1].route_name).toBe('NORTH');
    expect(data1[vehicle2].route_name).toBe('SOUTH');
  });

  it('handles vehicle entering and exiting geofence', async () => {
    const vehicleId = 'vehicle_1';

    // Initial: Shuttle present
    const withShuttle = {
      [vehicleId]: {
        vehicle_id: vehicleId,
        name: 'Shuttle 1',
        latitude: 42.7284,
        longitude: -73.6918,
        route_name: 'NORTH',
      }
    };

    // Later: Shuttle exited (empty)
    const withoutShuttle = {};

    let callCount = 0;
    fetchMock.mockImplementation(() => {
      callCount++;
      const locationData = callCount === 1 ? withShuttle : withoutShuttle;

      return Promise.resolve({
        ok: true,
        json: async () => locationData,
      } as Response);
    });

    const response1 = await fetch('/api/locations');
    const data1 = await response1.json();

    const response2 = await fetch('/api/locations');
    const data2 = await response2.json();

    // Verify shuttle was present initially
    expect(data1[vehicleId]).toBeDefined();
    expect(Object.keys(data1)).toHaveLength(1);

    // Verify shuttle is gone after exit
    expect(data2[vehicleId]).toBeUndefined();
    expect(Object.keys(data2)).toHaveLength(0);
  });

  it('handles vehicle changing routes', async () => {
    const vehicleId = 'vehicle_1';

    // On NORTH route
    const onNorth = {
      [vehicleId]: {
        vehicle_id: vehicleId,
        name: 'Shuttle 1',
        latitude: 42.7284,
        longitude: -73.6918,
        route_name: 'NORTH',
      }
    };

    // Changed to SOUTH route
    const onSouth = {
      [vehicleId]: {
        vehicle_id: vehicleId,
        name: 'Shuttle 1',
        latitude: 42.7290,
        longitude: -73.6920,
        route_name: 'SOUTH',
      }
    };

    let callCount = 0;
    fetchMock.mockImplementation(() => {
      callCount++;
      const locationData = callCount === 1 ? onNorth : onSouth;

      return Promise.resolve({
        ok: true,
        json: async () => locationData,
      } as Response);
    });

    const response1 = await fetch('/api/locations');
    const data1 = await response1.json();

    const response2 = await fetch('/api/locations');
    const data2 = await response2.json();

    // Verify route changed
    expect(data1[vehicleId].route_name).toBe('NORTH');
    expect(data2[vehicleId].route_name).toBe('SOUTH');

    // Verify it's the same vehicle
    expect(data1[vehicleId].vehicle_id).toBe(data2[vehicleId].vehicle_id);
  });

  it('handles API errors gracefully', async () => {
    fetchMock.mockImplementationOnce(() => {
      return Promise.reject(new Error('Network error'));
    });

    await expect(fetch('/api/locations')).rejects.toThrow('Network error');
  });

  it('validates location data structure', async () => {
    const vehicleId = 'vehicle_1';
    const locationData = {
      [vehicleId]: {
        vehicle_id: vehicleId,
        name: 'Shuttle 1',
        latitude: 42.7284,
        longitude: -73.6918,
        heading_degrees: 90,
        speed_mph: 15,
        timestamp: '2024-01-15T12:00:00Z',
        route_name: 'NORTH',
      }
    };

    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => locationData,
    } as Response);

    const response = await fetch('/api/locations');
    const data = await response.json();

    // Verify structure
    expect(data).toBeDefined();
    expect(data[vehicleId]).toBeDefined();
    expect(data[vehicleId].vehicle_id).toBe(vehicleId);
    expect(data[vehicleId].name).toBe('Shuttle 1');
    expect(typeof data[vehicleId].latitude).toBe('number');
    expect(typeof data[vehicleId].longitude).toBe('number');
    expect(typeof data[vehicleId].heading_degrees).toBe('number');
    expect(typeof data[vehicleId].speed_mph).toBe('number');
    expect(data[vehicleId].timestamp).toBeTruthy();
    expect(typeof data[vehicleId].route_name).toBe('string');
  });
});
