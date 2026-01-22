import type { VehicleLocationMap, VehicleVelocityMap, VehicleCombinedMap } from '../types/vehicleLocation';

/**
 * Prefetch module for vehicle data.
 * 
 * This module starts fetching /api/locations and /api/velocities immediately
 * after config is loaded, before React mounts. Components can then await
 * the prefetched promise instead of starting a fresh request.
 */

let vehicleDataPromise: Promise<VehicleCombinedMap> | null = null;

/**
 * Start prefetching vehicle location and velocity data.
 * Call this immediately after config is loaded in main.tsx.
 */
export function prefetchVehicleData(apiBaseUrl: string): void {
    vehicleDataPromise = (async () => {
        const [locationsResponse, velocitiesResponse] = await Promise.all([
            fetch(`${apiBaseUrl}/api/locations`, { cache: 'no-store' }),
            fetch(`${apiBaseUrl}/api/velocities`, { cache: 'no-store' })
        ]);

        if (!locationsResponse.ok) {
            throw new Error('Failed to fetch locations');
        }

        const locationsData: VehicleLocationMap = await locationsResponse.json();

        // Velocities are optional - don't fail if they're unavailable
        let velocitiesData: VehicleVelocityMap = {};
        if (velocitiesResponse.ok) {
            velocitiesData = await velocitiesResponse.json();
        }

        // Merge location and velocity data
        const combined: VehicleCombinedMap = {};
        for (const [vehicleId, location] of Object.entries(locationsData)) {
            const velocity = velocitiesData[vehicleId];
            combined[vehicleId] = {
                ...location,
                route_name: velocity?.route_name ?? null,
                polyline_index: velocity?.polyline_index ?? null,
                predicted_location: velocity && velocity.speed_kmh !== null && velocity.timestamp !== null ? {
                    speed_kmh: velocity.speed_kmh,
                    timestamp: velocity.timestamp
                } : undefined,
                is_at_stop: velocity?.is_at_stop,
                current_stop: velocity?.current_stop,
            };
        }

        return combined;
    })();
}

/**
 * Get the prefetched vehicle data promise.
 * Returns null if prefetch wasn't started or has been cleared.
 */
export function getPrefetchedVehicleData(): Promise<VehicleCombinedMap> | null {
    return vehicleDataPromise;
}

/**
 * Clear the prefetched data after it's been consumed.
 * This prevents stale data from being reused on subsequent mounts.
 */
export function clearPrefetchedData(): void {
    vehicleDataPromise = null;
}
