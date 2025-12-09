export type Coordinate = {
    latitude: number;
    longitude: number;
};

// Earth's radius in meters
const R = 6371e3;

/**
 * Calculates the distance between two coordinates in meters using the Haversine formula.
 */
export function haversineDistance(coord1: Coordinate, coord2: Coordinate): number {
    const lat1 = coord1.latitude * Math.PI / 180;
    const lat2 = coord2.latitude * Math.PI / 180;
    const deltaLat = (coord2.latitude - coord1.latitude) * Math.PI / 180;
    const deltaLon = (coord2.longitude - coord1.longitude) * Math.PI / 180;

    const a = Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
        Math.cos(lat1) * Math.cos(lat2) *
        Math.sin(deltaLon / 2) * Math.sin(deltaLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
}

/**
 * Finds the nearest point on a polyline to a given coordinate.
 * Returns the index of the start of the closest segment and the projected point.
 */
export function findNearestPointOnPolyline(
    target: Coordinate,
    polyline: Coordinate[]
): { index: number, point: Coordinate, distance: number } {
    let minDistance = Infinity;
    let bestIndex = 0;
    let bestPoint = polyline[0];

    for (let i = 0; i < polyline.length - 1; i++) {
        const p1 = polyline[i];
        const p2 = polyline[i + 1];

        const projected = projectPointOnSegment(target, p1, p2);
        const dist = haversineDistance(target, projected);

        if (dist < minDistance) {
            minDistance = dist;
            bestIndex = i;
            bestPoint = projected;
        }
    }

    return { index: bestIndex, point: bestPoint, distance: minDistance };
}

/**
 * Projects a point onto a line segment defined by p1 and p2.
 */
function projectPointOnSegment(p: Coordinate, p1: Coordinate, p2: Coordinate): Coordinate {
    const l2 = (p1.latitude - p2.latitude) ** 2 + (p1.longitude - p2.longitude) ** 2;
    if (l2 === 0) return p1;

    let t = ((p.latitude - p1.latitude) * (p2.latitude - p1.latitude) +
        (p.longitude - p1.longitude) * (p2.longitude - p1.longitude)) / l2;

    t = Math.max(0, Math.min(1, t));

    return {
        latitude: p1.latitude + t * (p2.latitude - p1.latitude),
        longitude: p1.longitude + t * (p2.longitude - p1.longitude)
    };
}

/**
 * Moves a point along the polyline by a certain distance (in meters).
 * Starts from a given segment index and fractional progress, or just finds the next point.
 * For simplicity, we assume we are at `startPoint` which is on the segment starting at `startIndex`.
 */
export function moveAlongPolyline(
    polyline: Coordinate[],
    startIndex: number,
    startPoint: Coordinate,
    distanceMeters: number
): { index: number, point: Coordinate } {
    let currentIndex = startIndex;
    let currentPoint = startPoint;
    let remainingDist = distanceMeters;

    while (remainingDist > 0 && currentIndex < polyline.length - 1) {
        const nextPoint = polyline[currentIndex + 1];
        const segmentDist = haversineDistance(currentPoint, nextPoint);

        if (remainingDist <= segmentDist) {
            // The target is on this segment
            const ratio = remainingDist / segmentDist;
            const newLat = currentPoint.latitude + (nextPoint.latitude - currentPoint.latitude) * ratio;
            const newLon = currentPoint.longitude + (nextPoint.longitude - currentPoint.longitude) * ratio;
            return { index: currentIndex, point: { latitude: newLat, longitude: newLon } };
        } else {
            // Move to the next segment
            remainingDist -= segmentDist;
            currentPoint = nextPoint;
            currentIndex++;
        }
    }

    // Reached the end of the polyline
    return { index: polyline.length - 1, point: polyline[polyline.length - 1] };
}
