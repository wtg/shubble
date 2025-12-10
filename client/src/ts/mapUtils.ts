export type Coordinate = {
    latitude: number;
    longitude: number;
};

// Earth's radius in meters
const R = 6371e3;

/**
 * Calculates the great-circle distance between two coordinates in meters using the Haversine formula.
 * 
 * MATH EXPLANATION:
 * The Haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes.
 * It is numerically stable for small distances (unlike using the law of cosines directly).
 * 
 * a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
 * c = 2 ⋅ atan2( √a, √(1−a) )
 * d = R ⋅ c
 * 
 * Where φ is latitude, λ is longitude, R is Earth's radius (approx 6371km).
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
 * Projects a point p onto the line segment defined by p1 and p2.
 * 
 * MATH EXPLANATION:
 * We treat the Earth's surface locally as a flat plane (Euclidean approximation).
 * However, because longitude lines converge at the poles, 1 degree of longitude is smaller than 1 degree of latitude.
 * We correct for this by scaling the longitude difference by cos(latitude).
 * 
 * The projection uses the "Vector Projection" formula.
 * Let vector A = p - p1 (vector from start of segment to point)
 * Let vector B = p2 - p1 (vector representing the segment)
 * 
 * We want to project A onto B to find how far along the segment the point lies.
 * The scalar projection t is given by the dot product:
 * 
 *      t = (A . B) / |B|^2
 * 
 *  - If t <= 0, the closest point is p1 (start of segment).
 *  - If t >= 1, the closest point is p2 (end of segment).
 *  - If 0 < t < 1, the closest point is p1 + t * B.
 */
function projectPointOnSegment(p: Coordinate, p1: Coordinate, p2: Coordinate): Coordinate {
    // Average latitude for the segment to calculate the longitude scaling factor
    const meanLat = (p1.latitude + p2.latitude) / 2 * (Math.PI / 180);
    const cosLat = Math.cos(meanLat);

    // Vector B (segment direction)
    // x corresponds to longitude (scaled), y corresponds to latitude
    const Bx = (p2.longitude - p1.longitude) * cosLat;
    const By = p2.latitude - p1.latitude;

    // Magnitude of B squared (|B|^2)
    const l2 = Bx * Bx + By * By;

    // If the segment is a single point (length 0), return that point
    if (l2 === 0) return p1;

    // Vector A (start to point)
    const Ax = (p.longitude - p1.longitude) * cosLat;
    const Ay = p.latitude - p1.latitude;

    // Dot product A . B
    // t represents the fractional distance along the segment (0.0 to 1.0)
    let t = (Ax * Bx + Ay * By) / l2;

    // Clamp t to the segment range [0, 1]
    t = Math.max(0, Math.min(1, t));

    // Calculate the new point
    // Note: We interpolate the raw Lat/Lon values directly, which is safe for short segments.
    // If we were dealing with massive segments (trans-oceanic), we'd need Great Circle interpolation.
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

/**
 * Calculates the distance along the polyline between a start point and an end point.
 * Assumes both points are "on" the polyline (projected).
 * startPoint is on segment starting at `startIndex`.
 * endPoint is on segment starting at `endIndex`.
 * If startIndex > endIndex, it assumes the route wrapped or something (returns 0 or positive dist).
 * Actually, for this use case, we just assume positive progress.
 */
export function calculateDistanceAlongPolyline(
    polyline: Coordinate[],
    startIndex: number,
    startPoint: Coordinate,
    endIndex: number,
    endPoint: Coordinate
): number {
    if (startIndex > endIndex) {
        // Could happen if we looped or re-routed. Return 0 to be safe.
        return 0;
    }

    if (startIndex === endIndex) {
        return haversineDistance(startPoint, endPoint);
    }

    let totalDistance = 0;

    // 1. Distance from startPoint to end of its segment
    totalDistance += haversineDistance(startPoint, polyline[startIndex + 1]);

    // 2. Full segments in between
    for (let i = startIndex + 1; i < endIndex; i++) {
        totalDistance += haversineDistance(polyline[i], polyline[i + 1]);
    }

    // 3. Distance from start of end segment to endPoint
    totalDistance += haversineDistance(polyline[endIndex], endPoint);

    return totalDistance;
}
