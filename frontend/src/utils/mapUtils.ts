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
 * Moves a point along the polyline by a certain distance (in meters).
 * Supports both positive (forward) and negative (backward) distances.
 * 
 * @param polyline - The route as an array of coordinates
 * @param startIndex - Index of the segment containing startPoint
 * @param startPoint - Current position on the polyline
 * @param distanceMeters - Distance to move (positive = forward, negative = backward)
 * @returns New position on the polyline
 */
export function moveAlongPolyline(
    polyline: Coordinate[],
    startIndex: number,
    startPoint: Coordinate,
    distanceMeters: number
): { index: number, point: Coordinate } {
    // Handle backward movement
    if (distanceMeters < 0) {
        return moveBackward(polyline, startIndex, startPoint, -distanceMeters);
    }

    // Forward movement
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
 * Helper function to move backward along the polyline.
 */
function moveBackward(
    polyline: Coordinate[],
    startIndex: number,
    startPoint: Coordinate,
    distanceMeters: number
): { index: number, point: Coordinate } {
    let currentIndex = startIndex;
    let currentPoint = startPoint;
    let remainingDist = distanceMeters;

    while (remainingDist > 0 && currentIndex >= 0) {
        const prevPoint = polyline[currentIndex];
        const segmentDist = haversineDistance(currentPoint, prevPoint);

        if (remainingDist <= segmentDist) {
            // The target is on this segment (moving toward start)
            const ratio = remainingDist / segmentDist;
            const newLat = currentPoint.latitude + (prevPoint.latitude - currentPoint.latitude) * ratio;
            const newLon = currentPoint.longitude + (prevPoint.longitude - currentPoint.longitude) * ratio;
            return { index: currentIndex, point: { latitude: newLat, longitude: newLon } };
        } else {
            // Move to the previous segment
            remainingDist -= segmentDist;
            currentPoint = prevPoint;
            currentIndex--;
        }
    }

    // Reached the start of the polyline
    return { index: 0, point: polyline[0] };
}