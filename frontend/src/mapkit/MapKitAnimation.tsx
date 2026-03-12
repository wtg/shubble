import { useEffect, useRef } from "react";
import type { KeyedAnnotation } from "./MapKitOverlays";
import {
  type Coordinate,
  moveAlongPolyline,
} from "../utils/mapUtils";

export interface AnimatedAnnotation extends KeyedAnnotation {
  heading: number;
  speedMph: number;
  timestamp: number;
  segmentIndex: number;
  routePolyline?: Coordinate[];
  predictedSpeedKmh?: number;
}

type AnimationState = {
  lastUpdateTime: number; // local time when we received the server update
  polylineIndex: number;
  currentPoint: Coordinate;
  targetDistance: number; // total distance to travel in this prediction window (meters)
  distanceTraveled: number; // distance already traveled in this window (meters)
  lastServerTime: number;
};

type MapKitAnimationProps = {
  annotations: AnimatedAnnotation[];
  vehicleAnnotations: Record<string, mapkit.Annotation>;
  showTrueLocation: boolean;
};

export default function MapKitAnimation({
  annotations,
  vehicleAnnotations,
  showTrueLocation
}: MapKitAnimationProps) {
  const vehicleAnimationStates = useRef<Record<string, AnimationState>>({});
  const animationFrameId = useRef<number | null>(null);

  // --- Update Animation State for new/updated vehicles ---
  useEffect(() => {
    if (!annotations || showTrueLocation) return;

    const now = Date.now();
    annotations.forEach((annotation) => {
      // If we don't have a route for this vehicle, we can't animate along a path nicely.
      // We'll just rely on the API updates or maybe simple linear extrapolation later?
      // For now, let's only set up animation if we have a valid route.
      if (!annotation.routePolyline) return;

      // Use the coordinate from the annotation props (source of truth from server)
      const vehicleCoord: Coordinate = {
        latitude: annotation.coordinate.latitude,
        longitude: annotation.coordinate.longitude,
      };

      const serverTime = annotation.timestamp; // already number? or Date? Assuming number based on usage.

      // Check if we already have state
      const animState = vehicleAnimationStates.current[annotation.id];

      // If the server data hasn't changed (cached response), ignore this update
      // and let the client-side prediction continue running.
      if (animState && animState.lastServerTime === serverTime) {
        return;
      }

      // =======================================================================
      // PREDICTION SMOOTHING ALGORITHM
      // =======================================================================
      const PREDICTION_WINDOW_SECONDS = 5;

      // Step 1: Calculate where the shuttle will be in 5 seconds
      // Use predicted speed if available, otherwise fall back to reported speed
      // If showTrueLocation is true, use 0 speed to disable animation
      const speedMph = annotation.predictedSpeedKmh 
        ? annotation.predictedSpeedKmh * 0.621371 
        : annotation.speedMph;
      // Convert speed from mph to meters/second (1 mph = 0.44704 m/s)
      const speedMetersPerSecond = speedMph * 0.44704;
      const projectedDistanceMeters = speedMetersPerSecond * PREDICTION_WINDOW_SECONDS;

      vehicleAnimationStates.current[annotation.id] = {
        lastUpdateTime: now,
        polylineIndex: annotation.segmentIndex,
        currentPoint: vehicleCoord,
        targetDistance: projectedDistanceMeters,
        distanceTraveled: 0,
        lastServerTime: serverTime,
      };
    });
  }, [annotations, showTrueLocation]);

  // --- Animation Loop ---
  useEffect(() => {
    if (showTrueLocation) return; // No animation needed

    let lastFrameTime = Date.now();

    const animate = () => {
      const now = Date.now();
      const dt = now - lastFrameTime; // ms
      lastFrameTime = now;

      // Avoid huge jumps if tab was backgrounded
      if (dt > 1000) {
        animationFrameId.current = requestAnimationFrame(animate);
        return;
      }

      // We need to iterate over *current* animation states
      Object.keys(vehicleAnimationStates.current).forEach(key => {
        const animState = vehicleAnimationStates.current[key];
        const annotation = vehicleAnnotations[key] as mapkit.ShuttleAnnotation;
        // Find the data object corresponding to this key to get the route
        const dataAnnotation = annotations.find(a => a.id === key);

        if (!dataAnnotation || !annotation || !animState) return;
        if (!dataAnnotation.routePolyline) return;

        const routePolyline = dataAnnotation.routePolyline;

        // =======================================================================
        // EASED ANIMATION
        // =======================================================================
        const PREDICTION_WINDOW_MS = 5000;
        const timeElapsed = now - animState.lastUpdateTime;

        // Calculate progress through the prediction window (0.0 to 1.0)
        const progress = Math.min(timeElapsed / PREDICTION_WINDOW_MS, 1.0);

        // Calculate how far along the target distance we should be (linear interpolation)
        const targetPosition = animState.targetDistance * progress;

        // Calculate how much to move this frame (can be negative for backward movement)
        const distanceToMove = targetPosition - animState.distanceTraveled;

        // Skip if no movement needed
        if (distanceToMove === 0) return;

        // Move along polyline
        const { index, point } = moveAlongPolyline(
          routePolyline,
          animState.polylineIndex,
          animState.currentPoint,
          distanceToMove
        );

        // Update state
        animState.polylineIndex = index;
        animState.currentPoint = point;
        animState.distanceTraveled = targetPosition;

        // Update MapKit annotation
        annotation.coordinate = new mapkit.Coordinate(point.latitude, point.longitude);
      });

      animationFrameId.current = requestAnimationFrame(animate);
    };

    animationFrameId.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [annotations, vehicleAnnotations]); // re-bind animate if annotations list changes significantly?
  // Note: relying on `annotations` in the effect dependency might reset the loop often if annotations changes every 5s.
  // Ideally `animate` closes over a ref to `annotations` or `annotations` is a stable object.
  // But `annotations` is a new array every 5s.
  // The `animate` function is recreated every 5s. This is fine, `requestAnimationFrame` is cancelled and restarted.
  // It might cause a slight stutter every 5s if `cancelAnimationFrame` happens.
  // But since `dt` handles time diff, it should be smooth.

  return null;
}