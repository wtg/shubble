import { useEffect, useLayoutEffect, useRef } from "react";
import type { MutableRefObject } from "react";

export interface KeyedAnnotation {
  id: string;
  coordinate: mapkit.Coordinate;
  title: string;
  subtitle: string;
  url?: { 1: string };
  size?: { width: number; height: number };
  anchorOffset?: DOMPoint;
  appearanceAnimation?: string;
}

export type ShuttleAnnotationRecord = Record<string, mapkit.ShuttleAnnotation>;
export type RenderedShuttleAnnotationStore = Map<string, mapkit.ShuttleAnnotation>;

export type MapKitOverlaysProps = {
  map: mapkit.Map | null;
  overlays: KeyedAnnotation[];
  onAnnotationsReady?: (annotations: ShuttleAnnotationRecord) => void;
  renderedAnnotationsByKey?: MutableRefObject<RenderedShuttleAnnotationStore>;
};

type SyncShuttleAnnotationsOptions = {
  map: mapkit.Map;
  overlays: KeyedAnnotation[];
  renderedAnnotationsByKey: RenderedShuttleAnnotationStore;
  onAnnotationsReady?: (annotations: ShuttleAnnotationRecord) => void;
};

function toAnnotationRecord(renderedAnnotationsByKey: RenderedShuttleAnnotationStore): ShuttleAnnotationRecord {
  const annotationsRecord: ShuttleAnnotationRecord = {};
  renderedAnnotationsByKey.forEach((annotation, key) => {
    annotationsRecord[key] = annotation;
  });
  return annotationsRecord;
}

function createShuttleAnnotation(overlay: KeyedAnnotation): mapkit.ShuttleAnnotation {
  const annotationOptions: mapkit.ImageAnnotationConstructorOptions = {
    title: overlay.title,
    subtitle: overlay.subtitle,
    url: overlay.url ?? { 1: '' },
    size: overlay.size,
    anchorOffset: overlay.anchorOffset,
    appearanceAnimation: overlay.appearanceAnimation ?? 'none',
  };

  return new window.mapkit.ImageAnnotation(overlay.coordinate, annotationOptions) as mapkit.ShuttleAnnotation;
}

export function syncShuttleAnnotations({
  map,
  overlays,
  renderedAnnotationsByKey,
  onAnnotationsReady,
}: SyncShuttleAnnotationsOptions) {
  const newOverlaysMap = new Map<string, KeyedAnnotation>();
  overlays.forEach((overlay) => {
    newOverlaysMap.set(overlay.id, overlay);
  });

  const currentOverlayKeys = new Set(renderedAnnotationsByKey.keys());
  const newOverlayKeys = new Set(newOverlaysMap.keys());

  const keysToRemove = Array.from(currentOverlayKeys).filter((key) => !newOverlayKeys.has(key));
  const overlaysToRemove: mapkit.ShuttleAnnotation[] = [];

  keysToRemove.forEach((key) => {
    const overlay = renderedAnnotationsByKey.get(key);
    if (overlay) {
      overlaysToRemove.push(overlay);
      renderedAnnotationsByKey.delete(key);
    }
  });

  if (overlaysToRemove.length > 0) {
    map.removeAnnotations(overlaysToRemove);
  }

  const keysToAdd = Array.from(newOverlayKeys).filter((key) => !currentOverlayKeys.has(key));
  const overlaysToAdd: mapkit.ShuttleAnnotation[] = [];

  keysToAdd.forEach((key) => {
    const overlay = newOverlaysMap.get(key);
    if (!overlay) {
      return;
    }

    const annotation = createShuttleAnnotation(overlay);
    overlaysToAdd.push(annotation);
    renderedAnnotationsByKey.set(key, annotation);
  });

  if (overlaysToAdd.length > 0) {
    map.addAnnotations(overlaysToAdd);
  }

  newOverlayKeys.forEach((key) => {
    const annotation = renderedAnnotationsByKey.get(key);
    const overlay = newOverlaysMap.get(key);
    if (!annotation || !overlay) {
      return;
    }

    annotation.coordinate = overlay.coordinate;
    annotation.title = overlay.title;
    annotation.subtitle = overlay.subtitle;

    if (overlay.url) {
      annotation.url = overlay.url;
    }
    if (overlay.size) {
      annotation.size = overlay.size;
    }
    if (overlay.anchorOffset) {
      annotation.anchorOffset = overlay.anchorOffset;
    }
    if (overlay.appearanceAnimation) {
      annotation.appearanceAnimation = overlay.appearanceAnimation;
    }
  });

  onAnnotationsReady?.(toAnnotationRecord(renderedAnnotationsByKey));
}

/**
 * MapKitOverlays manages adding and removing annotations/overlays on a MapKit map.
 *
 * This component accepts a list of KeyedAnnotation objects (stated props) and efficiently
 * manages which overlays are rendered on the map by comparing keys.
 */
export default function MapKitOverlays({
  map,
  overlays,
  onAnnotationsReady,
  renderedAnnotationsByKey,
}: MapKitOverlaysProps) {
  const localRenderedAnnotationsByKey = useRef<RenderedShuttleAnnotationStore>(new Map());
  const annotationStoreRef = renderedAnnotationsByKey ?? localRenderedAnnotationsByKey;

  useLayoutEffect(() => {
    if (!map) return;

    syncShuttleAnnotations({
      map,
      overlays,
      renderedAnnotationsByKey: annotationStoreRef.current,
      onAnnotationsReady,
    });
  }, [annotationStoreRef, map, onAnnotationsReady, overlays]);

  useEffect(() => {
    const currentOverlays = annotationStoreRef.current;

    return () => {
      if (map && currentOverlays.size > 0) {
        map.removeAnnotations(Array.from(currentOverlays.values()));
        currentOverlays.clear();
        onAnnotationsReady?.({});
      }
    };
  }, [annotationStoreRef, map, onAnnotationsReady]);

  return null;
}
