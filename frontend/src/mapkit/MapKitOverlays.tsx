import { useEffect, useRef } from "react";

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

type MapKitOverlaysProps = {
  map: mapkit.Map | null;
  overlays: KeyedAnnotation[];
  onAnnotationsReady?: (annotations: Record<string, mapkit.Annotation>) => void;
};

/**
 * MapKitOverlays manages adding and removing annotations/overlays on a MapKit map.
 *
 * This component accepts a list of KeyedAnnotation objects (stated props) and efficiently
 * manages which overlays are rendered on the map by comparing keys.
 */
export default function MapKitOverlays({ map, overlays, onAnnotationsReady }: MapKitOverlaysProps) {
  // Track currently rendered annotations by their keys
  const renderedOverlaysByKey = useRef<Map<string, mapkit.Annotation>>(new Map());

  useEffect(() => {
    if (!map) return;

    // Convert new overlays list to a map for easy lookup
    const newOverlaysMap = new Map<string, KeyedAnnotation>();
    overlays.forEach(overlay => {
      newOverlaysMap.set(overlay.id, overlay);
    });

    const currentOverlayKeys = new Set(renderedOverlaysByKey.current.keys());
    const newOverlayKeys = new Set(newOverlaysMap.keys());

    // Find overlays to remove (keys that exist in rendered but not in new)
    const keysToRemove = Array.from(currentOverlayKeys).filter(key => !newOverlayKeys.has(key));
    const overlaysToRemove: mapkit.Annotation[] = [];

    keysToRemove.forEach(key => {
      const overlay = renderedOverlaysByKey.current.get(key);
      if (overlay) {
        overlaysToRemove.push(overlay);
      }
    });

    if (overlaysToRemove.length > 0) {
      map.removeAnnotations(overlaysToRemove);
      keysToRemove.forEach(key => {
        renderedOverlaysByKey.current.delete(key);
      });
    }

    // Find overlays to add (keys that exist in new but not in rendered)
    const keysToAdd = Array.from(newOverlayKeys).filter(key => !currentOverlayKeys.has(key));
    const overlaysToAdd: mapkit.Annotation[] = [];

    keysToAdd.forEach(key => {
      const props = newOverlaysMap.get(key);
      if (props) {
        // Create new annotation from props
        const annotationOptions: mapkit.ImageAnnotationConstructorOptions = {
          title: props.title,
          subtitle: props.subtitle,
          url: props.url,
          size: props.size,
          anchorOffset: props.anchorOffset,
          appearanceAnimation: props.appearanceAnimation || 'none',
        };
        const annotation = new window.mapkit.ImageAnnotation(props.coordinate, annotationOptions) as mapkit.ShuttleAnnotation;
        overlaysToAdd.push(annotation);
        renderedOverlaysByKey.current.set(key, annotation);
      }
    });

    if (overlaysToAdd.length > 0) {
      map.addAnnotations(overlaysToAdd);
    }

    // Update existing annotations' properties
    const keysToUpdate = Array.from(newOverlayKeys).filter(key => currentOverlayKeys.has(key));
    keysToUpdate.forEach(key => {
      const annotation = renderedOverlaysByKey.current.get(key) as mapkit.ShuttleAnnotation;
      const props = newOverlaysMap.get(key);
      if (annotation && props) {
        // Update properties in place
        annotation.coordinate = props.coordinate;
        annotation.title = props.title;
        annotation.subtitle = props.subtitle;

        if (props.url) {
          annotation.url = props.url;
        }
        if (props.size) {
          annotation.size = props.size;
        }
        if (props.anchorOffset) {
          annotation.anchorOffset = props.anchorOffset;
        }
        if (props.appearanceAnimation) {
          (annotation as any).appearanceAnimation = props.appearanceAnimation;
        }
      }
    });
    // Notify parent of current annotations
    if (onAnnotationsReady) {
      const annotationsRecord: Record<string, mapkit.Annotation> = {};
      renderedOverlaysByKey.current.forEach((annotation, key) => {
        annotationsRecord[key] = annotation;
      });
      onAnnotationsReady(annotationsRecord);
    }

    // Cleanup on unmount
    return () => {
      // We don't necessarily want to remove annotations on every render,
      // but if the component unmounts fully, we might.
      // Current logic in original file did this, so we keep it.
      if (renderedOverlaysByKey.current.size > 0 && !map) {
        // Logic check: if map is gone, we can't remove.
      }
    };
  }, [map, overlays, onAnnotationsReady]);

  // Handle unmount cleanup separately to avoid stale map reference issues
  useEffect(() => {
    return () => {
      if (map && renderedOverlaysByKey.current.size > 0) {
        const allOverlays = Array.from(renderedOverlaysByKey.current.values());
        map.removeAnnotations(allOverlays);
        renderedOverlaysByKey.current.clear();
      }
    }
  }, [map]);

  return null;
}
