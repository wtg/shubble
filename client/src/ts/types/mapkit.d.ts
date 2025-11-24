declare namespace mapkit {

    // -------------------
    // Coordinates & Regions
    // -------------------
    class Coordinate {
      constructor(latitude: number, longitude: number);
      latitude: number;
      longitude: number;
    }

    class CoordinateSpan {
      constructor(latitudeDelta: number, longitudeDelta: number);
      latitudeDelta: number;
      longitudeDelta: number;
    }

    class CoordinateRegion {
      constructor(center: Coordinate, span: CoordinateSpan);
      center: Coordinate;
      span: CoordinateSpan;
    }

    // -------------------
    // Style
    // -------------------
    interface StyleOptions {
      strokeColor?: string;
      fillColor?: string;
      fillOpacity?: number;
      lineWidth?: number;
    }

    class Style {
      constructor(options?: StyleOptions);
    }

    // -------------------
    // Overlays
    // -------------------
    class Overlay {
      coordinate: Coordinate;
    }

    class CircleOverlay extends Overlay {
      constructor(coordinate: Coordinate, radius: number, options?: { style?: Style });
      radius: number;
      stopKey?: string;
      routeKey?: string;
      stopName?: string;
      style: Style;
    }

    class PolylineOverlay extends Overlay {
      constructor(coords: Coordinate[], options?: { style?: Style });
      coords: Coordinate[];
      style?: Style;
    }

    // -------------------
    // Marker Annotations
    // -------------------
    interface MarkerAnnotationOptions {
      title?: string;
      subtitle?: string;
      color?: string;
      glyphImage?: Record<number, string>;
      selectedGlyphImage?: Record<number, string>;
    }

    class MarkerAnnotation {
      constructor(coordinate: Coordinate, options?: MarkerAnnotationOptions);
      coordinate: Coordinate;
      title?: string;
      subtitle?: string;
      color?: string;
    }

    // -------------------
    // Image Annotations
    // -------------------
    interface ImageAnnotationOptions {
      title?: string;
      subtitle?: string;
      url?: Record<number, string>;
      size?: { width: number; height: number };
      anchorOffset?: DOMPoint;
    }

    class ImageAnnotation {
      constructor(coordinate: Coordinate, options?: ImageAnnotationOptions);
      coordinate: Coordinate;
      title?: string;
      subtitle?: string;
      url?: Record<number, string>;
      size?: { width: number; height: number };
      anchorOffset?: DOMPoint;
    }

    // -------------------
    // Shuttle-specific annotation extension
    // -------------------
    interface ShuttleAnnotation extends ImageAnnotation {
      lockedRoute?: string | null;
    }

    // -------------------
    // Directions
    // -------------------
    interface DirectionsOptions {
      origin: Coordinate;
      destination: Coordinate;
    }

    interface DirectionsRouteData {
      routes: {
        polyline: {
          points: Coordinate[];
        };
      }[];
    }

    class Directions {
      route(
        options: DirectionsOptions,
        callback: (error: Error | null, data: DirectionsRouteData) => void
      ): void;
    }

    // -------------------
    // Camera
    // -------------------
    class CameraZoomRange {
      constructor(min: number, max: number);
    }

    // -------------------
    // Map Events
    // -------------------
    interface MapSelectEvent {
      annotation?: MarkerAnnotation;
      overlay?: Overlay;
    }

    interface MapDeselectEvent {
      annotation?: MarkerAnnotation;
      overlay?: Overlay;
    }

    type MapEventHandler = (event: MapSelectEvent | MapDeselectEvent) => void;

    type MapEventName =
      | "select"
      | "deselect"
      | "region-change-start"
      | "region-change-end";

    // -------------------
    // Map
    // -------------------
    interface MapOptions {
      center?: Coordinate;
      region?: CoordinateRegion;
      isScrollEnabled?: boolean;
      isZoomEnabled?: boolean;
      showsZoomControl?: boolean;
      isRotationEnabled?: boolean;
      showsPointsOfInterest?: boolean;
      showsUserLocation?: boolean;
    }

    class Map {
      constructor(element: HTMLElement, options?: MapOptions);
      overlays: Overlay[];
      element: HTMLElement;

      addOverlays(overlays: Overlay[]): void;
      removeOverlays(overlays: Overlay[]): void;
      addAnnotation(annotation: MarkerAnnotation): void;
      removeAnnotation(annotation: MarkerAnnotation): void;

      addEventListener(type: MapEventName, listener: MapEventHandler): void;
      removeEventListener(type: MapEventName, listener: MapEventHandler): void;

      setCameraZoomRangeAnimated(range: CameraZoomRange, animated: boolean): void;
      setCameraBoundaryAnimated(region: CoordinateRegion, animated: boolean): void;
      setCameraDistanceAnimated(distance: number): void;

      region: CoordinateRegion;

      _hoverCleanup?: () => void;
    }

    // -------------------
    // MapKit initialization
    // -------------------
    interface MapKitInitOptions {
      authorizationCallback: (done: (token: string) => void) => void;
    }

    function init(options: MapKitInitOptions): void;
  }

  // -------------------
  // Extend window
  // -------------------
  declare interface Window {
    mapkit: typeof mapkit;
    initMapKit?: () => void;
  }
