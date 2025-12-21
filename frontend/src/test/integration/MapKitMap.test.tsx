import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, waitFor, act } from '@testing-library/react';
import MapKitMap from '../../components/MapKitMap';
import type { ShuttleRouteData } from '../../ts/types/route';

// Mock route data
const mockRouteData: ShuttleRouteData = {
  'NORTH': {
    COLOR: '#FF0000',
    STOPS: ['Union', 'Library'],
    POLYLINE_STOPS: [],
    ROUTES: [
      [
        [42.7284, -73.6918],
        [42.7290, -73.6920],
        [42.7295, -73.6922],
      ]
    ],
    Union: {
      COORDINATES: [42.7284, -73.6918],
      OFFSET: 0,
      NAME: 'Union',
    },
    Library: {
      COORDINATES: [42.7295, -73.6922],
      OFFSET: 1,
      NAME: 'Library',
    },
  }
};

// Mock vehicle locations for testing
const createMockLocation = (vehicleId: string, lat: number, lon: number, routeName?: string) => ({
  [vehicleId]: {
    vehicle_id: vehicleId,
    name: `Shuttle ${vehicleId}`,
    latitude: lat,
    longitude: lon,
    heading_degrees: 90,
    speed_mph: 15,
    timestamp: new Date().toISOString(),
    route_name: routeName || null,
  }
});

describe('MapKitMap - Vehicle Movement Integration Test', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  const originalFetch = global.fetch;

  beforeEach(() => {
    // Mock fetch before setting up MapKit to prevent component from making real requests
    fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    } as Response);
    global.fetch = fetchMock;

    // Mock Apple MapKit JS with proper constructors
    global.mapkit = {
      init: vi.fn(),
      loadedLibraries: ['full'],
      Map: class Map {
        annotations = [];
        overlays = [];
        region = { center: { latitude: 42.730082, longitude: -73.6778 }, span: { latitudeDelta: 0.02, longitudeDelta: 0.025 } };
        element: any;
        addAnnotations = vi.fn();
        removeAnnotations = vi.fn();
        addAnnotation = vi.fn();
        removeAnnotation = vi.fn();
        addOverlays = vi.fn();
        removeOverlays = vi.fn();
        setRegionAnimated = vi.fn();
        setCenterAnimated = vi.fn();
        setCameraZoomRangeAnimated = vi.fn();
        setCameraBoundaryAnimated = vi.fn();
        setCameraDistanceAnimated = vi.fn();
        setMapTypeAnimated = vi.fn();
        addEventListener = vi.fn();
        constructor(element: any, options?: any) {
          this.element = element;
        }
      },
      Coordinate: class Coordinate {
        latitude: number;
        longitude: number;
        constructor(lat: number, lon: number) {
          this.latitude = lat;
          this.longitude = lon;
        }
      },
      CoordinateSpan: class CoordinateSpan {
        latitudeDelta: number;
        longitudeDelta: number;
        constructor(latDelta: number, lonDelta: number) {
          this.latitudeDelta = latDelta;
          this.longitudeDelta = lonDelta;
        }
      },
      CoordinateRegion: class CoordinateRegion {
        center: any;
        span: any;
        constructor(center: any, span: any) {
          this.center = center;
          this.span = span;
        }
      },
      MarkerAnnotation: class MarkerAnnotation {
        coordinate: any;
        data: any;
        constructor(coords: any, options?: any) {
          this.coordinate = coords;
          this.data = options?.data || {};
        }
      },
      ImageAnnotation: class ImageAnnotation {
        coordinate: any;
        data: any;
        url: any;
        constructor(coords: any, options?: any) {
          this.coordinate = coords;
          this.data = options?.data || {};
          this.url = options?.url;
        }
      },
      PolylineOverlay: class PolylineOverlay {
        points: any[];
        style: any;
        constructor(points: any[], style?: any) {
          this.points = points;
          this.style = style;
        }
      },
      CircleOverlay: class CircleOverlay {
        coordinate: any;
        radius: number;
        style: any;
        stopName: any;
        stopKey: any;
        routeKey: any;
        constructor(coordinate: any, options?: any) {
          this.coordinate = coordinate;
          this.radius = options?.radius || 10;
          this.style = options?.style;
          this.stopName = options?.stopName;
          this.stopKey = options?.stopKey;
          this.routeKey = options?.routeKey;
        }
      },
      CameraZoomRange: class CameraZoomRange {
        minCameraDistance: number;
        maxCameraDistance: number;
        constructor(min: number, max: number) {
          this.minCameraDistance = min;
          this.maxCameraDistance = max;
        }
      },
      Style: class Style {
        constructor(options?: any) {
          Object.assign(this, options);
        }
        static FeatureVisibility = {
          Hidden: 'hidden',
        };
      },
      Directions: class Directions {
        route = vi.fn();
        constructor() {}
      },
      MapType: {
        Standard: 'standard',
        Satellite: 'satellite',
        Hybrid: 'hybrid',
      },
      FeatureType: {},
      addEventListener: vi.fn(),
    } as any;
  });

  afterEach(() => {
    vi.clearAllMocks();
    global.fetch = originalFetch;
  });

  it('renders without crashing with route data', () => {
    const { container } = render(
      <MapKitMap
        routeData={mockRouteData}
        selectedRoute={null}
        setSelectedRoute={() => {}}
      />
    );

    // Verify the map container is rendered
    expect(container.querySelector('.map')).toBeTruthy();
  });

  it('initializes MapKit with correct configuration', async () => {
    render(
      <MapKitMap
        routeData={mockRouteData}
        selectedRoute={null}
        setSelectedRoute={() => {}}
      />
    );

    // Wait for MapKit to be initialized
    await waitFor(() => {
      expect(global.mapkit.init).toHaveBeenCalled();
    });

    // Verify mapkit object exists and has expected properties
    expect(global.mapkit).toBeDefined();
    expect(global.mapkit.Map).toBeDefined();
    expect(global.mapkit.Coordinate).toBeDefined();
  });

  it('makes initial location fetch on mount', async () => {
    const initialLocation = createMockLocation('vehicle_1', 42.7284, -73.6918, 'NORTH');

    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => initialLocation,
    } as Response);

    render(
      <MapKitMap
        routeData={mockRouteData}
        selectedRoute={null}
        setSelectedRoute={() => {}}
      />
    );

    // Verify fetch was called
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });
  });
});
