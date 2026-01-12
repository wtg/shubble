# MapKit Components

This directory contains the refactored MapKit implementation, split into three focused components for better maintainability and separation of concerns.

## Components

### MapKitCanvas.tsx
**Purpose**: Base map initialization and rendering

**Responsibilities**:
- Initializes Apple MapKit JS library
- Creates and configures the map instance (zoom, boundaries, center)
- Renders route polylines and stop circles as overlays
- Handles map event listeners (select, deselect, hover, region-change)
- Supports route generation via MapKit Directions API

**Props**:
- `routeData`: Route and stop data to display
- `generateRoutes`: Whether to generate routes using Directions API
- `selectedRoute`: Currently selected route
- `setSelectedRoute`: Callback to update selected route
- `isFullscreen`: Whether to render in fullscreen mode
- `onMapReady`: Callback when map is initialized

### MapKitAnimation.tsx
**Purpose**: Vehicle animation on polylines

**Responsibilities**:
- Manages animation state for each vehicle (position, speed, direction)
- Implements prediction smoothing algorithm to avoid rubberbanding
- Runs animation loop using requestAnimationFrame
- Projects vehicles onto route polylines
- Handles backward movement when over-prediction occurs
- Directly modifies annotation coordinates for smooth animation

**Props**:
- `vehicles`: Current vehicle location data
- `vehicleAnnotations`: Record (keyed object) of vehicle annotations to animate
- `flattenedRoutes`: Flattened route polylines for animation
- `showTrueLocation`: Whether to disable animation and show raw GPS

**Relationship with MapKitOverlays**:
This component receives annotation objects that are created and managed by MapKitOverlays.
The animation loop directly modifies the `coordinate` property of annotations. Since
MapKitOverlays tracks annotations by key and updates properties in place, these coordinate
modifications don't trigger remove/add operations - only the visual position changes.

**Algorithm**:
The animation uses a prediction smoothing algorithm that:
1. Calculates where the vehicle will be in 5 seconds based on current speed
2. Smoothly animates from current visual position to predicted position
3. Handles direction validation (stops if moving wrong way)
4. Snaps to server position if gap exceeds 250 meters

### MapKitOverlays.tsx
**Purpose**: Manages adding/removing annotations and overlays on the map

**Responsibilities**:
- Accepts a keyed object of annotation **props** (stated props)
- Creates MapKit annotation objects internally
- Efficiently adds new overlays to the map by comparing keys
- Updates existing annotation properties in place (coordinate, title, subtitle)
- Removes overlays whose keys are no longer present
- Tracks currently rendered overlays internally
- Exposes created annotations via callback for animation
- Cleans up on unmount

**Props**:
- `map`: MapKit map instance
- `overlays`: Record (keyed object) of VehicleAnnotationProps to render
- `onAnnotationsReady`: Optional callback that receives created annotation objects

**Stated props approach**:
This component uses stated props - annotation properties are passed as a keyed object where
keys represent vehicle IDs. The component:
1. Compares keys between renders to determine what to add/remove
2. Creates new annotation objects for new keys
3. Updates existing annotation properties in place for existing keys
4. Never recreates annotation objects unless the key is removed and re-added

This prevents unnecessary remove/add operations and allows MapKitAnimation to safely
modify annotation coordinates.

**Usage**:
```tsx
const overlayProps = useMemo(() => {
  const result: Record<string, VehicleAnnotationProps> = {};
  Object.keys(vehicles).forEach(key => {
    const vehicle = vehicles[key];
    result[key] = {
      coordinate: { latitude: vehicle.lat, longitude: vehicle.lon },
      title: vehicle.name,
      subtitle: `${vehicle.speed} mph`,
      svgUrl: svgDataUrl,
      size: { width: 25, height: 25 },
      anchorOffset: { x: 0, y: -13 }
    };
  });
  return result;
}, [vehicles]);

<MapKitOverlays
  map={map}
  overlays={overlayProps}
  onAnnotationsReady={setAnnotations}
/>
```

## Architecture: Stated Props Flow

The MapKit components use a **stated props** architecture where data flows explicitly through props:

```
LiveLocationMapKit
  ↓ computes
vehicleAnnotationProps: Record<string, VehicleAnnotationProps>
  ↓ passed to
MapKitOverlays
  ↓ creates/manages
vehicleAnnotations: Record<string, Annotation>
  ↓ exposed via callback
  ├─→ LiveLocationMapKit (via onAnnotationsReady)
  └─→ MapKitAnimation (modifies coordinates in place)
```

**Key benefits**:
1. **Explicit data flow**: Annotation props computed in LiveLocationMapKit
2. **Efficient updates**: Annotation objects created once and reused
3. **In-place property updates**: MapKitOverlays updates properties without recreating objects
4. **No unnecessary operations**: Only adds/removes when keys change
5. **Smooth animation**: MapKitAnimation modifies coordinates without triggering add/remove

**Example flow**:
1. Vehicle data updates → LiveLocationMapKit computes new annotation props
2. MapKitOverlays compares keys → adds new vehicles, updates existing, removes stale
3. MapKitOverlays calls onAnnotationsReady → LiveLocationMapKit gets annotation objects
4. MapKitAnimation updates coordinates → visual position changes smoothly
5. On next render → same annotation objects reused with updated props (no remove/add churn)

## Usage

### LiveLocationMapKit (Main Component)
Located in `frontend/src/locations/components/LiveLocationMapKit.tsx`

**Purpose**: Main orchestrator component that composes MapKitCanvas and MapKitAnimation

**Responsibilities**:
- Fetches vehicle location data from API every 5 seconds
- Creates and updates vehicle annotations (visual markers)
- Manages vehicle colors based on route assignment
- Handles vehicle speed display
- Composes MapKitCanvas and MapKitAnimation
- Flattens route polylines for animation use

**Props**:
- `routeData`: Route and stop data
- `displayVehicles`: Whether to show vehicle markers
- `generateRoutes`: Whether to generate routes dynamically
- `selectedRoute`: Currently selected route
- `setSelectedRoute`: Callback to update selected route
- `isFullscreen`: Fullscreen mode flag
- `showTrueLocation`: Whether to show raw GPS (no animation)

**Example**:
```tsx
import LiveLocationMapKit from './locations/components/LiveLocationMapKit';

function MyComponent() {
  return (
    <LiveLocationMapKit
      routeData={routeData}
      displayVehicles={true}
      showTrueLocation={false}
      isFullscreen={false}
    />
  );
}
```

## File Structure

```
mapkit/
├── README.md              # This file
├── index.ts               # Exports for all mapkit components
├── MapKitCanvas.tsx       # Map initialization and overlays
├── MapKitAnimation.tsx    # Animation logic
└── MapKitOverlays.tsx     # Overlay management (add/remove annotations)

locations/components/
└── LiveLocationMapKit.tsx # Main orchestrator (uses components from mapkit/)
```
