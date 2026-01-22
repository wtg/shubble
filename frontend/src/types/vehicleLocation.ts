// Raw vehicle location data from /api/locations
export type VehicleLocationData = {
    address_id: string;
    address_name: string;
    asset_type: string;
    formatted_location: string;
    gateway_model: string;
    gateway_serial: string;
    heading_degrees: number;
    is_ecu_speed: boolean;
    latitude: number;
    license_plate: string;
    longitude: number;
    name: string;
    speed_mph: number;
    timestamp: string; // ISO 8601 format
    vin: string;
    driver?: {
        id: string;
        name: string;
    } | null;
}

// ETA and route data from /api/etas
export type VehicleETAData = {
    stop_times: Record<string, string>; // stop_key -> ISO 8601 format datetime
    timestamp: string; // ISO 8601 format
}

// Velocity and route data from /api/velocities
export type VehicleVelocityData = {
    speed_kmh: number | null;
    timestamp: string | null;
    route_name: string | null;
    polyline_index: number | null;
    is_at_stop?: boolean;
    current_stop?: string | null;
}

// Prediction data (deprecated, use VehicleETAData or VehicleVelocityData)
export type VehiclePredictionData = {
    route_name: string | null;
    polyline_index: number | null;
    stop_times?: {
        stop_times: Record<string, string>;
        timestamp: string;
    } | null;
    predicted_location?: {
        speed_kmh: number;
        timestamp: string;
    } | null;
    is_at_stop?: boolean;
    current_stop?: string | null;
}

// Combined vehicle data (location + etas + velocities merged)
export type VehicleCombinedData = VehicleLocationData & {
    route_name: string | null;
    polyline_index: number | null;
    stop_times?: {
        stop_times: Record<string, string>;
        timestamp: string;
    } | null;
    predicted_location?: {
        speed_kmh: number;
        timestamp: string;
    } | null;
    is_at_stop?: boolean;
    current_stop?: string | null;
};

export type VehicleLocationMap = Record<string, VehicleLocationData>;
export type VehicleETAMap = Record<string, VehicleETAData>;
export type VehicleVelocityMap = Record<string, VehicleVelocityData>;
export type VehicleCombinedMap = Record<string, VehicleCombinedData>;

// ETAs: vehicle_id -> stop_name -> ISO 8601 datetime
export type VehicleETAs = Record<string, Record<string, string>>;

export type VehicleAnnotationProps = {
  coordinate: { latitude: number; longitude: number };
  title: string;
  subtitle: string;
  svgUrl: string;
  size: { width: number; height: number };
  anchorOffset: { x: number; y: number };
};
