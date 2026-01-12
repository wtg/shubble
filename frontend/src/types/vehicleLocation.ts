type VehicleLocationData = {
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
    polyline_index: number | null;
    route_name: string;
    speed_mph: number;
    timestamp: string; // ISO 8601 format
    vehicle_name: string;
    vin: string;
    eta?: {
        etas: {
            next_stop: string; // ISO 8601 format datetime
        };
        timestamp: string; // ISO 8601 format
    } | null;
    predicted_location?: {
        speed_kmh: number;
        timestamp: string; // ISO 8601 format
    } | null;
}

export type VehicleInformationMap = Record<string, VehicleLocationData>;

export type VehicleAnnotationProps = {
  coordinate: { latitude: number; longitude: number };
  title: string;
  subtitle: string;
  svgUrl: string;
  size: { width: number; height: number };
  anchorOffset: { x: number; y: number };
};