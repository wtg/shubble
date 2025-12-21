from datetime import datetime
from . import db

class Vehicle(db.Model):
    __tablename__ = 'vehicles'

    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String, nullable=False)
    asset_type = db.Column(db.String, default='vehicle')
    license_plate = db.Column(db.String, nullable=True)
    vin = db.Column(db.String, nullable=True)

    maintenance_id = db.Column(db.String, nullable=True)
    gateway_model = db.Column(db.String, nullable=True)
    gateway_serial = db.Column(db.String, nullable=True)

    def __repr__(self):
        return f"<Vehicle {self.id} - {self.name}>"

class GeofenceEvent(db.Model):
    __tablename__ = 'geofence_events'

    id = db.Column(db.String, primary_key=True)  # eventId from webhook
    vehicle_id = db.Column(db.String, db.ForeignKey('vehicles.id'), nullable=False)
    event_type = db.Column(db.String, nullable=False)
    event_time = db.Column(db.DateTime, nullable=False)

    address_name = db.Column(db.String)
    address_formatted = db.Column(db.String)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    vehicle = db.relationship("Vehicle", backref=db.backref("geofence_events", lazy=True))

    def __repr__(self):
        return f"<GeofenceEvent {self.id} {self.event_type} for vehicle {self.vehicle_id}>"

class VehicleLocation(db.Model):
    __tablename__ = 'vehicle_locations'

    id = db.Column(db.Integer, primary_key=True)

    # Foreign key to vehicles.id
    vehicle_id = db.Column(db.String, db.ForeignKey('vehicles.id'), nullable=False, index=True)
    vehicle = db.relationship('Vehicle', backref='locations', lazy=True)

    name = db.Column(db.String, nullable=True)

    timestamp = db.Column(db.DateTime, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)

    heading_degrees = db.Column(db.Float, nullable=True)
    speed_mph = db.Column(db.Float, nullable=True)
    is_ecu_speed = db.Column(db.Boolean, default=False)

    formatted_location = db.Column(db.String, nullable=True)

    address_id = db.Column(db.String, nullable=True)
    address_name = db.Column(db.String, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ShuttleLocation {self.vehicle_id} @ {self.timestamp}>"


class Driver(db.Model):
    __tablename__ = 'drivers'

    id = db.Column(db.String, primary_key=True)  # Samsara driver ID
    name = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Driver {self.id} - {self.name}>"


class DriverVehicleAssignment(db.Model):
    __tablename__ = 'driver_vehicle_assignments'

    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.String, db.ForeignKey('drivers.id'), nullable=False, index=True)
    vehicle_id = db.Column(db.String, db.ForeignKey('vehicles.id'), nullable=False, index=True)
    assignment_start = db.Column(db.DateTime, nullable=False)
    assignment_end = db.Column(db.DateTime, nullable=True)  # null = currently assigned
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    driver = db.relationship('Driver', backref='assignments', lazy=True)
    vehicle = db.relationship('Vehicle', backref='driver_assignments', lazy=True)

    def __repr__(self):
        return f"<DriverVehicleAssignment {self.driver_id} -> {self.vehicle_id}>"
