"""SQLAlchemy models for async database operations."""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, Index, UniqueConstraint, TIME, ARRAY, JSON, DATE
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    asset_type: Mapped[str] = mapped_column(String, default="vehicle")
    license_plate: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    vin: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    maintenance_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    gateway_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    gateway_serial: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships
    geofence_events: Mapped[list["GeofenceEvent"]] = relationship(back_populates="vehicle", lazy="selectin")
    locations: Mapped[list["VehicleLocation"]] = relationship(back_populates="vehicle", lazy="selectin")
    driver_assignments: Mapped[list["DriverVehicleAssignment"]] = relationship(back_populates="vehicle", lazy="selectin")
    etas: Mapped[list["ETA"]] = relationship(back_populates="vehicle", lazy="selectin")
    predicted_locations: Mapped[list["PredictedLocation"]] = relationship(back_populates="vehicle", lazy="selectin")

    def __repr__(self):
        return f"<Vehicle {self.id} - {self.name}>"


class GeofenceEvent(Base):
    __tablename__ = "geofence_events"
    __table_args__ = (
        Index("ix_geofence_events_vehicle_time", "vehicle_id", "event_time"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)  # eventId from webhook
    vehicle_id: Mapped[str] = mapped_column(String, ForeignKey("vehicles.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    address_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    address_formatted: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    vehicle: Mapped["Vehicle"] = relationship(back_populates="geofence_events")

    def __repr__(self):
        return f"<GeofenceEvent {self.id} {self.event_type} for vehicle {self.vehicle_id}>"


class VehicleLocation(Base):
    __tablename__ = "vehicle_locations"
    __table_args__ = (
        Index("ix_vehicle_locations_vehicle_timestamp", "vehicle_id", "timestamp"),
        UniqueConstraint("vehicle_id", "timestamp", name="uq_vehicle_locations_vehicle_timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vehicle_id: Mapped[str] = mapped_column(String, ForeignKey("vehicles.id"), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    heading_degrees: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_ecu_speed: Mapped[bool] = mapped_column(Boolean, default=False)
    formatted_location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    address_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    address_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    vehicle: Mapped["Vehicle"] = relationship(back_populates="locations")

    def __repr__(self):
        return f"<VehicleLocation {self.vehicle_id} @ {self.timestamp}>"


class Driver(Base):
    __tablename__ = "drivers"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # Samsara driver ID
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    assignments: Mapped[list["DriverVehicleAssignment"]] = relationship(back_populates="driver", lazy="selectin")

    def __repr__(self):
        return f"<Driver {self.id} - {self.name}>"


class DriverVehicleAssignment(Base):
    __tablename__ = "driver_vehicle_assignments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    driver_id: Mapped[str] = mapped_column(String, ForeignKey("drivers.id"), nullable=False, index=True)
    vehicle_id: Mapped[str] = mapped_column(String, ForeignKey("vehicles.id"), nullable=False, index=True)
    assignment_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    assignment_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)  # null = currently assigned
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    driver: Mapped["Driver"] = relationship(back_populates="assignments")
    vehicle: Mapped["Vehicle"] = relationship(back_populates="driver_assignments")

    def __repr__(self):
        return f"<DriverVehicleAssignment {self.driver_id} -> {self.vehicle_id}>"


class ETA(Base):
    __tablename__ = "etas"
    __table_args__ = (
        Index("ix_etas_vehicle_timestamp", "vehicle_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vehicle_id: Mapped[str] = mapped_column(String, ForeignKey("vehicles.id"), nullable=False)
    etas: Mapped[dict] = mapped_column(JSON, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    vehicle: Mapped["Vehicle"] = relationship(back_populates="etas")

    def __repr__(self):
        return f"<ETA {self.vehicle_id} @ {self.timestamp}>"


class PredictedLocation(Base):
    __tablename__ = "predicted_locations"
    __table_args__ = (
        Index("ix_predicted_locations_vehicle_timestamp", "vehicle_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vehicle_id: Mapped[str] = mapped_column(String, ForeignKey("vehicles.id"), nullable=False)
    speed_kmh: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    vehicle: Mapped["Vehicle"] = relationship(back_populates="predicted_locations")

    def __repr__(self):
        return f"<PredictedLocation {self.vehicle_id} @ {self.timestamp} - {self.speed_kmh} km/h>"

class RouteToBusSchedule(Base):
    __tablename__ =  "route_to_bus_schedules"
    __table_args__ = (
        Index("route_id", "time"),
        UniqueConstraint("bus_schedule_id", "time", name="uq_route_to_bus_schedules_bus_schedule_id_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True )
    route_id: Mapped[int] = mapped_column(Integer, ForeignKey("routes.id"), nullable=False)
    bus_schedule_id: Mapped[int] = mapped_column( Integer, ForeignKey("bus_schedules.id"), nullable=False)
    time: Mapped[datetime] = mapped_column(TIME(timezone=True), nullable=False) 
    
    #Relationships
    route: Mapped["Route"] = relationship(back_populates="route_to_bus_schedules")
    bus_schedule: Mapped["BusSchedule"] = relationship(back_populates="route_to_bus_schedules")

    def __repr__(self):
        return f"<Route {self.id} - {self.timestamp}>"

class DateToDaySchedule(Base):
    __tablename__ = "date_to_day_schedules"
    __table_args__ = (
        Index("ix_date_to_day_schedules_day_schedule_id", "day_schedule_id"),
    )


    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True )
    day_schedule_id: Mapped[int] = mapped_column(Integer, ForeignKey("day_schedules.id"), nullable=False)
    date: Mapped[datetime] = mapped_column(DATE, nullable=False)

    #Relationships
    day_schedule: Mapped["DaySchedule"] = relationship(back_populates="date_to_day_schedule")

class DaySchedule(Base):
    __tablename__ = "day_schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True )
    name: Mapped[str] = mapped_column(String, nullable=False)

    #Relationships
    bus_schedule_to_day_schedule: Mapped[list["BusScheduleToDaySchedule"]] = relationship(back_populates="day_schedule")
    date_to_day_schedule: Mapped[list["DateToDaySchedule"]] = relationship(back_populates="day_schedule")


    def __repr__(self):
        return f"<DaySchedule {self.id} - {self.name}>"
    
class BusScheduleToDaySchedule(Base):
    __tablename__ = "bus_schedules_to_day_schedules"
    __table_args__ = (
        Index("bus_schedule_id", "day_schedule_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True )
    bus_schedule_id: Mapped[int] = mapped_column(Integer, ForeignKey("bus_schedules.id"), nullable=False)
    day_schedule_id: Mapped[int] = mapped_column( Integer, ForeignKey("day_schedules.id"), nullable=False)

    #Relationships
    bus_schedule: Mapped["BusSchedule"] = relationship(back_populates="bus_schedule_to_day_schedule")
    day_schedule: Mapped["DaySchedule"] = relationship(back_populates="bus_schedule_to_day_schedule")

    def __repr__(self):
        return f"<BusScheduleToDaySchedule {self.id} - {self.bus_schedule_id} - {self.day_schedule_id}>"

class BusSchedule(Base):
    __tablename__ = "bus_schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True )
    name: Mapped[str] = mapped_column(String, nullable=False)
    
    #Relationships
    route_to_bus_schedules: Mapped[list["RouteToBusSchedule"]] = relationship(back_populates="bus_schedule")
    bus_schedule_to_day_schedule: Mapped["BusScheduleToDaySchedule"] = relationship(back_populates = "bus_schedule")

    def __repr__(self):
        return f"<BusSchedule {self.id} - {self.name}>"
    
class Route(Base):
    __tablename__ = "routes"

    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True )
    name: Mapped[str] = mapped_column(String, nullable = False )
    route_color: Mapped[str] = mapped_column(String, nullable=False)
    
    #Relationships
    stops: Mapped[list["Stop"]] = relationship(back_populates="route", lazy="selectin")
    route_to_bus_schedules: Mapped[list["RouteToBusSchedule"]] = relationship(back_populates="route")

    def __repr__(self):
        return f"<Route {self.id} - {self.name}>"
    
class Stop(Base):
    __tablename__ = "stops"
    __table_args__ = (
        Index("ix_stops_route_id", "route_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True )
    name: Mapped[str] = mapped_column(String, nullable = False )
    route_id: Mapped[int] = mapped_column(Integer, ForeignKey("routes.id"), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
  
    # Relationships
    route: Mapped["Route"] = relationship(back_populates="stops", lazy="selectin")
    arrival_polyline: Mapped[list["Polyline"]] = relationship(back_populates="arrival_stop", foreign_keys="Polyline.arrival_stop_id", lazy="selectin")
    departure_polyline: Mapped[list["Polyline"]] = relationship(back_populates="departure_stop", foreign_keys="Polyline.departure_stop_id", lazy="selectin")

    def __repr__(self):
        return f"<Stop {self.id} - {self.stop_name}>"
    
class Polyline(Base):
    __tablename__ = "polylines"
    __table_args__ = (
        Index("arrival_stop_id", "departure_stop_id"),
        UniqueConstraint("arrival_stop_id", "departure_stop_id", name="uq_polylines_arrival_departure_stops"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True )
    departure_stop_id:  Mapped[int] = mapped_column(Integer, ForeignKey("stops.id"), nullable=False)
    arrival_stop_id:  Mapped[int] = mapped_column(Integer, ForeignKey("stops.id"), nullable=False)
    coordinates: Mapped[list[tuple]] = mapped_column(ARRAY(String), nullable=False)

    # Relationships
    arrival_stop: Mapped["Stop"] = relationship(back_populates="arrival_polyline", foreign_keys=[arrival_stop_id], lazy="selectin")
    departure_stop: Mapped["Stop"] = relationship(back_populates="departure_polyline", foreign_keys=[departure_stop_id], lazy="selectin")

    def __repr__(self):
        return f"<Polyline {self.id} - {self.route_id} From: {self.departure_stop} - {self.arrival_stop}>"
