"""SQLAlchemy models for async database operations."""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


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
