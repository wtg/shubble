"""update schedule tables

Revision ID: a1b2c3d4e5f6
Revises: f7bf7132d19b
Create Date: 2026-03-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '80692b8f8d92'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old route_to_bus_schedules (timestamp -> time column, add unique constraint)
    op.drop_table('route_to_bus_schedules')
    op.create_table('route_to_bus_schedules',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('route_id', sa.Integer(), nullable=False),
        sa.Column('bus_schedule_id', sa.Integer(), nullable=False),
        sa.Column('time', sa.TIME(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['bus_schedule_id'], ['bus_schedules.id']),
        sa.ForeignKeyConstraint(['route_id'], ['routes.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('bus_schedule_id', 'time', name='uq_route_to_bus_schedules_bus_schedule_id_timestamp'),
    )

    # Add date column to date_to_day_schedules
    op.add_column('date_to_day_schedules', sa.Column('date', sa.Date(), nullable=False, server_default='2026-01-01'))
    op.alter_column('date_to_day_schedules', 'date', server_default=None)

    # Add route_id to stops
    op.add_column('stops', sa.Column('route_id', sa.Integer(), nullable=True))
    op.create_foreign_key('stops_route_id_fkey', 'stops', 'routes', ['route_id'], ['id'])
    op.alter_column('stops', 'route_id', nullable=False)

    # Add unique constraint to polylines
    op.create_unique_constraint(
        'uq_polylines_arrival_departure_stops',
        'polylines',
        ['arrival_stop_id', 'departure_stop_id']
    )


def downgrade() -> None:
    op.drop_constraint('uq_polylines_arrival_departure_stops', 'polylines', type_='unique')
    op.drop_constraint('stops_route_id_fkey', 'stops', type_='foreignkey')
    op.drop_column('stops', 'route_id')
    op.drop_column('date_to_day_schedules', 'date')

    op.drop_table('route_to_bus_schedules')
    op.create_table('route_to_bus_schedules',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('route_id', sa.Integer(), nullable=False),
        sa.Column('bus_schedule_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['bus_schedule_id'], ['bus_schedules.id']),
        sa.ForeignKeyConstraint(['route_id'], ['routes.id']),
        sa.PrimaryKeyConstraint('id'),
    )
