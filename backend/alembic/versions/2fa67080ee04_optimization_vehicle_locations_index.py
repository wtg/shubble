"""optimization vehicle_locations index

Revision ID: 2fa67080ee04
Revises: 80692b8f8d92
Create Date: 2026-02-28 13:14:05.710512

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2fa67080ee04"
down_revision: Union[str, None] = "80692b8f8d92"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index(
        "ix_vehicle_locations_vehicle_timestamp", table_name="vehicle_locations"
    )
    op.create_index(
        "ix_vehicle_locations_vehicle_timestamp",
        "vehicle_locations",
        ["vehicle_id", sa.text("timestamp DESC")],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_vehicle_locations_vehicle_timestamp", table_name="vehicle_locations"
    )
    op.create_index(
        "ix_vehicle_locations_vehicle_timestamp",
        "vehicle_locations",
        ["vehicle_id", "timestamp"],
    )
