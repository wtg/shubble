"""Drop created_by_admin from suggested_announcements

Revision ID: d3e1b7c0a452
Revises: c7d4a2e9f031
Create Date: 2026-03-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3e1b7c0a452'
down_revision: Union[str, None] = 'c7d4a2e9f031'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop the created_by_admin column."""
    op.drop_column('suggested_announcements', 'created_by_admin')


def downgrade() -> None:
    """Re-add the created_by_admin column."""
    op.add_column(
        'suggested_announcements',
        sa.Column('created_by_admin', sa.Boolean(), nullable=False, server_default='false'),
    )
