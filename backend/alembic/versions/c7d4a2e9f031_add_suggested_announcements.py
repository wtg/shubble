"""Add suggested_announcements table

Revision ID: c7d4a2e9f031
Revises: 1cc2da604c8f
Create Date: 2026-03-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7d4a2e9f031'
down_revision: Union[str, None] = '1cc2da604c8f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'suggested_announcements',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('suggestion', sa.String(), nullable=False),
        sa.Column('created_by_admin', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('suggested_announcements')
