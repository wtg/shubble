"""Add upvotes and downvotes to announcements

Revision ID: a1f3c8e2d905
Revises: d3e1b7c0a452
Create Date: 2026-03-07 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1f3c8e2d905'
down_revision: Union[str, None] = 'd3e1b7c0a452'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add upvotes and downvotes columns."""
    op.add_column('announcements', sa.Column('upvotes', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('announcements', sa.Column('downvotes', sa.Integer(), nullable=False, server_default='0'))


def downgrade() -> None:
    """Remove upvotes and downvotes columns."""
    op.drop_column('announcements', 'upvotes')
    op.drop_column('announcements', 'downvotes')
