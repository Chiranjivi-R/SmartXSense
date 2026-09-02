"""Add content_hash column to datasets table

Revision ID: a1b2c3d4e5f6
Revises: 97fc1ac4fb26
Create Date: 2026-08-19

This migration adds the content_hash column that was added to the ORM model
during Phase 7B-6 (Dataset Identity / SHA-256 deduplication) but was never
applied to the existing database.  The column is nullable so existing rows
are unaffected.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '97fc1ac4fb26'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.add_column(sa.Column('content_hash', sa.String(length=64), nullable=True))
        batch_op.create_index('ix_datasets_content_hash', ['content_hash'], unique=False)


def downgrade():
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_index('ix_datasets_content_hash')
        batch_op.drop_column('content_hash')
