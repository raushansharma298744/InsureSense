"""Initial migration - create all tables

Revision ID: 001
Revises: 
Create Date: 2026-11-07 02:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=20), nullable=False, server_default='employee'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)

    # Create predictions table
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('gender', sa.String(length=20), nullable=False),
        sa.Column('age', sa.Integer(), nullable=False),
        sa.Column('driving_license', sa.Integer(), nullable=False),
        sa.Column('region_code', sa.Float(), nullable=False),
        sa.Column('previously_insured', sa.Integer(), nullable=False),
        sa.Column('vehicle_age', sa.String(length=20), nullable=False),
        sa.Column('vehicle_damage', sa.String(length=10), nullable=False),
        sa.Column('annual_premium', sa.Float(), nullable=False),
        sa.Column('policy_sales_channel', sa.Float(), nullable=False),
        sa.Column('vintage', sa.Integer(), nullable=False),
        sa.Column('predicted_response', sa.Integer(), nullable=False),
        sa.Column('purchase_probability', sa.Float(), nullable=False),
        sa.Column('propensity_category', sa.String(length=20), nullable=False),
        sa.Column('threshold_used', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_predictions_id'), 'predictions', ['id'], unique=False)
    op.create_index(op.f('ix_predictions_user_id'), 'predictions', ['user_id'], unique=False)
    op.create_index(op.f('ix_predictions_propensity_category'), 'predictions', ['propensity_category'], unique=False)
    op.create_index('ix_predictions_user_created', 'predictions', ['user_id', 'created_at'], unique=False)

    # Create batch_jobs table
    op.create_table(
        'batch_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('total_rows', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('processed_rows', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_rows', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('result_file_path', sa.String(length=500), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_batch_jobs_id'), 'batch_jobs', ['id'], unique=False)
    op.create_index(op.f('ix_batch_jobs_user_id'), 'batch_jobs', ['user_id'], unique=False)
    op.create_index(op.f('ix_batch_jobs_status'), 'batch_jobs', ['status'], unique=False)
    op.create_index('ix_batch_jobs_user_created', 'batch_jobs', ['user_id', 'created_at'], unique=False)

    # Create batch_predictions table
    op.create_table(
        'batch_predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('batch_job_id', sa.Integer(), nullable=False),
        sa.Column('row_id', sa.Integer(), nullable=True),
        sa.Column('customer_id', sa.String(length=100), nullable=True),
        sa.Column('input_features', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('predicted_response', sa.Integer(), nullable=True),
        sa.Column('purchase_probability', sa.Float(), nullable=True),
        sa.Column('propensity_category', sa.String(length=20), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['batch_job_id'], ['batch_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_batch_predictions_id'), 'batch_predictions', ['id'], unique=False)
    op.create_index(op.f('ix_batch_predictions_batch_job_id'), 'batch_predictions', ['batch_job_id'], unique=False)

    # Create model_versions table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('model_type', sa.String(length=100), nullable=False),
        sa.Column('training_date', sa.DateTime(), nullable=False),
        sa.Column('training_dataset_size', sa.Integer(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('precision', sa.Float(), nullable=True),
        sa.Column('recall', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('roc_auc', sa.Float(), nullable=True),
        sa.Column('pr_auc', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='staged'),
        sa.Column('model_path', sa.String(length=500), nullable=True),
        sa.Column('pipeline_path', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('activated_at', sa.DateTime(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('version')
    )
    op.create_index(op.f('ix_model_versions_id'), 'model_versions', ['id'], unique=False)
    op.create_index(op.f('ix_model_versions_version'), 'model_versions', ['version'], unique=True)
    op.create_index(op.f('ix_model_versions_is_active'), 'model_versions', ['is_active'], unique=False)

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_id'), 'audit_logs', ['id'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index('ix_audit_logs_user_created', 'audit_logs', ['user_id', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_audit_logs_user_created', table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_action'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_id'), table_name='audit_logs')
    op.drop_table('audit_logs')

    op.drop_index(op.f('ix_model_versions_is_active'), table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_version'), table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_id'), table_name='model_versions')
    op.drop_table('model_versions')

    op.drop_index(op.f('ix_batch_predictions_batch_job_id'), table_name='batch_predictions')
    op.drop_index(op.f('ix_batch_predictions_id'), table_name='batch_predictions')
    op.drop_table('batch_predictions')

    op.drop_index('ix_batch_jobs_user_created', table_name='batch_jobs')
    op.drop_index(op.f('ix_batch_jobs_status'), table_name='batch_jobs')
    op.drop_index(op.f('ix_batch_jobs_user_id'), table_name='batch_jobs')
    op.drop_index(op.f('ix_batch_jobs_id'), table_name='batch_jobs')
    op.drop_table('batch_jobs')

    op.drop_index('ix_predictions_user_created', table_name='predictions')
    op.drop_index(op.f('ix_predictions_propensity_category'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_user_id'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_id'), table_name='predictions')
    op.drop_table('predictions')

    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')