import os
from sqlalchemy import create_engine, inspect

database_url = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://postgres:changeme@postgres:5432/insuresense')
engine = create_engine(database_url)
insp = inspect(engine)
print('Tables:', insp.get_table_names())