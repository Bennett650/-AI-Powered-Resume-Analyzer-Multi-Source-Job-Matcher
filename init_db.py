from sqlalchemy import create_engine, Column, String, Integer, Text, MetaData, Table

# Create SQLite engine and metadata
engine = create_engine('sqlite:///jobs.db')
metadata = MetaData()

# Define the jobs table
jobs_table = Table(
    'jobs', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('title', String, nullable=False),
    Column('company', String, nullable=False),
    Column('description', Text, nullable=False)
)

# Create the table
metadata.create_all(engine)

print("✅ Database and 'jobs' table created successfully.")
