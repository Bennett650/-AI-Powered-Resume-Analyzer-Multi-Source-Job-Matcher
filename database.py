from sqlalchemy import create_engine, Column, String, Integer, Text, MetaData, Table
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///jobs.db')
metadata = MetaData()

jobs_table = Table(
    'jobs', metadata,
    Column('id', Integer, primary_key=True),
    Column('title', String),
    Column('company', String),
    Column('description', Text)
)

metadata.create_all(engine)
