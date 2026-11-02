
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Read database settings
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not DB_PASSWORD or DB_PASSWORD == "YOUR_MYSQL_PASSWORD":
    raise RuntimeError(
        "Set your MySQL password in the .env file."
    )

# Build the MySQL connection URL safely
DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("DB_USER", "root"),
    password=DB_PASSWORD,
    host=os.getenv("DB_HOST", "127.0.0.1"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME", "emotion_detection"),
)

# Create the database engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

# Create database sessions
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

# Base class for database models
Base = declarative_base()


# Provide a session to API endpoints
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
