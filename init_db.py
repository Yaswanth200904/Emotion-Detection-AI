from database import Base, engine
from models import Prediction

# Create the database table if it doesn't exist
Base.metadata.create_all(bind=engine)

print("MySQL connection successful!")
print("Predictions table is ready.")