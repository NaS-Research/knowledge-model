"""
init_db.py
----------
Initializes the database (creates tables if they don't exist).
"""

from knowledge_model.db.db_session import init_db

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
