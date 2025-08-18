from app.db.database import Base, engine
from app.db.models import User, Chat, Message
from app.core.security import get_password_hash
import uuid
from datetime import datetime

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")
    
def create_demo_user():
    from sqlalchemy.orm import sessionmaker
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.username == "demo").first()
        if not user:
            user = User(
                id=str(uuid.uuid4()),
                username="demo",
                display_name="Demo User",
                hashed_password=get_password_hash("password"),
                created_at=datetime.utcnow()
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Demo user created: demo (ID: {user.id})")
            
            chat = Chat(
                id=str(uuid.uuid4()),
                user_id=user.id,
                title="Welcome to the Remote Sensing Image Analysis System",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            db.add(chat)
            db.commit()
            db.refresh(chat)
            print(f"Default chat created: {chat.title} (ID: {chat.id})")
            
            welcome_message = Message(
                id=str(uuid.uuid4()),
                chat_id=chat.id,
                text="Hello, welcome to the YAOGAN Chat System! Please start a conversation or upload a remote sensing image for analysis.",
                sender="ai",
                timestamp=datetime.utcnow()
            )
            db.add(welcome_message)
            db.commit()
            print("Welcome message added.")
        else:
            print("Demo user already exists, skipping creation.")
            
    except Exception as e:
        print(f"Error creating demo user: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Starting database initialization...")
    init_db()
    create_demo_user()
    print("Database initialization complete.")