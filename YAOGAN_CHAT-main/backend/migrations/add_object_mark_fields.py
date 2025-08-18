import sqlite3

def run_migration():
    print("Starting to run migration script...")
    
    conn = sqlite3.connect('yaogan_chat.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_info(messages)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'object_coordinates' not in columns:
            print("Adding 'object_coordinates' column...")
            cursor.execute("ALTER TABLE messages ADD COLUMN object_coordinates TEXT")
        
        if 'is_object_mark' not in columns:
            print("Adding 'is_object_mark' column...")
            cursor.execute("ALTER TABLE messages ADD COLUMN is_object_mark BOOLEAN DEFAULT 0")
        
        conn.commit()
        print("Migration completed!")
        
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_migration()