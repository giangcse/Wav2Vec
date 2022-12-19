import sqlite3

db = 'database.db'
conn = sqlite3.connect(db, check_same_thread=False)
cursor = conn.cursor()

print([x[0] for x in cursor.execute("SELECT AUDIO_NAME FROM audios WHERE USERNAME = ?", ('admin',))])