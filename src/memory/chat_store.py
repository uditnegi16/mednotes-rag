import sqlite3
from datetime import datetime


DB_PATH = "data/chat_history.db"


class ChatStore:


    def __init__(self):

        self.conn = sqlite3.connect(
            DB_PATH,
            check_same_thread=False
        )

        self.create_tables()



    def create_tables(self):

        cursor = self.conn.cursor()


        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT
        )
        """)


        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
        """)


        self.conn.commit()



    def create_chat(self):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO chats(created_at)
            VALUES(?)
            """,
            (
                datetime.now().isoformat(),
            )
        )

        self.conn.commit()

        return cursor.lastrowid



    def add_message(
        self,
        chat_id,
        role,
        content
    ):

        cursor = self.conn.cursor()


        cursor.execute(
            """
            INSERT INTO messages(
                chat_id,
                role,
                content,
                created_at
            )
            VALUES(?,?,?,?)
            """,
            (
                chat_id,
                role,
                content,
                datetime.now().isoformat()
            )
        )


        self.conn.commit()



    def get_history(
        self,
        chat_id
    ):

        cursor = self.conn.cursor()


        cursor.execute(
            """
            SELECT role, content
            FROM messages
            WHERE chat_id=?
            ORDER BY id
            """,
            (
                chat_id,
            )
        )


        rows = cursor.fetchall()


        return [
            {
                "role": row[0],
                "content": row[1]
            }
            for row in rows
        ]



    def get_all_chats(self):

        cursor = self.conn.cursor()


        cursor.execute(
            """
            SELECT id, created_at
            FROM chats
            ORDER BY id DESC
            """
        )


        return cursor.fetchall()