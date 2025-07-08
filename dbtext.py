import sqlite3

DB_NAME = "sessions.db"
OUTPUT_FILE = "database_report.txt"

def export_db_to_text(db_name, output_file):
    """
    Reads all tables and their data from the SQLite database and writes it to a text file.

    Args:
        db_name (str): Name of the SQLite database file.
        output_file (str): Name of the text file to write the data to.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Get all table names in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # If no tables exist, print a message
        if not tables:
            print("No tables found in the database.")
            return

        # Open the text file for writing
        with open(output_file, "w") as f:
            for table in tables:
                table_name = table[0]
                f.write(f"\n\nTable: {table_name}\n")
                f.write("=" * (len(table_name) + 7) + "\n")

                # Fetch all rows from the current table
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()

                # Write the column headers (for the current table)
                column_names = [description[0] for description in cursor.description]
                f.write(f"{' | '.join(column_names)}\n")
                f.write("-" * 50 + "\n")

                # Write the rows of the table
                for row in rows:
                    f.write(f"{' | '.join(map(str, row))}\n")
                f.write("=" * 50 + "\n")  # End of table

        print(f"Database content successfully exported to '{output_file}'")

    except sqlite3.Error as e:
        print(f"Database error: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()

# Run the function
if __name__ == "__main__":
    export_db_to_text(DB_NAME, OUTPUT_FILE)
