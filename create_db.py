import sqlite3

# Connect to the database (or create it if it doesn't exist)
conn = sqlite3.connect('sample.db')
cursor = conn.cursor()

# Create the employees table
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    job TEXT NOT NULL,
    salary INTEGER NOT NULL
)
''')

# Clear existing data to avoid duplicates on re-running
cursor.execute("DELETE FROM employees")

# Insert some sample data
employees = [
    ('Alice', 'Software Engineer', 120000),
    ('Bob', 'Data Scientist', 135000),
    ('Charlie', 'Product Manager', 140000),
    ('David', 'DevOps Engineer', 130000)
]

cursor.executemany("INSERT INTO employees (name, job, salary) VALUES (?, ?, ?)", employees)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database 'sample.db' created and populated successfully.")
