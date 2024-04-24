import psycopg2

# Database configuration
DATABASE_URL = "dbname='investment_db' user='username' host='localhost' password='password'"

def create_tables():
    """ Create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS company_profiles (
            company_id SERIAL PRIMARY KEY,
            about TEXT,
            keywords TEXT,
            label INTEGER
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS api_calls (
            call_id SERIAL PRIMARY KEY,
            query TEXT,
            prediction INTEGER,
            prediction_confidence FLOAT
        )
        """)
    conn = None
    try:
        # connect to the PostgreSQL server
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not closed:
            conn.close()

def insert_company_profile(about, keywords, label):
    """ Insert a new row into the company_profiles table """
    sql = """INSERT INTO company_profiles(about, keywords, label)
             VALUES(%s, %s, %s) RETURNING company_id;"""
    conn = None
    company_id = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(sql, (about, keywords, label))
        company_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not closed:
            conn.close()
    return company_id

def insert_api_call(query, prediction, prediction_confidence):
    """ Insert a record into the api_calls table """
    sql = """INSERT INTO api_calls(query, prediction, prediction_confidence)
             VALUES(%s, %s, %s) RETURNING call_id;"""
    conn = None
    call_id = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(sql, (query, prediction, prediction_confidence))
        call_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not closed:
            conn.close()
    return call_id
