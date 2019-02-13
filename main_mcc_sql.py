import csv
import logging
from mmc.trace import Trace
from mmc.mmc import Mmc
import glob
import numpy
from os import listdir , makedirs 
from os.path import isfile, join, exists, dirname, realpath
import ConfigParser
import sys
import sys
import pprint
from datetime import date, timedelta
import sqlalchemy



logging.getLogger().setLevel(logging.INFO)

def connect(user='stic', password='stic2019', db='stic', host='172.17.1.23', port=5432):
    '''Returns a connection and a metadata object'''
    # We connect with the help of the PostgreSQL URL
    # postgresql://federer:grandestslam@localhost:5432/tennis
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    # The return value of create_engine() is our connection object
    con = sqlalchemy.create_engine(url, client_encoding='utf8')

    # We then bind the connection to MetaData()
    meta = sqlalchemy.MetaData(bind=con, reflect=True)
    logging.info(con)
    logging.info(meta)

    return con, meta

def connexionBastion():
    #Connect to DB
    conn_string = "dbname='stic' port='5432' user='stic' password='stic2019' host='172.17.1.23'";
    logging.info ("Connecting to database ->{}".format(conn_string))
    conn = psycopg2.connect(conn_string);
    return conn


def extractUsersDistric():

    #Query on DB
    conn = connexionBastion()
    cursor = conn.cursor();

    #Captures Column Names 
    column_names = [];

    query = """
      SELECT aux.client_id 
      FROM (
        SELECT client_id, date_trunc('month', date) AS txn_month, SUM(amount_usd) AS monthly_sum
        FROM bbva
        WHERE agency_district = 'SAN JUAN DE LURIGANCHO'
        GROUP BY client_id, txn_month
       ) AS aux
      GROUP BY aux.client_id
      HAVING (AVG(a
      """

    cursor.execute(query);
    column_names = [desc[2] for desc in cursor.description]
    all_cols=', '.join([str(x) for x in column_names])
    print (all_cols)

if __name__ == "__main__":
   connect()
