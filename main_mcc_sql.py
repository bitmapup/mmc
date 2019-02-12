import csv
import logging
#from query import extract_users_mob_traces
from cluster.djcluster import Djcluster
from cluster.dbscan_mmc import Dbscan_mmc
from mmc.trace import Trace
from mmc.mmc import Mmc
from query import *
import glob
import numpy
from os import listdir , makedirs 
from os.path import isfile, join, exists, dirname, realpath
import ConfigParser
import sys
import multiprocessing
import datetime


logging.getLogger().setLevel(logging.INFO)


def connexionBastion():
    conn = psycopg2.connect(
           host="localhost",
           database="suppliers", 
           user="postgres", 
           password="postgres"
           )

if __name__ == "__main__":
   #buildSubscribersMmc("data/TD184244.csv")
   buildSubscribersMmc("data/TD184258.csv")
   #buildSubscribersMmc("data/TD184271.csv")
   #buildSubscribersMmc("data/TD184277.csv")
