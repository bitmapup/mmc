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

LABELS = ["CODMES","CLIENTE1","FECHA_OPER","CODGIRO","CODPAIS","MONTO","TRX","CODCLIAEDAD","CODCLIASEXO","CODCLIAUBIGEO_BANCO","CODCLIADEPARTA    MENTO","CODCLIAPROVINCIA","CODCLIADISTRITO","CODCLIAFLAG_LIMA_PROVINCIA","CODCLIAREGION","CODCLIAINGRESO","CODCOMERCIO","COMERCIO","    CODCOMERRUBRO","CODCOMERUBIGEO_ESTAB","CODCOMERDEPARTAMENTO_ESTAB","CODCOMERPROVINCIA_ESTAB","CODCOMERDISTRITO_ESTAB","CODCOMERLONG_X_MC_C","CODCOMERLONG_Y_MC_C"]

#################################################
#	Model construction			#
#################################################

def buildSubscribersMmc (mtFile):
    #print inputFilePath
    labels = ["user_id","timestamp","arr_id"]
    trailmt = list()
    processedUsers = list()
    pDaysArray=[False,False,False,False,False,False,False,False,False,True]
    pTimeslices =  1
    state = set()
    trailTraces = list()
    idUser = ""

    with open(mtFile) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter='|')
      eps = 0.1
      for row in csv_reader:
        if (row[LABELS.index("CODGIRO")]!="" and 
           row[LABELS.index("CODCOMERLONG_X_MC_C")]!="" and
           row[LABELS.index("CODCOMERLONG_Y_MC_C")]!=""):

           state.add(row[LABELS.index("CODGIRO")])
           idUser = row[LABELS.index("CLIENTE1")]
           ts = row[LABELS.index("FECHA_OPER")]
           mcc = row[LABELS.index("CODGIRO")]
           amount=float(row[LABELS.index("MONTO")])
           latitude=float(row[LABELS.index("CODCOMERLONG_X_MC_C")])
           longitude=float(row[LABELS.index("CODCOMERLONG_Y_MC_C")])
           
           t = Trace(idUser,ts,mcc,amount,latitude,longitude)
           trailTraces.append(t)
           
           print(row[LABELS.index("CLIENTE1")],
                 row[LABELS.index("FECHA_OPER")],
                 row[LABELS.index("CODGIRO")],
                 float(row[LABELS.index("MONTO")]),
                 amount,
                 float(row[LABELS.index("CODCOMERLONG_X_MC_C")]),
                 float(row[LABELS.index("CODCOMERLONG_Y_MC_C")])
                 )
           print(t)

    #building mobility models
    if (len(state)>= 3):
        sort_state = sorted(state) 
	oMmc = Mmc(list(sort_state),
		trailTraces,idUser,
		daysArray=pDaysArray,
		timeSlices=pTimeslices,
		radius=eps
		)
	oMmc.buildModel()

        for srow in oMmc._transitionMatrix:
            srow_len = len (srow)
            sum_zeros = numpy.count_nonzero(srow_len)
            if (sum_zeros == srow_len):
               print ("ROW EMPTY !!!")


        print (oMmc.stationaryVector)
        print (oMmc.shannonEntropy(),
               oMmc.predictability(),
               Trace.compute_frequency_update(trailTraces),
               Trace.compute_cumulated_distance(trailTraces),
               Trace.compute_cumulated_spent(trailTraces)
               )
        print (oMmc)
        print (oMmc.distance(oMmc))
      
	#oMmc.export(outputFolder)
	#print (oMmc._spatiaTemporallLabeledTrailmt)

#end buildSubscribersMmc

#################################################

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
