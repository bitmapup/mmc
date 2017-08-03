import csv
import logging
#from query import extract_users_mob_traces
from cluster.djcluster import Djcluster
from cluster.dbscan_mmc import Dbscan_mmc
from mmc.mobilitytrace import MobilityTrace
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
#import os
#script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
#abs_file_path = os.path.join(script_dir, rel_path)

global REGIONS 
global OUTPUTPATH
global INPUTPATH
global INPUT
global OUTPUT
global SCRIPT_DIR 
global LOGPATH

logging.getLogger().setLevel(logging.INFO)

#################################################
#	Model construction			#
#################################################

def getRegionLocation (path,pDelimiter=","):
    labels = ["","arr_id","lat","lon"]
    dic_antennas = dict()
    abs_file_path = join(SCRIPT_DIR, path)
    with open(abs_file_path,'rb') as tsvin:
        logging.info("Open: {0}".format(path))
        tsvin = csv.reader(tsvin, delimiter=pDelimiter)
        tsvin.next()
        for row in tsvin:
            arr_id = row[labels.index("arr_id")]
            lon = row[labels.index("lon")]
            lat = row[labels.index("lat")]
            latlon = [lat,lon]
            if arr_id in dic_antennas:
                dic_antennas[arr_id].append(latlon)
            else:
                dic_antennas[arr_id] = [latlon]

    return (dic_antennas)
#end getLanLon

def buildSubscribersMmc (mtFile):
    locationDictionary =  REGIONS
    minpts = MINPTS
    eps = EPS
    outputFolder = OUTPUTPATH    
    inputFilePath = INPUTPATH+"/"+mtFile
    #print inputFilePath
    labels = ["user_id","timestamp","arr_id"]
    trailmt = list()
    processedUsers = list()
    pDaysArray=[False,False,False,False,False,False,False,False,False,True]
    pTimeslices =  1

    with open(inputFilePath,'rb') as tsvin:
        logging.info("Open: {0}".format(inputFilePath))
        tsvin = csv.reader(tsvin, delimiter=',')
        for row in tsvin:
            idUser = row[labels.index("user_id")]
            idArr = row[labels.index("arr_id")]#arr_id"
            latitude = 0.0
            longitude = 0.0
            if idArr in locationDictionary:
                latitude = (locationDictionary[idArr][0])[0]
                longitude = (locationDictionary[idArr][0])[1]

            aux_mt = MobilityTrace(
                    row[labels.index("timestamp")],
                    idArr,#arr_id"
                    latitude,
                    longitude,
                    "gsm"
                    )
	    trailmt.append(aux_mt)
    for t in trailmt:
	print t
    oDjCluster = Djcluster(minpts,eps,trailmt)
    #clustering
    oDjCluster.doCluster()
    oDjCluster.post_proccessing()
    #building mobility models
    if (len(oDjCluster.dict_clusters)>= 2):
	oMmc = Mmc(oDjCluster,
		trailmt,idUser,
		daysArray=pDaysArray,
		timeSlices=pTimeslices,
		radius=eps
		)
	oMmc.buildModel()
	oMmc.export(outputFolder)

#end buildSubscribersMmc

def collect_result(result):
    results.append(result)
#################################################

if __name__ == "__main__":
    result = []    
    if len(sys.argv)<=1:
	    print "ERROR: You need to specify the path of the config file"
    else:
            cfgName = sys.argv[1]
	    # Read config file
	    config = ConfigParser.ConfigParser()
	    config.read(cfgName)
	    logging.info("Reading configuration")
	    print (config.sections())
	    inputFilePath  =  str(config.get('path','inputFilePath'))
	    inputFilePath = inputFilePath.replace("\"","")
	    logging.info("inputFilePath: {}".format(inputFilePath))
	    outputFilePath = config.get('path','outputFilePath')
	    experimentName = config.get('experiment','name')
	    experimentName = experimentName.replace("\"","")
	    outputFilePath =  outputFilePath+experimentName+"/"
	    outputFilePath = outputFilePath.replace("\"","")
	    logging.info("outputfile: {}".format(outputFilePath))
	    locationPath =  config.get('path','locationFile')
	    locationPath = locationPath.replace("\"","")
	    logging.info("locationPath: {}".format(locationPath))
	    logPath =  config.get('experiment','log')
	    logPath = logPath.replace("\"","")
	    logging.info("locationPath: {}".format(logPath))


	    #this method is for getting the region location once is already computed
	    SCRIPT_DIR = dirname(realpath('__file__'))
	    REGIONS = getRegionLocation(locationPath)
	    OUTPUTPATH =  join(SCRIPT_DIR,outputFilePath)
	    INPUTPATH = join(SCRIPT_DIR,inputFilePath)
	    MINPTS = int(config.get('parameters','minpts'))
	    EPS = float(config.get('parameters','eps'))
	    LOGPATH = join(SCRIPT_DIR,"users_{}.txt".format(experimentName))
	    print "SCRIPT_DIR: {} ".format(LOGPATH)
	    logging.info("Parameters:  mintps:{}  eps:{}".format(MINPTS,EPS))
	    #Read mobility traces files
	    onlyfiles = [f for f in listdir(inputFilePath) if isfile(join(inputFilePath, f))]
	    logging.info("Number of loaded files : {}".format(len(onlyfiles)))
	    # create output folder
	    if not exists(outputFilePath):
    		makedirs(outputFilePath)
	    #Multiprocessing
	    t_begin =  datetime.datetime.now()
	    t_end =  datetime.datetime.now()
	    try:
		    t_begin =  datetime.datetime.now()
		    #use all available cores, otherwise specify the number you want as an argument
		    pool = multiprocessing.Pool(11) 
		    #logfile=open("logusers/users_{}.txt".format(experimentName),"w")
		    logfile=open(LOGPATH,"w")
		    for f in onlyfiles:
			logfile.write("{}\n".format(f))
			pool.apply_async(buildSubscribersMmc, [f])
		    pool.close()
		    pool.join()
		    t_end =  datetime.datetime.now()
		    print "The work end successfully in {} time".format(str(t_end-t_begin))
	    finally:
		    print "[If there are 2 messages with the same time, work is OK] The work did not end  successfully in {} time".format(str(t_end-t_begin))
	    #buildSubscribersMmc("113161.csv")
