import csv
import logging
#from query import extract_users_mob_traces
from cluster.djcluster import Djcluster
from cluster.dbscan_mmc import Dbscan_mmc
from mmc.mobilitytrace import MobilityTrace
from mmc.mmc import Mmc
import glob
import numpy
from os import listdir
from os.path import isfile, join
import collections


def getArrLocation (path="/srv/data/ssd/nunez/extra_data/d4d/root/ContextData/SITE_ARR_LONLAT.CSV"):
    labels = ["site_id","arr_id","lon","lat"]
    dic_antennas = dict()
    with open(path,'rb') as tsvin:
        logging.info("Open: {0}".format(path))
        tsvin = csv.reader(tsvin, delimiter=',')
        tsvin.next()
        for row in tsvin:
            #print (row)
            site_id = row[labels.index("site_id")]
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

def computeCentroid (dic_antennas):
  with open('regions.csv','wb') as f:
    f.write("arr_id,latitude,longitude\n")

    dic_regions = dict()
    for arr_id in dic_antennas:
        list_antennas = dic_antennas[arr_id]

        nb_antennas = 0
        mean_lat = 0
        mean_lon = 0

        for antenna in list_antennas:
            mean_lat += float(antenna[0])
            mean_lon += float(antenna[1])
            nb_antennas += 1

        mean_lat = float(mean_lat)/float(nb_antennas)
        mean_lon = float(mean_lon)/float(nb_antennas)
        dic_regions[arr_id] = [mean_lat,mean_lon]
        f.write("{0},{1},{2}\n".format(arr_id,mean_lat,mean_lon))
  f.close()
  return (dic_regions)
#end computeCentroid


def getRegionLocation (path="SUBPREF_POS_LONLAT.TSV"):
    labels = ["arr_id","lat","lon"]
    dic_antennas = dict()
    with open(path,'rb') as tsvin:
        logging.info("Open: {0}".format(path))
        tsvin = csv.reader(tsvin, delimiter='\t')
        tsvin.next()
        for row in tsvin:
            #print (row)
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

def buildSubscribersMmc (locationDictionary,inputFilePath,outputFolder):
    labels = ["user_id","timestamp","arr_id"]
    trailmt = dict()
    processedUsers = list()
    pDaysArray=[False,False,False,False,False,False,False,False,False,True]
    pTimeslices =  1

    with open(inputFilePath,'rb') as tsvin:
        logging.info("Open: {0}".format(inputFilePath))
        tsvin = csv.reader(tsvin, delimiter='\t')
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
            if ((aux_mt.latitude != 0) & (aux_mt.longitude != 0)):
                if (idUser in trailmt):
                    (trailmt[idUser]).append(aux_mt)
                elif(len(trailmt.items()) == 0):
                    trailmt[idUser] = [aux_mt]
                    if (not idUser in processedUsers):
                        processedUsers.append(idUser)
                else:
                    minpts = 2
                    eps = 10
                    key = processedUsers[-1]
                    oDjCluster = Djcluster(minpts,eps,trailmt[key])
                    #clustering
                    oDjCluster.doCluster()
                    oDjCluster.post_proccessing()

                    #building mobility models
                    if (len(oDjCluster.dict_clusters)>= 3):
                        oMmc = Mmc(oDjCluster,
                                trailmt[key],key,
                                daysArray=pDaysArray,
                                timeSlices=pTimeslices,
                                radius=eps
                                )
                        oMmc.buildModel()
                        oMmc.export(outputFolder)
                        #oMmc.export("models/d4d/")
                    trailmt = dict()
                    trailmt[idUser] = [aux_mt]
                    if (not idUser in processedUsers):
                        processedUsers.append(idUser)

#end buildSubscribersMmc


def buildHeatMap (locationDictionary):
    labels = ["user_id","timestamp","arr_id"]
    dict_count = dict()
    i = 1
    inputFilePath = ""
    while (i<13):
        if i<10:
            inputFilePath = "/srv/data/ssd/nunez/extra_data/d4d/root/SET3/sorted_SET3_M0{0}.CSV".format(i)
        else:
            inputFilePath = "/srv/data/ssd/nunez/extra_data/d4d/root/SET3/sorted_SET3_M{0}.CSV".format(i)

        with open(inputFilePath,'rb') as tsvin:
            logging.info("Open: {0}".format(inputFilePath))
            tsvin = csv.reader(tsvin, delimiter=',')
            for row in tsvin:
                idUser = row[labels.index("user_id")]
                idArr = row[labels.index("arr_id")]#arr_id"
                latitude = 0.0
                longitude = 0.0
                if idArr in locationDictionary:
                    lat = (locationDictionary[idArr][0])[0]
                    lon = (locationDictionary[idArr][0])[1]

                if idArr in dict_count:
                    (dict_count[idArr])[2]+=1
                else:
                    dict_count[idArr] = [lat,lon,0]
        i += 1

    print "id,count,latitude,longitude"

    for idArr in dict_count:
        print "{0},{1},{2},{3}".format(idArr,dict_count[idArr][2],dict_count[idArr][0],dict_count[idArr][1])
