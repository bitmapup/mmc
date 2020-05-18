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
import os
import pandas as pd
import psycopg2

global OUTPUTPATH
global INPUTPATH
global INPUT
global OUTPUT
global SCRIPT_DIR
global LOGPATH

logging.getLogger().setLevel(logging.INFO)

def conectionDB():
    conn_string = "dbname='covid' port='5432' user='coviduser' password='covid' host='172.250.1.42'";
    conn = psycopg2.connect(conn_string);
    return conn

def buildSubscribersMmc (mtFile):
    minpts = MINPTS
    eps = EPS
    outputFolder = OUTPUTPATH
    inputFilePath = INPUTPATH+"/"+mtFile
    labels = ["user_id","timestamp","latitude","longitude"]
    trailmt = list()
    processedUsers = list()
    pDaysArray=[False,False,False,False,False,False,False,False,False,True]
    pTimeslices =  1
    id=0
    try:
        with open(inputFilePath,'rb') as tsvin:
            logging.info("Open: {0}".format(inputFilePath))
            tsvin = csv.reader(tsvin, delimiter=',')
            next(tsvin)
            for row in tsvin:
                idUser = row[labels.index("user_id")]
                id=id+1
                latitude=row[labels.index("latitude")]
                longitude=row[labels.index("longitude")]
                latitude=latitude.replace(",",".")
                longitude=longitude.replace(",",".")
                aux_mt = MobilityTrace(row[labels.index("timestamp")],id,latitude,longitude)
                trailmt.append(aux_mt)
        #for t in trailmt:
        #    print(t)
        oDjCluster = Djcluster(minpts,eps,trailmt)
        oDjCluster.doCluster()
        oDjCluster.post_proccessing()
        print(len(oDjCluster.dict_clusters),"cluster")
        if(len(oDjCluster.dict_clusters)<= 1):
            if(len(oDjCluster.dict_clusters)== 0):
                pass
            else:
                now=datetime.datetime.now()
                d=oDjCluster.dict_clusters
                lat=[float(d[0][1].latitude)]
                lon=[float(d[0][1].longitude)]
                prob=[1.0]
                poi=[0]
                device=[idUser]
                computedate=[now]
                predictability=[1.0]
                entropy=[1.0]
                average=[0.0]
            df2 = pd.DataFrame(
             {
             'device_id': idUser,
             'predictability': predictability,
             'entropy': entropy ,
             'average_distance': average,
             'computedate' : computedate
             },columns=["device_id","predictability","entropy","average_distance","computedate"])
            df2.to_csv("insertdata2.csv",index=False)
        else:
            oMmc = Mmc(oDjCluster,trailmt,idUser,daysArray=pDaysArray,timeSlices=pTimeslices,radius=eps)
            oMmc.buildModel()
            now = datetime.datetime.now()
            vector=oMmc.vectorstationary()
            vector.sort(key=lambda x: x[0],reverse=True)
            prob=map(float, [vector[i][0] for i in range(len(vector))])
            lat=map(float, [vector[i][1] for i in range(len(vector))])
            lon=map(float, [vector[i][2] for i in range(len(vector))])
            poi=list(range(0,len(vector)))
            device=[idUser]*len(vector)
            computedate = [now]*len(vector)
            computedate2 = [now]
            predictability=[oMmc.predictability()]
            entropy=[oMmc.shannonEntropy()]
            a,deviation=oMmc.averageperday()
            average=[a]
            oMmc.export(outputFolder)
            df2 = pd.DataFrame(
             {
             'device_id': idUser,
             'predictability': predictability,
             'entropy': entropy ,
             'average_distance': average,
             'computedate' : computedate2
             },columns=["device_id","predictability","entropy","average_distance","computedate"])
            df2.to_csv("insertdata2.csv",index=False)
        try:
            df = pd.DataFrame(
             {
             'device_id': device,
             'cod_poi': poi,
             'probabilidad': prob,
             'x': lon,
             'y': lat,
             'computedate' : computedate
             },columns=["device_id","cod_poi","probabilidad","x","y","computedate"])
            conn = conectionDB()
            cursor=conn.cursor()
            df.to_csv("insertdata.csv",index=False)
            with open("insertdata.csv",'r') as f:
                print("Inserting data...")
                reader = csv.reader(f,delimiter=',')
                reader.next()
                for row in reader:
                    cursor.execute(
                    "INSERT INTO mmc VALUES (%s,%s,%s,%s,%s,%s)",
                    row
                    )
                    conn.commit()
            conn.close()
            conn = conectionDB()
            cursor=conn.cursor()
            df.to_csv("insertdata.csv",index=False)
            with open("insertdata2.csv",'r') as f:
                print("Inserting data2...")
                reader = csv.reader(f,delimiter=',')
                reader.next()
                for row in reader:
                    cursor.execute(
                    "INSERT INTO mmcproperties VALUES (%s,%s,%s,%s,%s)",
                    row
                    )
                    conn.commit()
            conn.close()
        except:
            pass
    except:
        pass



if __name__ == "__main__":
    result = []
    if len(sys.argv)<=1:
        print "ERROR: You need to specify the path of the config file"
    else:
        cfgName = sys.argv[1]
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

        logPath =  config.get('experiment','log')
        logPath = logPath.replace("\"","")
        logging.info("locationPath: {}".format(logPath))
        SCRIPT_DIR = dirname(realpath('__file__'))
        OUTPUTPATH =  join(SCRIPT_DIR,outputFilePath)
        INPUTPATH = join(SCRIPT_DIR,inputFilePath)
        MINPTS = int(config.get('parameters','minpts'))
        EPS = float(config.get('parameters','eps'))
        LOGPATH = join(SCRIPT_DIR,"users_{}.txt".format(experimentName))
        print "SCRIPT_DIR: {} ".format(LOGPATH)
        if not exists(outputFilePath):
            makedirs(outputFilePath)
        t_begin =  datetime.datetime.now()
        t_end =  datetime.datetime.now()
        labels2=["idUser","Pois","Age","Gender","Entropy","Averageperday","Predictability","Deviation"]
        with open("properties.csv",'wb') as f:
            writer=csv.writer(f,delimiter=',')
            writer.writerow(labels2)
        lista_dataframe=[]
        try:
            t_begin =  datetime.datetime.now()
            conn=conectionDB()
            cursor=conn.cursor();
            query = '''
                select device_id,datetime,y,x
                from location
                where device_id like 'f%'
                and x is not null and y is not null and x!=y
                order by device_id,datetime ;
                '''
            cursor.execute(query)
            rows = cursor.fetchall()
            df = pd.read_sql(query, conn)
            conn.close()
            conn=conectionDB()
            cursor=conn.cursor();
            query = '''
                select device_id
                from register
                where device_id like 'f%' and datetime<'2020-05-10 00:00:00';
            '''
            cursor.execute(query)
            rows = cursor.fetchall()
            df2=pd.read_sql(query,conn)
            datos=df2['device_id'].tolist()
            print(len(datos))
            cont=1
            for d in datos:
                print(cont)
                dataframe=df[df['device_id']==d].reset_index(drop=True)
                dataframe.to_csv("data/d4d_se/SET3M01/datos.csv",index=False)
                cont=cont+1
                onlyfiles = [f for f in listdir(inputFilePath) if isfile(join(inputFilePath, f))]
                for f in onlyfiles:
                    logfile=open(LOGPATH,"w")
                    data=buildSubscribersMmc(f)
            t_end =  datetime.datetime.now()
            print "The work end successfully in {} time".format(str(t_end-t_begin))
        finally:
		    print "[If there are 2 messages with the same time, work is OK] The work did not end  successfully in {} time".format(str(t_end-t_begin))
