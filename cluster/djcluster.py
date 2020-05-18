#!/usr/bin/python2
# -*- coding: latin-1 -*-
"""
***************************************************************************************
*    Title: <title of program/source code>
*    Author: Gambs, S., Killijian, M. O., & Nunez-del-Prado, M. 
*    Date: 2010
*    Conferece: n Proceedings of the 3rd ACM SIGSPATIAL International Workshop on Security and Privacy in GIS and LBS (pp. 34-41).
*    Availability: https://dl.acm.org/doi/pdf/10.1145/1868470.1868479
*
***************************************************************************************
__author__ = “Miguel Nunez-del-Prado“
__copyright__ = "Copyright 2020, The Cogent Project"
__credits__ = ["Miguel Nunez-del-Pradot", “Sebastien Gambs”, “Marc-Olivier Killijian”]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Miguel Nunez-del-Prado"
__email__ = “m.nunezdelpradoc@up.edu.pe”
__status__ = "Production"
"""

import logging
from mmc.mobilitytrace import MobilityTrace
from cluster import Cluster
class Djcluster (Cluster):

    def __init__ (self, minPts, eps, trialMobilityTraces, speed=0.1):
        self._mintPts = minPts
        self._eps = eps
        self._trialMobilityTraces = trialMobilityTraces
        self._speed=speed
        self._clusters =[]  # List of sets of Mobility traces grouped by cluster
        """  self._clusters =[] List of sets of Mobility traces grouped by cluster
             self._clusters = [ [ [mt_1,mt_2,mt_n],mt_medioid], [mt_1,...],mt_medioid], ... ]
        """
        self._noise = []
        self._userid = ""
        super(Djcluster,self).__init__()

    def preProcess(self):

	#h=MobilityTrace.repeated(self._trialMobilityTraces)
        #Pre-proccess
        logging.info("Number of traces: {0}".format(len(self._trialMobilityTraces)))
	#static_spaceFiltered=self._trialMobilityTraces
	#print(self._trialMobilityTraces)

        static_mts = MobilityTrace.filterSpeed(self._trialMobilityTraces,self._speed)
        logging.info("Number of traces after speed filter: {0}".format(len(static_mts)))

        static_spaceFiltered = MobilityTrace.spatial_filter(static_mts)
	#static_spaceFiltered = MobilityTrace.spatial_filter(self._trialMobilityTraces)
        logging.info("Number of traces after contiguos repeated: {0}".format(len(static_spaceFiltered)))

        return static_spaceFiltered
    #end preProcess

    def doCluster(self, preprocess = True):
        static_spaceFiltered = dict()
        #print(len(self._trialMobilityTraces),"h")

        #Pre-proccess
        if (preprocess):
                static_spaceFiltered =  self.preProcess()
        else:
            static_spaceFiltered = self._trialMobilityTraces
        #print(len(static_spaceFiltered),"static")
        new_cluster = set()

        for mt in static_spaceFiltered:
            new_cluster = set()
            new_cluster.add(mt)
            for mt1 in static_spaceFiltered:
                if (mt.distance(mt1) <=  self._eps):
                    new_cluster.add(mt1)
            #print "new cluster length: {0}, {1}".format(len(new_cluster),(len(new_cluster) >= self._mintPts ))
            if(len(new_cluster) >= self._mintPts):
                merge=False

                for c in self._clusters:
                    if(new_cluster.isdisjoint(c)==False):
                        merge = True
                        c = c.union(new_cluster)
                        break;

                if(merge==False):
                    self._clusters.append(new_cluster)
            else:
                self._noise.append(new_cluster)
        logging.info("Clusters: {0}".format(len(self._clusters)))
        logging.info("Noise: {0}".format(len(self._noise)))
        
    def doCluster2(self, preprocess = True):
        static_spaceFiltered = self._trialMobilityTraces
        new_cluster = set()

        for mt in static_spaceFiltered:
            new_cluster = set()
            new_cluster.add(mt)
            for mt1 in static_spaceFiltered:
                if (mt.distance(mt1) <=  self._eps):
                    new_cluster.add(mt1)
            #print "new cluster length: {0}, {1}".format(len(new_cluster),(len(new_cluster) >= self._mintPts ))
            if(len(new_cluster) >= self._mintPts):
                merge=False

                for c in self._clusters:
                    if(new_cluster.isdisjoint(c)==False):
                        merge = True
                        c = c.union(new_cluster)
                        break;

                if(merge==False):
                    self._clusters.append(new_cluster)
            else:
                self._noise.append(new_cluster)
        logging.info("Clusters: {0}".format(len(self._clusters)))
        logging.info("Noise: {0}".format(len(self._noise)))

    def post_proccessing(self):
        index = 0
        for c in self._clusters:
            #computes medioid
            aux_medioid=MobilityTrace.computeMediod(list(c))
            #add to dictionary
            self.dict_clusters[index]=[c,aux_medioid]
            index += 1
	#print(len(self.dict_clusters),"hello")

    def getClusters (self):
        return self.dict_clusters

#######################################
#   Properties
#######################################

    @property
    def userid (self):
        return self._userid
    @userid.setter
    def userid (self,value):
        self._userid = value


#######################################
#   export clusters
#######################################

    def getStops (self):
        str_result = ""
        setClusters = self.getClusters()

        for key in setClusters:
            aux_medioid = (setClusters[key])[1]
            str_result += str(self.userid)+","+str(aux_medioid.latitude)+","+str(aux_medioid.longitude)+"\n"
            #print(str_result,"hola")
        return str_result[:-2]
