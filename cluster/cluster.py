#!/usr/bin/python2
# -*- coding: latin-1 -*-
"""
/***************************************************************************************
*    Title: <title of program/source code>
*    Author: Gambs, S., Killijian, M. O., & Nunez-del-Prado, M. 
*    Date: 2010
*    Conferece: n Proceedings of the 3rd ACM SIGSPATIAL International Workshop on Security and Privacy in GIS and LBS (pp. 34-41).
*    Availability: https://dl.acm.org/doi/pdf/10.1145/1868470.1868479
*
***************************************************************************************/

__author__ = “Miguel Nunez-del-Prado“
__copyright__ = "Copyright 2020, The Cogent Project"
__credits__ = ["Miguel Nunez-del-Pradot", “Sebastien Gambs”, “Marc-Olivier Killijian”]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Miguel Nunez-del-Prado"
__email__ = “m.nunezdelpradoc@up.edu.pe”
__status__ = "Production"
"""

class Cluster( object ):
    """ Abstract class to use as pattern for clustering algorithms
        implementations.
        The basic attribute is a dictionary having cluster index as key,
        a list containing another list of properties like avg speed,
        entropie, ... and a trial of mobilitraces in a list as kalues
        [cluster_id : [properties], [trialMobilityTraces] ]
        properties is [avg_speed, entropie, etc..] (flexible order)
        trialMobilityTrace is [mt1, mt2, ... mtn]
    """

    def __init__ (self):
        """ _dict_cluster contains the computed clusters in the form:
            [number : [[mt1, mt2,... mtn], mt_medioid]
        """
        self._dict_cluster=dict()


    @property
    def dict_clusters(self):
        return self._dict_cluster
    @dict_clusters.setter
    def dict_clusters(self,dict_cluster):
        self._dict_cluster = dict_cluster

    def __repr__ (self):
        strResult = "index : density of cluster, lat, long medioid \n"

        for keys in self.dict_clusters:
            aux_list=self.dict_clusters[keys]
            strResult +="{0}: {1},{2} \n".format(keys,len(aux_list[0]),aux_list[1])
        return strResult
    """
    @property
    def clusters(self):
        return self._dict_cluster
    """
    def doCluster(self):
        raise NotImplementedError( "Should have implemented this" )


