from mobilitytrace import MobilityTrace
import numpy
import logging
import sys
from datetime import time
import numpy
from sklearn.preprocessing import normalize
import pickle
from operator import itemgetter
import glob
import csv
import math

class Day (object):
    MON = 0
    TUE = 1
    WED = 2
    THU = 3
    FRI = 4
    SAT = 5
    SUN = 6
    WEEKDAYS = 7
    WEEKENDS = 8
    ALL = 9

class Mmc(object):

    def __init__ (self,cluster,trailMobilityTrace,userid,
            order = 1,
            daysArray=[False,False,False,False,False,False,False,True,True,False],
            timeSlices = 4,
            radius = 0.1):
        """
            cluster: is the dictionary containning the groups in the form of inde:[set of mt, medioid]
            trailMobilityTrace: is the list of mobility traces objects
            order: is the number of points of interest (poi) to remember in the model
            daysArray: is an boolean array of 10 positions where 0 is monday to 6 sunday, 7 is weekdays,
            8 is weekends and 9 is all days  [False,False,False,False,False,False,False,True,True,False]
            timeSlices: is the number of time windows in a given day e.g. timeSlices=2 is equals to two
            timelabels, is a dictionary containig the begin-end time of a windows time
            radius, in Km is the distance threshold to decide if a point is within the POI or not
        """
        self._pois = cluster.dict_clusters
        self._trailMobilityTrace = trailMobilityTrace
        self._user = userid
        self._order = order
        """ there are 3 cases: we can check only days from monday to sunday,
            or we can check weekdays and/or weekends
            or we can select all (no difference from any days)
        """
        self._daysArrayBoolean = daysArray
        self._daysArrayEnum = [Day.MON,Day.TUE,Day.WED,Day.THU,Day.FRI,Day.SAT,Day.SUN,Day.        WEEKDAYS,Day.WEEKENDS,Day.ALL]
        self._daysArrayEnumString = ["MON","TUE","WED","THU","FRI","SAT","SUN","WD","WE","ALL"]

        self._timeSlices = timeSlices
        self._radius = radius
        """ computed attributes  """
        # build in a dict the time windows without taking into account the days
        self._timeLabels = self.buildTimeLabels()
        self._spatioTemporalLabels = self.buildSpatioTemporalLabels()
        """  self._spatioTemporalLabels = []: List of tuples having
            [label_1, label_2, label_3, ... , label_n]
            where label_x is a list  [POI_a, POI_b, ...  ,POI_n, Day, timeWindows_begin, timeWindows_end]
            the number of pois depends on the order of model by default there should be just one poi in
            the list [POI_a, Day, timeWindows_begin, timeWindows_end]
        """
        self._spatialLabelRaw = []
        """ self._spatialLabelRaw: only has list of trace  in the form [[index,poi_label],
            [index,poi_label], ... [index,poi_label]] used for the pre-label and trajectory
            extraction
        """
        self._spatiaTemporallLabeledTrailmt=[]
        """self._spatiaTemporallLabeledTrailmt: This list contains the spatio (poi) and temporal (time
            windows) labels to compute the transition matrix.
        """
        size = len(self._spatioTemporalLabels)
        self._countMatrix = numpy.zeros(shape=(size,size))   #Count matrix
        self._transitionMatrix = numpy.zeros(shape=(size,size))   #Transition matrix
        self._stationaryVector = -1
        self._cumulatedStationary = dict()
        """_cumulatedStationaryVector: has the stationary value for each POI.
            It dosent take into account time windows only the proporion of time
            spent in a given POI. We computed it to measure distance between mmcs
        """
        #These dictionaries stock the trajectories from one poi A to another poi B
        #in the form of [POI_origin,POI_destination]: [[mt_1, mt_2, ... mt_n],[mt_1',mt_2', .., mt_n']]
        self._dict_trajectory = dict() #trajectories are represented by spatial labels: 0, 1, 2 ..
        self._dict_mt_trajectory = dict() ##trajectories are represented by mobility traces

############################################
#          Overwrite methods               #
############################################

    def equals(self, other):

        a = isinstance(other, self.__class__)
        c = (self._daysArrayBoolean == other._daysArrayBoolean)
        b = (other.__dict ==  self.__dict__)
        d = (self._order ==  other._order)
        e = (self._timeSlices ==  other._timeSlices)

        if ( a & b &  c & d & e ):
            return True
        else:
            return False

    def __eq__(self, other):
        """
            This method overwrite the "==" simbol. In our case the equality is acomplished
            when the model is formed taking into account the same days, order as well as the
            instance of the class (Mmc)
        """
        a = isinstance(other, self.__class__)
        c = (self._daysArrayBoolean == other._daysArrayBoolean)
        d = (self._order ==  other._order)
        e = (self._timeSlices ==  other._timeSlices)

        if ( a &  c & d & e ):
            return True
        else:
            return False

    def __ne__(self, other):
        return not __eq__(self, other)

############################################
#       Method                             #
############################################

    def buildTimeLabels (self):
        """ Returns a dict having as key the number of the time windows and as value a list with the
        begin and end time [timeWindows : [begin time, end time]]
        """
        HOURDAY = 24
        windowsSize = HOURDAY/self._timeSlices
        j = 0
        index = 0
        dict_timeLabels = dict()
        size = len(self._daysArrayBoolean)

        while (j < size):
            if ( self._daysArrayBoolean[j] ):
                i = 0
                while (i < self._timeSlices):
                    t_begin = time(i*windowsSize)
                    t_end = 0
                    if (i+1)*windowsSize == 24:
                        t_end = time.max
                    else:
                        t_end = time((i+1)*windowsSize)
                    dict_timeLabels[index] =  [self._daysArrayEnum[j], t_begin, t_end]
                    index += 1
                    i += 1
            j += 1
        return dict_timeLabels
    # buildTimeLabels


    def buildSpatioTemporalLabels (self):
        """
        Builds the spatio (taking into account the order of the model) and temporal labels
        """
        aux_spatioTemporalLabels = []
        #get pois labels
        poi_labels = self._pois.keys()##.getClusters
        #make the combination of places
        #we only take into account the two first orders (n=1 n=2)
        size = len(poi_labels)
        if (self._order == 2 ):
            i = 0
            while ( i < size ):
                j = 0
                while ( j < size ):
                    if (i != j):
                        k = 0
                        while ( k< len(self._timeLabels)):
                            aux_tuple = [poi_labels[i],poi_labels[j],\
                                    aux_timeLabels[k][0],self._timeLabels[k][1],self._timeLabels[k][2]]
                            self._spatioTemporalLabels.append(aux_tuple)
                            k += 1
                    j += 1
                i += 1
        else:# order == 1 or we force to be 1 (order < 3 state explosion)
            j = 0
            while ( j < size ):
                k = 0
                while ( k< len(self._timeLabels)):
                    aux_tuple = [poi_labels[j],self._timeLabels[k][0],\
                                self._timeLabels[k][1],self._timeLabels[k][2]]
                    aux_spatioTemporalLabels.append(aux_tuple)
                    k += 1
                j += 1

        return aux_spatioTemporalLabels
        #print "{0}".format(self._spatioTemporalLabels)
    #end buildSpatioTemporalLabels

########################################
#  Put every thing togheter
########################################
    def buildModel (self,local=True):
        logging.info("Building mmc...")
        logging.info("Creating labels")
        self.__prelabelMobilityTraces__()
        # Spatio temporal labeling of mobility traces
        logging.info("Labeling mobility traces")
        self.__labelMobilityTrace__()
        # computes transition matrix
        logging.info("Building transition matrix")
        self.computedTransitionMatrix(local)
        # computes  Stationary Vector
        logging.info("Computing stationary vector")
        self.computedStationaryVector()
        # buildCumulatedStationary
        logging.info("Calculating cumulated stationary vector")
        self.__buildCumulatedStationary__()
        logging.info("Model built successfully !")
    #end buildModel


########################################
# Spatial pre label of mobility traces #
########################################


    def __getSpatialLabel__ (self, mobilityTrace, dist_based = True):
        """ Take as input: a mobility trace and
            returns the id of the label     """
        id_poi = -1

        for key in self._pois:
            group =  self._pois[key]
            medioid = group[1]

            if (dist_based):
                if ( mobilityTrace.distance(medioid) <= self._radius ):
                    id_poi = key
                    break
            else:
                for mt in group[0]:
                    if (mobilityTrace.cellid ==  mt.cellid):
                        id_poi = key

        return id_poi
    #end def


    def __extractTrajectories__(self):
        dict_trajectory = dict()
        dict_mt_trajectory = dict()

        size = len(self._spatialLabelRaw)
        i = 0
        begin_label = -2
        end_label = -2
        aux_trajectory = [] #labels
        aux_mt_trajectory = [] #mobility traces
        while (i < size-1 ):
            label = self._spatialLabelRaw[i][1]
            label_1 = self._spatialLabelRaw[i+1][1]
            #Detecting the begining of a trajectory
            if ( (label > -1) & (label != label_1) ):
                begin_label = label
                begin_trajectory = self._trailMobilityTrace[i]
                j = i+1
                #looking for the end of the trajectory
                while (j < size -1):
                    label = self._spatialLabelRaw[j][1]
                    aux_label = self._spatialLabelRaw[j]
                    if (label > -1):
                        end_label = label
                        end_trajectory = self._trailMobilityTrace[j]
                        i = j + 1
                        break
                    else:
                        aux_trajectory.append(aux_label)
                        aux_mt_trajectory.append(self._trailMobilityTrace[j])
                    j += 1
                #end while (j < size -1)
            if ( int(begin_label)!=-2 & int(end_label)!=-2 & len(aux_trajectory)>0):
                aux_tuple = (begin_label,end_label)
                key_tuple = (begin_label,end_label)
                #key_tuple = (begin_trajectory,end_trajectory)
                #insert into the dictionary
                if (aux_tuple in dict_trajectory):
                    dict_trajectory[aux_tuple].append(aux_trajectory)
                    dict_mt_trajectory[key_tuple].append(aux_mt_trajectory)
                elif ( int(begin_label)!=-2 & int(end_label)!=-2):
                    dict_trajectory[aux_tuple]=[aux_trajectory]
                    dict_mt_trajectory[key_tuple] = [aux_mt_trajectory]

                #reinitialize
                begin_label = -2
                end_label = -2
                aux_trajectory = []
                aux_mt_trajectory = []
            i += 1
        self._dict_mt_trajectory = dict_mt_trajectory

        return dict_trajectory
    #end __getTrajectories__

    def printTrajectories(self):
        """
        Method to print the set of trajectories in the format:
        (x,y) where x is the origin and y is the destination
        trajectory, cell id. For instance:
        (5, 4)
        1,208-10-49205-10741
        1,208-10-1901-10216
        2,208-10-49221-47081
        2,208-10-1901-10216
        2,208-10-1901-10216
        3,208-10-49205-10741
        3,208-10-49205-10742
        """
        print "Trajectories:\n"
        trajectory = 0
        if len(self._dict_mt_trajectory) != 0:
            for key in self._dict_mt_trajectory:
                print key
                #gat the set of trajectories
                for t in self._dict_mt_trajectory[key]:
                    #print mobility traces of each trajecotry
                    trajectory += 1
                    for mt in t:
                        print "{0},{1}".format(trajectory,mt.cellid)
        else:
            print "Empty trajectories"
        #print self._dict_mt_trajectory
    #end printTrajectories


    def printTrajectoriesToMap(self):
        """
        Method to print the set of trajectories in the format:
        id_trajectory,cell_id,latitude,longitude
        1,208-10-49221-46688,49.408316,3.405161
        1,208-10-49205-10741,49.370438,3.493799
        1,208-10-49205-10741,49.370438,3.493799
        1,208-10-49221-46688,49.408316,3.405161
        2,208-10-49221-46688,49.408316,3.405161
        2,208-10-49205-10741,49.370438,3.493799
        2,208-10-49205-10742,49.370111,3.407806
        2,208-10-49221-46688,49.408316,3.405161
        """
        print "Trajectories:\n"
        trajectory = 0
        print "id_trajectory,cell_id,latitude,longitude,icon"
        if len(self._dict_mt_trajectory) != 0:
            for key in self._dict_mt_trajectory:
                index_origin = key[0]
                origin_mt = (self._pois[index_origin])[-1]
                index_destination = key[1]
                destination_mt = (self._pois[index_destination])[-1]

                #Do not show loops
                if index_origin == index_destination:
                    #gat the set of trajectories
                    for t in self._dict_mt_trajectory[key]:
                        #print mobility traces of each trajecotry
                        trajectory += 1
                        print "{0},{1},{2},{3},large_blue".format(
                                trajectory,
                                origin_mt.cellid,
                                origin_mt.latitude,
                                origin_mt.longitude)

                        #print mobility traces of the trajectory
                        for mt in t:
                            print "{0},{1},{2},{3},small_red".format(trajectory,
                                mt.cellid,
                                mt.latitude,
                                mt.longitude)

                        print "{0},{1},{2},{3},large_blue".format(trajectory,
                                destination_mt.cellid,
                                destination_mt.latitude,
                                destination_mt.longitude)
        else:
            print "Empty trajectories"
        #print self._dict_mt_trajectory
    #end printTrajectoriesToMap

    def __prelabelMobilityTraces__(self):
        """ loop over all the trail of mobility trace to label the pois  """
        spatialLabels = []

        if (len(self._trailMobilityTrace)>1):
            i = 0
            #generate spatial labels
            for mt in self._trailMobilityTrace:
                index = self.__getSpatialLabel__(mt,False)
                spatialLabels.append([i,index])
                i += 1
            #print "labels {0}: ".format(spatialLabels)
            self._spatialLabelRaw = spatialLabels
            #extracts trajectories from mobility traces
            #We have to verify how to get tracetories: _spatialLabelRaw contains index to delimit
#begin and end of a trajectory
            self._dict_trajectory =  self.__extractTrajectories__()
            #erase unknown
            self.__eraseUnknownTraces__()
            #print "{0}".format(self._spatialLabelRaw)
            self.__squash__()
            #print "{0}".format(self._spatialLabelRaw)
        else:
            logging.info("Trail of mobility traces dosen't have traces")
            #sys.exit(0)

    #end labelMobilityTraces

    def __eraseUnknownTraces__ (self):
        """ Erases all labels =  -1 i.e. unknown"""
        index = len(self._spatialLabelRaw)-1
        while (index >=  0):
            if (self._spatialLabelRaw[index][1] ==-1):
                del self._spatialLabelRaw[index]
            index -=1
    #eraseUnknownTraces

    def __squash__(self):
        """ Erases consecutive repeated labels """
        index_to_erase=[]
        size = len(self._spatialLabelRaw)-1
        index = 0
        while (index <  size):
            if (self._spatialLabelRaw[index][1] == self._spatialLabelRaw[index+1][1]):
                index_to_erase.append(index)
            index += 1

        index = len(index_to_erase)-1
        #print "squash: ".format()
        while (index >= 0):
            del self._spatialLabelRaw[index_to_erase[index]]
            index -= 1

    #end __squash__

###############################################
# Spatio temporal labeling of mobility traces #
##############################################

    def __labelMobilityTrace__ (self):
        """This method label each mobility trace in the same format as the spatiotemporal
           labels to use it for the transition matrix
        """

        labeledMobilityTraces = []
        size = len (self._spatialLabelRaw)
        if (self._order == 2):
            pass
        else: #if order <> 2 we consider order = 1
            i=0
            while (i < size):
                #gets the index of the mobility trace
                index = self._spatialLabelRaw[i][0]
                #get the mobility trace
                mt = self._trailMobilityTrace[index]
                aux_list = self.__getTimeWindows__(mt)
                if (len(aux_list)>0):
                   t_day = aux_list[0]
                   t_begin = aux_list[1]
                   t_end = aux_list[2]
                   aux_poi =self._spatialLabelRaw[i][1]
                   labeledMobilityTraces.append( [aux_poi, t_day, t_begin , t_end] )
                i += 1
        self._spatiaTemporallLabeledTrailmt=labeledMobilityTraces
    #end labelMobilityTrace


    def __getTimeWindows__(self, oMobilityTrace):
        oTime = (oMobilityTrace.timestamp).time()
        t_day = 0
        index = len(self._daysArrayBoolean)-1
        result = []
        #label day
        while (index >= 0):
            # test for day:all
            if ( (self._daysArrayBoolean[index] == True) and (index == 9) ):
                t_day = Day.ALL
                break
            # test for day:weekend & weekday
            if  (self._daysArrayBoolean[index] == True) and (index == 8 or index == 7):
                num_day = oMobilityTrace.timestamp.weekday()
                if ( self.__isWeekday__(num_day)):
                    t_day = Day.WEEKDAYS
                    break
                else:
                    t_day = Day.WEEKENDS
                    break
            # test for day:
            if ( (self._daysArrayBoolean[index] == True) and (index <= 6) ):
                t_day = oMobilityTrace.timestamp.weekday()
                break
            index -= 1


        for key in  self._timeLabels:
            aux_day = self._timeLabels[key][0]
            t_begin = self._timeLabels[key][1]
            t_end = self._timeLabels[key][2]
            if ( (t_begin <= oTime) & (oTime < t_end) & (aux_day==t_day) ):
                 result.append(t_day)
                 result.append(t_begin)
                 result.append(t_end)
        return result
    #end __getTimeWindows__

    def __isWeekday__(sefl,num_day):
        if (num_day>=0 & num_day<=4):
            return True
        else:
            return False

#########################################
#   Transition matrix  construction     #
#########################################

    def computedTransitionMatrix (self, local=True):
        """
            This method comutes the transition matrix of a local (individual user)
            or global (many users) mobility model
            @param: local set by default as true computes the transition model for
            an individual. Set a false means the global model in which case
            we do not erase establish POIs
        """
        #spatio temporal labeled list to built transition matrix
        size = len(self._spatiaTemporallLabeledTrailmt)-1
        i = 0
        #print "labels: {0}".format(self._spatioTemporalLabels)
        while (i < size):
            label_i = self._spatiaTemporallLabeledTrailmt[i]
            index_i = self._spatioTemporalLabels.index(label_i)
            label_j = self._spatiaTemporallLabeledTrailmt[i+1]
            index_j = self._spatioTemporalLabels.index(label_j)
            self._countMatrix[index_i][index_j] += 1
            i += 1

        if (local):
            index = len(self._spatioTemporalLabels)-1
            #Erases rows and columns without value
        
            print "before count: \n {0}".format(self._countMatrix)

            while (index >= 0):
                if (self._countMatrix[index].sum() == 0):
                    self._countMatrix = numpy.delete(self._countMatrix, (index), axis=0)
                    print "axis=0 count: \n {0}".format(self._countMatrix)
                    self._countMatrix = numpy.delete(self._countMatrix, (index), axis=1)
                    print "axis=1 count: \n {0}".format(self._countMatrix)
                    
                    del self._spatioTemporalLabels[index]
                index -= 1

            print "after count: \n {0}".format(self._countMatrix)

        #Normalize matrix (L1 norm)
        self._transitionMatrix = normalize(self._countMatrix, axis=1, norm='l1')
    #end computedTransitionMatrix

    def computedStationaryVector(self):
        #we assume that the transition matrix is alredy computred
        size = self._transitionMatrix.shape[0]
        if (size > 0):
            self._stationaryVector = numpy.ones(shape=(1,size))
            self._stationaryVector.fill(1.0/size)

            #print "stationary: {0}".format(self._stationaryVector)
            aux_stationaryVector = numpy.dot(self._stationaryVector , self._transitionMatrix)
            threshold = 0.001
            i = 0
            while ( not (self.__haveConverged__(aux_stationaryVector,threshold) ) ):
                self._stationaryVector = aux_stationaryVector
                aux_stationaryVector = numpy.dot(self._stationaryVector , self._transitionMatrix)
                i += 1
                #to avoid infinite loop
                if (i == 150):
                    break
        #print "stationary: {0}, sum: {1}, i:{2}".format(aux_stationaryVector,aux_stationaryVector[0].sum(),i)
        else:
            logging.info("No elements in transition matrix")
    #end computedStationaryVector

    def __haveConverged__ (self, new_stationaryVector, threshold):
        t = False
        if (len(self._stationaryVector) == len(new_stationaryVector)):

            aux_vector= numpy.subtract(self._stationaryVector,new_stationaryVector)
            aux_diff= (numpy.absolute(aux_vector)).sum()
            if (aux_diff <= threshold):
                t = True
        else:
            logging.info("Stationary vectors are not the same size")

        return t

    #end __haveConverged__

    def __buildCumulatedStationary__(self):
        #get pois labels
        poi_labels = self._pois.keys()##.getClusters
        size = len(self._spatioTemporalLabels)
        i = 0
        while (i < size):
            index = self._spatioTemporalLabels[i][0]
            if (index in self._cumulatedStationary):
                self._cumulatedStationary[index] += self._stationaryVector[0][i]
            else:
                self._cumulatedStationary[index] = self._stationaryVector[0][i]
            i += 1
#aqui
    #end __buildCumulatedStationary__

#################################################
#           __repr__ overwrite                  #
#################################################
    def __repr__ (self):
        strResult = "Model parameters \n"
        strResult += "================ \n"
        strResult += "Number of traces: {0} \n".format(len(self._trailMobilityTrace))
        strResult += "Order: {0} \n".format(self._order)
        strResult += "Selected days for the model: \n"
        strResult += "\t Weekdays: Monday: {0}, Tuesday: {1}, Wednesday: {2}, Thursday: {3}, Friday: {4} \n".format(self._daysArrayBoolean[0],self._daysArrayBoolean[1],self._daysArrayBoolean[2],self._daysArrayBoolean[3],self._daysArrayBoolean[4])
        strResult += "\t Weekends: Saturday: {0}, sunday: {1}\n".format(self._daysArrayBoolean[5],self._daysArrayBoolean[6])
        strResult += "\t Weeks: Weekdays: {0}, Weekends: {1}, all: {2} \n".format(self._daysArrayBoolean[7],self._daysArrayBoolean[8],self._daysArrayBoolean[9])
        strResult += "Time slices: {0} \n".format(self._timeSlices)
        strResult += "Time windows: id:[begin hour, end hour] \n"

        for keys in self._timeLabels:
            strResult += "\t {0}:{1} \n".format(keys,self._timeLabels[keys])

        strResult += "___________________________________________________ \n"
        strResult += "Model results \n"
        strResult += "================ \n"
        strResult += "POIs: \n"
        strResult += "index : density of cluster, lat, long medioid \n"

        for keys in self._pois:
            aux_list=self._pois[keys]
            strResult +="{0}: {1},{2} \n".format(keys,len(aux_list[0]),aux_list[1])


        strResult += "================ \n"
        strResult += "\n Transition matrix:  \n".format(self._transitionMatrix)
        strResult += "================ \n"
        strResult += "Stationary vector \n"

        size = len(self._spatioTemporalLabels)
        i = 0

        #strResult += "{0}:{1}".format(len(self._spatioTemporalLabels[i]),len(self._stationaryVector))
        while (i < size):
            strResult += "{0}:{1} \n ".format(self._spatioTemporalLabels[i],self._stationaryVector[0][i])
            i += 1

        strResult += "================ \n"
        strResult += "Cumulated Stationary vector \n"

        for keys in self._cumulatedStationary:
            strResult +="{0}: {1} \n".format(keys,self._cumulatedStationary[keys])


        return strResult
############################################
#               Properties                 #
############################################
    @property
    def getPoi(self):
        return self._pois

    @property
    def getTimeLabels(self):
        return self._timeLabels

    @property
    def trailMobilityTrace(self):
        return self._trailMobilityTrace
    @trailMobilityTrace.setter
    def trailMobilityTrace(self,trailMobilityTrace):
        self._trailMobilityTrace = trailMobilityTrace

    @property
    def stationaryVector(self):
        return self._stationaryVector


#############################################
#       Write and read mmc object           #
#############################################

    def export(self,filepath):
        _filepath = filepath + str(self._user) + ".mmc"
        pickle.dump(self, open(_filepath, "wb"))
    @classmethod
    def export_matrix(cls,matrix,filepath):
        _filepath = filepath  + ".dmt"
        pickle.dump(matrix, open(_filepath, "wb"))

    @classmethod
    def load(cls,filepath):
        return pickle.load(open(filepath, "rb"))


#############################################
#       Distance between mmc                #
#############################################
    def distance (self,oMmc,method="stationary",threshold=1):
        result = 0
        if  (method == "stationary"):
            result = self.stationaryDistance(oMmc)
        elif(method == "relative"):
            result = self.relativeDistance(oMmc,threshold)
        else:
            result = self.coverageRate(oMmc,threshold)
        return result
    #end distance
    def __stationaryMetric__(self, oMmc):
        """ this method computes the distance in Km between 2 mmc models,
            this metric uses the stationary vector to weight the distance
        """
        distance = 0

        for key in self._cumulatedStationary:
            group =  self._pois[key]
            medioid = group[1]
            aux_distance = 1000000

            #looks for the closest poi in the other model
            for okey in oMmc._pois:
                oGroup =  oMmc._pois[okey]
                oMedioid = oGroup[1]
                aux = medioid.distance(oMedioid)
                if (aux < aux_distance):
                    aux_distance = aux
            #add the distance weighted by the stationary value
            distance += aux_distance * self._cumulatedStationary[key]
            aux_distance = 1000000
        return distance

    def stationaryDistance(self, oMmc):
        stat_distance = (self.__stationaryMetric__(oMmc) + oMmc.__stationaryMetric__(self))/2
        if (stat_distance == None):
            stat_distance = 0
        return stat_distance
    # stationaryDistance

    def relativeDistance(self,oMmc, pThreshold=1):
        """This method measure the relativge distance based on importance of places
            the first POIs is twice iportant than the second and so on. The smaller the
            distance the more likely models are.
        """
        size = -1
        threshold = pThreshold #Km
        score = 0
        importance = 10
        """
        if ( len(self._pois) >= len(oMmc._pois) ):
            size = len(oMmc._pois)
        else:
            size = len(self._pois)
        """
        values = self._cumulatedStationary.items()
        oValues = oMmc._cumulatedStationary.items()

        sorted(values, key=itemgetter(1), reverse=True)
        sorted(oValues,key=itemgetter(1), reverse=True)

        if ( len(values) >= len(oValues) ):
            size = len(oValues)
        else:
            size = len(values)


        #print "{0} : {1} \ {2}".format (len(oMmc._pois),len(self._pois),size)
        i = 0
        while (i < size):
            index = (values[i])[0]
            medioid = (self._pois[index])[1]
            """
            print ("{0}".format(i))
            print ("{0}".format(oValues[i]))

            if ((size == 4) & (i == 0)):
                print "{0}\n".format(values)
                print "{0}\n".format(oValues)
                print "{0}\n".format(self._cumulatedStationary)
                print "{0}\n".format(oMmc._cumulatedStationary)
                print "{0}".format(oMmc)
            """
            oIndex = (oValues[i])[0]
            oMedioid = (oMmc._pois[oIndex])[1]
            distance = medioid.distance(oMedioid)

            if (distance < threshold):
                score += importance

            importance /= 2
            i += 1

        distance = 300000

        if (score > 0):
            distance = 1.0/score

        return distance
    #end relativeDistance

    def coverageRate (self, oMmc, threshold = 1):
        coverage_rate = 0
        coverage = 0

        for key in self._cumulatedStationary:
            group =  self._pois[key]
            medioid = group[1]
            aux_distance = 1000000
            #looks for the closest poi in the other model
            for okey in oMmc._pois:
                oGroup =  oMmc._pois[okey]
                oMedioid = oGroup[1]
                aux = medioid.distance(oMedioid)

                if (aux < aux_distance):
                    aux_distance = aux

            #add the distance weighted by the stationary value
            if ( aux_distance <= threshold ):
                coverage += 1

            aux_distance = 1000000
        if (coverage > 0):
            coverage_rate = 1.0/coverage

        return coverage_rate
    #end def coverageRate

#############################################
#       Inner properties in mmc             #
#############################################

    def shannonEntropy(self):
        """
            This method computes the Shannon entropy of the model
            It takes into account the acumulated stationary vector
        """
        entropy = 0
        aStationaryVec = list(self._cumulatedStationary.values())
        #print "stationary: {0}".format(aStationaryVec)
        for item in aStationaryVec:
            if (item != 0):
                entropy += item*math.log(item,2)

        return entropy*-1
    #end shannonEntropy

    def shannonEntropyTime(self):
        """
            This method computes the Shannon entropy of the model
            It takes into account the stationary vector. Thus it takes
            into account the time windows
        """
        entropy = 0
        aStationaryVec = list(self._stationaryVector.values())
        for item in aStationaryVec:
            if (item != 0):
                entropy += item*math.log(item,2)

        return entropy*-1
    #end shannonEntropy

    def predictability(self):
        """
            This method computes the predictability, which measures
            the maximal probability we can achiave making a location
            forecasting using the mmc
        """
        rows = self._transitionMatrix.shape[0]
        row = 0
        predictability = 0

        while row < rows:
            maxValueRow = numpy.amax(self._transitionMatrix[row,])
            #print "rows:",rows
            #print self._stationaryVector.shape
            predictability +=  self._stationaryVector[0,row] * maxValueRow
            row += 1

        return predictability
    #end predictability(self):


#################################################
#           Distance matrices                   #
#################################################
    @classmethod
    def buildDistanceMatrix (cls, pathToListOfModels="models/*.mmc",distance="stationary",threshold=1):
        listModels = glob.glob(pathToListOfModels)
        size = len (listModels)
        distance_matrix = numpy.zeros(shape=(size,size))

        i = 0
        j = 0
        #print ("size: {0}").format(size)
        while (i < size ):
            j = i
            mmc_i = Mmc.load(listModels[i])
            while (j < size):
                if (i != j):
                    mmc_j = Mmc.load(listModels[j])
                    aux = mmc_i.distance(mmc_j, distance, threshold)
                    #print "aux : {0}".format(aux)
                    distance_matrix[i][j] = aux
                    distance_matrix[j][i] = aux
                j += 1
            i += 1

        return distance_matrix

    #end buildDistanceMatrix


#################################################
#               Merge mmc                       #
#################################################
    @classmethod
    def mergeMmc(cls,aMmc, bMmc, threshold ):
        if (aMmc.equals(bMmc)):
            return aMmc
        elif (self != mmc):
            return None
        else:
            #map pois of both models common pois
            close_poiss_dict = Mmc.mapClosePois(aMmc,bMmc,0.1)
            #merge transition matrix
            newMatrixTransition,newMatrixLabels = mergeTransMatrix(aMmc,bMmc,close_poiss_dict)
            #merge trajectories matrix
            #Test with new QTS will be on-line
            #/!\ new_trajectory = newTrajectories(aMmc,bMmc,close_poiss_dict)
            #Map clusters
            #change poi names
            new_pois = newPois(aMmc,bMmc,close_pois_dict)
            #(re)compute stationary vector
            #(re)compute cumulated stationary vector
    #end mergeMmc


    @classmethod
    def newPois(cls, aMmc,bMmc,close_pois_dict,addMobilityTraces=True):
        """
        """
        aPois = aMmc._pois
        bPois = bMmc._pois

        #Mapping pois from model A
        for aKey in aPois.keys():
            aux_key = (str(aKey)+'a')
            aPois[aux_key]=  aPois.pop(aKey)

        #Mapping pois from model B
        aux_dict=dict()
        for key, val in close_pois_dict.items():
            if (len(val) > 0):
                aux_dict[key]=val

        for bKey in bPois.keys():
            aux_label = str(bKey)+'b'
            if ( aux_label in close_pois_dict.keys() ):
                bPois[aux_label]=bPois.pop(bKey)
            else:
                for auxKey, val in aux_dict.items():
                    #val is a list
                    for item in val:
                        if (item == bKey):
                            bPois[auxKey]=bPois.pop(bKey)

        new_pois = dict()
        if (addMobilityTraces):
            #merging Poi dictionaries

            for aKey in aPois.keys():
                    new_pois[aKey] = aPois[aKey]

            for bKey in bPois.keys():
                if (bKey in new_pois):
                    (new_pois[bKey])[0]  = set ( list((new_pois[bKey])[0])+ list((bPois[bKey])[0]))
                else:
                    new_pois[bKey] = bPois[bKey]

            #update medioid
            for mkey in aux_dict.keys():
                trialMobilityTraces=list((new_pois[mkey])[0])
                medioid = MobilityTrace.computeMediod(trialMobilityTraces)
                (new_pois[mkey])[1] = medioida
        else:
            #merge dictionaries but only medioids (discarting mts)
            for aKey in aPois.keys():
                aux_value = list()
                aux_value.append((aPois[aKey])[1])
                new_pois[aKey] = aux_value

            for bKey in bPois.keys():
                if (bKey in new_pois):
                    new_pois[bKey].append( (bPois[bKey])[1]  )#append medioid
                else:
                    aux_value = list()
                    aux_value.append((bPois[bKey])[1])
                    new_pois[bKey] = aux_value


        return new_pois
#end newPois

    @classmethod
    def newTrajectories(cls, aMmc,bMmc,close_pois_dict):

        aTrajectory = aMmc._dict_trajectory
        bTrajectory = bMmc._dict_trajectory

        #Mapping trajectories from model A (changing names )
        for aKey in aTrajectory:
            aux_key = (str(aKey[0])+'a',str(bKey[1])+'a')
            aTrajectory[aux_key]= aTrajectory.pop(aKey)

        #Maps only labels to need to be renamed
        aux_dict=dict()
        for key, val in close_pois_dict.items():
            if (len(val) > 0):
                aux_dict[key] = val

        #Mapping trajectories from model B
        for bKey in bTrajectory:
            beginLabel = str(bKey[0])+'b'
            endLabel = str(bKey[1])+'b'

            if ((beginLabel in close_pois_dict) & (endLabel in close_pois_dict)):
                pass
            #we have the beginLabel in close_pois_dictve, we look for the endLabel
            elif (beginLabel in close_pois_dict):
                for item in aux_dict:
                    aux_values = aux_dict[item]
                    for val in aux_values:
                        if (val==bKey[1]):
                            endLabel = item
                            break
            #we have the endLabel we look for the beginLabel
            elif endLabel in aux_dict:
                aux_values = aux_dict[item]
                for val in aux_values:
                    if (val==bKey[0]):
                        beginLabel = item
                        break

            aux_key = (beginLabel,endLabel)
            bTrajectory[aux_key]= bTrajectory.pop(aKey)

            new_trajectory=dict()

            #merging trajectories

            for aKey in aTrajectory:
                new_trajectory[aKey]=(aTrajectory[aKey])
            #TODO: check the append of the trajectories
            for bKey in bTrajectory:
                if (bKey in  new_trajectory):
                    new_trajectory[bKey].append(aTrajectory[bKey])
                else:
                    new_trajectory[bKey]=(bTrajectory[aKey])


        return new_trajectory
    # newTrajectories


    @classmethod
    def mergeTransMatrix(cls,aMmc,bMmc,close_poiss_dict):
        """
            @param aMmc: is the base mobility mmodel
            @param bMmc: is the mobility model to add
            @param close_poiss_dict: is a dictionary that maps pois of both aMmc and bMmc
            returns: the new transition matrix and the new labels of the matrix
        """
        aMatrixLabels=aMmc._spatioTemporalLabels
        bMatrixLabels=bMmc._spatioTemporalLabels

        # Normalize names of labels in models A and B to merge
        for key in close_poiss_dict.keys():
            size = len(str(key))
            model = (str(key))[size-1:]
            poi = ((str(key))[:size-1])


            if (model == 'b'):
                # update labels in model b
                for label in bMatrixLabels:
                    if ( str(poi) == str(label[0]) ):
                        label[0] = key
            elif (model ==  'a'):
                # update labels in model a
                labelSize = len(close_poiss_dict[key])
                # update all pois that belong to the same poi in the A model
                if ( labelSize > 0 ):
                    labelsToChange  = close_poiss_dict[key]

                    for item in labelsToChange:
                        for label in bMatrixLabels:
                            if (str(item) == str(label[0])):
                                label[0] = key
                # update labels in model a
                for label in aMatrixLabels:
                    if ( str(poi) == str(label[0])):
                        label[0] = key
            else:
                logging.error("Unknown model !")

        #join labels

        newMatrixLabels = list()

        for aItem in aMatrixLabels:
            newMatrixLabels.append(aItem)

        for bItem in bMatrixLabels:
            add = True
            for aItem in aMatrixLabels:
                if (aItem == bItem):
                    add = False
                    break
            if (add):
                newMatrixLabels.append(bItem)

        #creates the common transition matrix

        size = len(newMatrixLabels)
        newMatrixTransition = numpy.zeros(shape=(size,size))

        for xItem in aMatrixLabels:
            for yItem in aMatrixLabels:
                ax = aMatrixLabels.index(xItem)
                ay = aMatrixLabels.index(yItem)
                nx = newMatrixLabels.index(xItem)
                ny = newMatrixLabels.index(yItem)
                newMatrixTransition[nx][ny] = aMmc._transitionMatrix [ax][ay]

        for xItem in bMatrixLabels:
            for yItem in bMatrixLabels:
                bx = bMatrixLabels.index(xItem)
                by = bMatrixLabels.index(yItem)
                nx = newMatrixLabels.index(xItem)
                ny = newMatrixLabels.index(yItem)
                newMatrixTransition[nx][ny] = bMmc._transitionMatrix[bx][by]

        newMatrixTransition = normalize(newMatrixTransition, axis=1, norm='l1')

        return (newMatrixTransition,newMatrixLabels)

    #end mergeTransMatrix


    @classmethod
    def mapClosePois(cls,aMmc,bMmc,threshold = 0.1):
        """
            Maps the pois wich are within a distance threshold
            aMmc: the mobility base model
            bMmc: the mobility model to merge in the base model
            threshold: the distance threshold in Km, from one poi centroid to
                another, to consider two pois mergable
        """
        #aMmc looks for colsest pois in bMmc based on stationary values
        aPois = list(aMmc._pois.keys())
        aStationaryVec = list(aMmc._cumulatedStationary.values())
        bPois = list(bMmc._pois.keys())
        #aStationaryVec = aMmc._cumulatedStationary

        #sorts the pois of the firs model
        sortStationary, sortPois = zip(*sorted(zip(aStationaryVec,aPois)))
        close_poiss_dict = dict()
        mapPois = list () #auxiliar list of pois in mmc B  belonging to mmc A

        #for all pois in mmc A:
        for aPoi in sortPois:
            aMedioid = (aMmc._pois).get(aPoi)[1]
            #we look for those poi in mmc B close to poi A
            for bPoi in bPois:
                bMedioid = (bMmc._pois).get(bPoi)[1]

                if (aMedioid.distance(bMedioid)<=threshold):
                    mapPois.append(bPoi)
                    bPois.remove(bPoi)

            #add list of close pois
            close_poiss_dict[str(aPoi)+'a'] = list(mapPois)
            mapPois = list ()

        if (len(bPois) > 0):
            for bPoi in bPois:
                close_poiss_dict[str(bPoi)+'b'] = mapPois
                mapPois = list()


        return close_poiss_dict
    #end mapClosePois
#################################################
#               Export csv to map
################################################

    def exportToMap(self, path):
        """
        This method generate a csv to draw POI in a map using R plotmap.R script
        """
        with open(path, 'wb') as csvfile:
            oWriter = csv.writer(csvfile, delimiter=';', quotechar=' ', quoting=csv.QUOTE_ALL)
            oWriter.writerow(["poi","lat","lon"])

            for poi in self._pois:
                medioid=(self._pois[poi])[1]
                oWriter.writerow([poi,medioid.latitude,+medioid.longitude])


    #end exportToMap

    def exportMatrixod(self,path):
        """
        Exports the count matrix to a csv file
        """
        numpy.set_printoptions(precision=3)
        numpy.set_printoptions(suppress=True)

        with open(path, 'wb') as csvfile:
            oWriter = csv.writer(csvfile, delimiter=';', quotechar=' ', quoting=csv.QUOTE_ALL)
            #size = self._countMatrix.shape
            for rows in self._countMatrix:
                oWriter.writerow(rows)

    #end exportMatrixod():

    def exportGraphviz(self,path):
        """
        Exports the transition matrix to a dot file
        """
        numpy.set_printoptions(precision=3)
        numpy.set_printoptions(suppress=True)

        #build
        subgraph = dict()

        for  i in self._timeLabels:
            subgraph_key = 'subgraph cluster{0}'.format(i)
            day_label = self._daysArrayEnumString[ self.__getDayStringIndex__(i) ]

            #get the time_windows label= [day,begin hour, end hour]
            time_windows = self._timeLabels[i]
            begin_hour = time_windows[1]
            end_hour = time_windows[2]
            subgraph_name = '{0}_{1}_{2}'.format(day_label,begin_hour,end_hour)

            nodes_name = list()
            #get the spatio temporal windows label= [place(cluster),day,begin hour, end hour]
            #item is the spatio temporal windows label
            for item in self._spatioTemporalLabels:
                if time_windows[0] == item[1] \
                        and time_windows[1] == item[2] \
                        and time_windows[2] == item[3]:
                    #draw
                        name="{0}_{1}_{2}_{3}".format(item[0],
                                self._daysArrayEnumString[item[1]],
                                item[2].hour,
                                item[3].hour)
                        nodes_name.append(name)
            if len(nodes_name) > 0:
                subgraph[subgraph_key] = [subgraph_name,nodes_name]

        with open(path, 'wb') as f:
            f.write('digraph  proccess{ \n')
            f.write('\t node [fontsize=10,  style=filled, fillcolor=grey89]; \n')

            #Draw clusters
            for key,item in subgraph.items():
                subgraph_name = item[0]
                nodes_names = item[1]
                f.write('\t'+key+'{\n')
                f.write('\t\tlabel = "'+subgraph_name+'";\n')

                for node in nodes_names:
                    f.write('\t\t"'+node+'";\n')

                f.write('\t}\n')
            #Draw transitions
            (row,col) = self._transitionMatrix.shape
            i = 0
            while i < row:
                item = self._spatioTemporalLabels[i]
                i_name="{0}_{1}_{2}_{3}".format(item[0],
                    self._daysArrayEnumString[item[1]],
                    item[2].hour,
                    item[3].hour)
                j = 0
                while j < col:
                    value = self._transitionMatrix[i,j]
                    if (i != j ) and (float(value) != 0):
                        item = self._spatioTemporalLabels[j]
                        j_name="{0}_{1}_{2}_{3}".format(item[0],
                            self._daysArrayEnumString[item[1]],
                            item[2].hour,
                            item[3].hour)
                        f.write('\t"{0}" -> "{1}" [label="{2:.2f}"];\n'.format(i_name,j_name,value))
                    j += 1
                #end (while j < col)
                i += 1
            #end (while i < row)
            f.write('\t}\n')
            f.close()

    #def exportGraphviz


    def exportGraphviz_heatmap(self,path,value_vector):
        """
        Exports the transition matrix to a dot file using the value_vector
        to draw a heatmap of the nodes based on the value_vector to compute
        the heat color of each node form red (hottest) to blue (coldest)
        """
        if (len(value_vector) == len(self._spatioTemporalLabels)):
            numpy.set_printoptions(precision=3)
            numpy.set_printoptions(suppress=True)

            max_value = max(value_vector)
            #min_value = min(value_vector)

            #build
            subgraph = dict()

            for  i in self._timeLabels:
                subgraph_key = 'subgraph cluster{0}'.format(i)
                day_label = self._daysArrayEnumString[ self.__getDayStringIndex__(i) ]

                #get the time_windows label= [day,begin hour, end hour]
                time_windows = self._timeLabels[i]
                begin_hour = time_windows[1]
                end_hour = time_windows[2]
                subgraph_name = '{0}_{1}_{2}'.format(day_label,begin_hour,end_hour)

                nodes_name = list()
                #get the spatio temporal windows label= [place(cluster),day,begin hour, end hour]
                #item is the spatio temporal windows label
                size = len (self._spatioTemporalLabels)
                j = 0
                #for item in self._spatioTemporalLabels:
                while j<size:
                    item = self._spatioTemporalLabels[j]
                    if time_windows[0] == item[1] \
                            and time_windows[1] == item[2] \
                            and time_windows[2] == item[3]:
                        #draw
                        name="{0}_{1}_{2}_{3}".format(item[0],
                                    self._daysArrayEnumString[item[1]],
                                    item[2].hour,
                                    item[3].hour)
                        color_value = self.__rgb__(max_value,value_vector[j])
                        nodes_name.append([name,color_value])
                    j += 1
                if len(nodes_name) > 0:
                    subgraph[subgraph_key] = [subgraph_name,nodes_name]
            with open(path, 'wb') as f:
                f.write('digraph  proccess{ \n')
                f.write('\t node [fontsize=10,  style=filled, colorscheme=spectral9]; \n')

                #Draw clusters
                for key,item in subgraph.items():
                    subgraph_name = item[0]
                    nodes_names = item[1]
                    f.write('\t'+key+'{\n')
                    f.write('\t\tlabel = "'+subgraph_name+'";\n')
                    index = 0
                    for node in nodes_names:

                        #rgb = self.__rgb__(max_value,value_vector[index])
                        f.write('\t\t"'+node[0]+'" [fillcolor={0}]  ;\n'.format(node[1]))
                        index += 1
                    f.write('\t}\n')
                #Draw transitions
                (row,col) = self._transitionMatrix.shape
                i = 0
                while i < row:
                    item = self._spatioTemporalLabels[i]
                    i_name="{0}_{1}_{2}_{3}".format(item[0],
                        self._daysArrayEnumString[item[1]],
                        item[2].hour,
                        item[3].hour)
                    j = 0
                    while j < col:
                        value = self._transitionMatrix[i,j]
                        if (i != j ) and (float(value) != 0):
                            item = self._spatioTemporalLabels[j]
                            j_name="{0}_{1}_{2}_{3}".format(item[0],
                                self._daysArrayEnumString[item[1]],
                                item[2].hour,
                                item[3].hour)
                            f.write('\t"{0}" -> "{1}" [label="{2:.2f}"];\n'.format(i_name,j_name,value))
                        j += 1
                    #end (while j < col)
                    i += 1
                #end (while i < row)
                f.write('\t}\n')
                f.close()

        else:
            print "WARNING value_vector is not of the same size of the nodes"

    #def exportGraphviz_heatmap


    def __getDayStringIndex__(self,day):
        """
        Return the correct index for the node label strinf array of the
        graphviz export method
        """
        if ( (self._daysArrayBoolean[9] == True)  ):
            return Day.ALL
        # test for day:weekend & weekday
        elif  (self._daysArrayBoolean[8] or self._daysArrayBoolean[7]):
            if ( self.__isWeekday__(day)):
                return Day.WEEKDAYS
            else:
                return Day.WEEKENDS
        else:
            return day
    #end __getDayLabel__



    def __rgb__(self, maximum, value):
        maximum = float(maximum)
        value - float(value)
        SCALE = float(9)
        MAX = SCALE + float(1)
        r = (float(value/maximum))
        rs = r*SCALE
        color = int( MAX - int(rs))
        #print "value:{0} r:{1} color:{2}".format(value,rs,color)
        if color>SCALE:
            color = int(SCALE)
        if color<1:
            color = int(1)
        return color
    #end rgb


#########################
#       TEST            #
#########################
if __name__ == "__main__":
    pass

