import copy
import math
import numpy
import csv
from datetime import datetime

class Trace(object):
    """ This class represents a mobility point represented by:
    - userid
    - timestamp
    - mcc
    - spend
    - latitude
    - longitude
    """


    def __init__ (self, userid, timestamp, mcc, spend, latitude, longitude):
        """ Constructor of the MobilityTrace class"""
        if ( type(timestamp) is datetime):
            self._timestamp = timestamp
        elif ( type(timestamp) is str):
            #self._timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            if (len(timestamp)>10):
               self._timestamp = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S')
            else:
               self._timestamp = datetime.strptime(timestamp, '%d/%m/%Y')
        else:
            self._timestamp = datetime.fromtimestamp(timestamp)

        self._userid = userid
	self._mcc = mcc
	self._spend = spend
        self._latitude = latitude
        self._longitude = longitude


#################################################
#		Properties			#
#################################################

    @property
    def timestamp(self):
        return self._timestamp
    @timestamp.setter
    def timestamp(self,timestamp):
        self._timestamp = timestamp

    @property
    def mcc(self):
        return self._mcc
    @mcc.setter
    def mcc(self, mcc):
        self._mcc = mcc

    @property
    def userid(self):
        return self._userid
    @userid.setter
    def userid(self,userid):
        self._userid = userid

    @property
    def latitude(self):
        return self._latitude
    @latitude.setter
    def latitude(self,latitude):
        self._latitude = latitude

    @property
    def longitude(self):
        return self._longitude
    @longitude.setter
    def longitude(self,longitude):
        self._longitude = longitude

    @property
    def spend(self):
        return self._spend
    @userid.setter
    def spend(self,spend):
        self._spend = spend

#################################################
#################################################
#		Methods				#
#################################################

    def __repr__ (self):
	    return "'{0}' '{1}' '{2}' '{3}' '{4}' '{5}'".format(
                   self._userid,
                   self.timestamp,
                   self._mcc,
                   self._spend,
                   self.latitude,
                   self.longitude)

    def distance(self, mobilityTrace):
        #print mobilityTrace
        #print self.latitude

        """ Computes the euclidean distance  in Km"""
        PIov180 = 0.017453292519943295
        dLat = (float(mobilityTrace.latitude) - float(self.latitude)) * PIov180
        dLon = (float(mobilityTrace.longitude) - float(self.longitude)) * PIov180
        a = math.sin(dLat/2)**2 + (math.sin(dLon/2)**2) * math.cos(float(self.latitude)*PIov180) * math.cos(float(mobilityTrace.latitude)*PIov180)
        divisor = math.sqrt(1-a)
        if (divisor == 0):
            divisor = 1

        return round(12742 * math.atan(math.sqrt(a)/ divisor), 3)


    def time_distance(self, trace):
        """ Computes de difference between the timestamp of the eventement passed in arguments and the timestamp of the object that calls the method in seconds"""
        deltaTime = trace.timestamp - self.timestamp
        return deltaTime.total_seconds()


#################################################
#		Class methods			#
#################################################


    @classmethod
    def compute_frequency_update(cls, trialTraces):
        """ Computes the averge time in seconds of the position update (in seconds)
        and the average distance between two updates (in Km) """
        size = len(trialTraces)
        frequency_time = 0
        frequency_distance = 0
        i = 0
        while (i < size - 1):
            mt1 = trialTraces[i]
            mt2 = trialTraces[i+1]
            frequency_time += mt1.time_distance(mt2)
            i += 1

        if (i != 0):
            frequency_time=frequency_time/i

        return frequency_time 

    @classmethod
    def compute_cumulated_spent(cls, trailTraces):
        """
            This method returns a float representing the cumulated distance of a trail of
            mobility trace
            @param: The trail of mobility traces
            @result: the acumulated distance
        """
        spent = 0
        size = len(trailTraces)
        if (size > 1):
            i = 0
            while (i < size):
                spent += float( trailTraces[i]._spend )
                i += 1

            avg_spent = float(spent/len(trailTraces))
        return (spent,avg_spent)

    @classmethod
    def compute_cumulated_distance(cls, trailTraces):
        """
            This method returns a float representing the cumulated distance of a trail of
            mobility trace
            @param: The trail of mobility traces
            @result: the acumulated distance
        """
        distance = 0
        size = len(trailTraces)
        if (size > 1):
            i = 0
            while (i < size-1):
                distance += float( trailTraces[i].distance(trailTraces[i+1]))
                i += 1

            avg_distance = float(distance/len(trailTraces))
        return (distance,avg_distance)


    def squashTraces (cls, trialTraces):
        """
            Erases all consecutive mobility traces returning only position transitions
        """
        new_list = list (trialTraces)
        i = len(new_list)
        while (i > 0):
            mtn = new_list[i]
            mtn_1 = new_list[i-1]
            if (mtn.mcc == mtn_1.mcc):
                new_list.remove(mtn)
            i-=1

        return  new_list

    @classmethod
    def compute_acumulated_time(cls, trailTraces):
        """
            This method returns a float representing the cumulated time of a trail of
            mobility trace in seconds
            @param: The trail of  traces
            @result: the acumulated time in seconds
        """
        distance = 0
        size = len(trailMobilityTraces)
        if (size > 1):
            i = 0
            while (i < size-1):
                distance += float( trailTraces[i].time_distance(trailTraces[i+1]))
                i += 1
        return (distance)

#########################
#       TEST            #
#########################
if __name__ == "__main__":
   pass
