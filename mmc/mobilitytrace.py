import copy
import math
import numpy
import csv
from datetime import datetime

class MobilityTrace(object):
    """ This class represents a mobility point represented by:
    - imsi
    - timestamp
    - cell id
    - latitude
    - longitude
    - event_type
    - mcc
    - mnc
    - lac
    - radius
    - eps e.g. (speed or direction ...) """


    def __init__ (self, timestamp, cellid, latitude, longitude, eventType,
            mcc=None, mnc=None, lac=None,radius=None, eps=None, userid=None):
        """ Constructor of the MobilityTrace class"""
        if ( type(timestamp) is datetime):
            self._timestamp = timestamp
        elif ( type(timestamp) is str):
            self._timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        else:
            self._timestamp = datetime.fromtimestamp(timestamp)

        self._cellid = cellid
        self._latitude = latitude
        self._longitude = longitude
        self._eventType = eventType

        if mcc is None:
	    self._mcc = -1
        else:
	    self._mcc = mcc
        if mnc is None:
	    self._mnc = -1
        else:
	    self._mnc = mnc
        if lac is None:
            self._lac = -1
        else:
	    self._lac = lac
        if radius is None:
	    self._radius = -1
        else:
	    self._radius = radius
        if eps is None:
            self._eps = -1
        else:
            self._eps = eps
        if userid is None:
            self._userid = ""
        else:
            self._userid = userid

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
    def cellid(self):
        return self._cellid
    @cellid.setter
    def cellid(self,cellid):
        self._cellid = cellid

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
    def eventType(self):
        return self._eventType
    @eventType.setter
    def eventType(event_type):
        self._eventType = event_type

    @property
    def mcc(self):
        return self._mcc
    @mcc.setter
    def mcc(self, mcc):
        self._mcc = mcc

    @property
    def mnc(self):
        return self.mnc
    @mnc.setter
    def mnc(self,mnc):
        self._mnc = mnc

    @property
    def lac(self,lac):
        self._lac = lac
    @lac.setter
    def lac(self):
        return self._lac

    @property
    def radius(self):
        return self._radius
    @radius.setter
    def radius(self,radius):
        self._radius = radius

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self,eps):
        self._eps = eps

    @property
    def userid(self):
        return self._userid
    @userid.setter
    def userid(self,userid):
        self._userid = userid

#################################################
#		Methods				#
#################################################

    def __repr__ (self):
	    return "'{0}' '{1}' '{2}' '{3}' '{4}'".format(self.timestamp,self.cellid,self.latitude,self.longitude,self.eventType)

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

    def distance_latlon(self, latitude, longitude):
        """ Computes the euclidean distance  in Km"""
        PIov180 = 0.017453292519943295
        dLat = (float(latitude) - float(self.latitude)) * PIov180
        dLon = (float(longitude) - float(self.longitude)) * PIov180
        a = math.sin(dLat/2)**2 + (math.sin(dLon/2)**2) * math.cos(float(self.latitude)*PIov180) * math.cos(float(latitude)*PIov180)
        divisor = math.sqrt(1-a)
        if (divisor == 0):
            divisor = 1
        return round(12742 * math.atan(math.sqrt(a)/ divisor), 3)

    @classmethod
    def distance_only_latlon(cls, latitude1, longitude1,latitude2, longitude2):
        """ Computes the euclidean distance  in Km"""
        PIov180 = 0.017453292519943295
        dLat = (float(latitude2) - float(latitude1)) * PIov180
        dLon = (float(longitude2) - float(longitude1)) * PIov180
        a = math.sin(dLat/2)**2 + (math.sin(dLon/2)**2) * math.cos(float(latitude1)*PIov180) * math.cos(float(latitude2)*PIov180)
        divisor = math.sqrt(1-a)
        if (divisor == 0):
            divisor = 1
        return round(12742 * math.atan(math.sqrt(a)/ divisor), 6)


    def time_distance(self, mobilityTrace):
        """ Computes de difference between the timestamp of the eventement passed in arguments and the timestamp of the object that calls the method in seconds"""
        deltaTime = mobilityTrace.timestamp - self.timestamp
        return deltaTime.total_seconds()

    def speed (self, mobilityTrace):
        """ computes the speed between two points  """
        deltaTime=mobilityTrace.timestamp-self.timestamp
        computedSpeed = -1
        if (deltaTime.total_seconds() != 0):
            computedSpeed =  self.distance(mobilityTrace)/(deltaTime.total_seconds()/3600)
        else:
            computedSpeed = 0
        return math.fabs(computedSpeed)

#################################################
#		Class methods			#
#################################################

    @classmethod
    def computeMediod(cls,trialMobilityTraces):
        """ Computes the mobility trace that minimizes the distance
            among a set of mobility traces"""
        size=len(trialMobilityTraces)
        matrix = [[0]*size for i in range(size)]
        cumulatedDistance = [0]*size
        i = 0
        j = 0

        while (i < size):
            j = i
            while (j < size):
                if (i != j):
                    mt1=trialMobilityTraces[i]
                    mt2=trialMobilityTraces[j]
                    aux = mt1.distance(mt2)
                    matrix[i][j] = aux
                    matrix[j][i] = aux
                    cumulatedDistance[i] = cumulatedDistance[i] + aux
                    cumulatedDistance[j] = cumulatedDistance[j] + aux
                j = j + 1
            i = i + 1

        minValue = min(cumulatedDistance)
        medioid =  trialMobilityTraces[cumulatedDistance.index(minValue)]
        return copy.deepcopy(medioid)
    @classmethod
    def computeCentroid(cls,trialMobilityTraces):
        """ Coputes the mobility trace corresponding to the average
            of the set of points """
        size=len(trialMobilityTraces)
        i = 0
        avg_latitude = 0
        avg_longitude = 0
        while (i < size-1):
            mt1=trialMobilityTraces[i]
            avg_latitude = avg_latitude + mt1.latitude
            avg_longitude = avg_longitude + mt1.longitude
            i = i + 1
            mt = MobilityTrace(0,0,avg_latitude/i,avg_longitude/i,"Centroid")
        return mt
    @classmethod
    def filterSpeed (cls, trialMobilityTraces, eps = 0):
        """ Erase all points where the speed is greater tha eps  """
        new_trialMobilityTraces = list (trialMobilityTraces)
        size = len(trialMobilityTraces)
        index_todelete = list ()
        i=0

        while (i < size - 1):
            mt1 = trialMobilityTraces[i]
            mt2 = trialMobilityTraces[i+1]
            aux_speed = mt1.speed(mt2)

            if (aux_speed >= eps):
                index_todelete.append(i)

            i = i + 1

        for offset, index in enumerate(index_todelete):
            index -= offset
            del new_trialMobilityTraces[index]

        return new_trialMobilityTraces

    @classmethod
    def compute_frequency_update(cls, trialMobilityTraces):
        """ Computes the averge time in seconds of the position update (in seconds)
        and the average distance between two updates (in Km) """
        size = len(trialMobilityTraces)
        frequency_time = 0
        frequency_distance = 0
        i = 0
        while (i < size - 1):
            mt1 = trialMobilityTraces[i]
            mt2 = trialMobilityTraces[i+1]
            frequency_time += mt1.time_distance(mt2)
            frequency_distance += mt1.distance(mt2)
            i += 1

        if (i != 0):
            frequency_time=frequency_time/i
            frequency_distance=frequency_distance/i

        return frequency_time , frequency_distance

    @classmethod
    def distance_matrix(cls, trialMobilityTraces, euclidean = True):
        """ Compute a distance matrix, in numpy form,  of the given trial of mobility traces"""
        if (euclidean):
            size= len( trialMobilityTraces )
            matrix = numpy.zeros(shape=(size,size))
            i = 0
            j = 0

            while (i < size):
                j = i
                while (j < size):
                    mt1=trialMobilityTraces[i]
                    mt2=trialMobilityTraces[j]
                    dist = mt1.distance(mt2)
                    matrix[i][j] = dist
                    matrix[j][i] = dist
                    j = j + 1
                i = i + 1
            return matrix


    @classmethod
    def spatial_filter (cls, trialMobilityTraces, eps=0):
        """ Erase the repeated configuous points.
        It returns a new list  with no contiguos repeated points
        """

        new_list = list (trialMobilityTraces)
        i = size= len( trialMobilityTraces ) - 1

        while (i > 0):
            mt1 = new_list[i]
            mt2 = new_list[i-1]
            dist = mt1.distance(mt2)
            if ( dist <= eps ):
                new_list.remove(mt1)
            i = i - 1
        return new_list


    @classmethod
    def count_number_of_unique_antennas (cls, trialMobilityTraces, eps=0):
        """ Count the unique number of antennas in a trail of mobility traces
            it returns a dictionary having as key the cell id and as value
            the count of the given antenna: [cell_id : int ]
        """
        stat_antennas = {}

        for mt in  trialMobilityTraces:
            if (mt.cellid in stat_antennas):
                stat_antennas[mt.cellid]+=1
            else:
                stat_antennas[mt.cellid]=1

        return stat_antennas

    @classmethod
    def squashMobilityTraces (cls, trialMobilityTraces):
        """
            Erases all consecutive mobility traces returning only position transitions
        """
        new_list = list (trialMobilityTraces)
        i = len(new_list)
        while (i > 0):
            mtn = new_list[i]
            mtn_1 = new_list[i-1]
            if (mtn.cellid == mtn_1.cellid):
                new_list.remove(mtn)
            i-=1

        return  new_list


    @classmethod
    def exportMobilityTraceToMap (cls,listMobTraces,path):
        """ Exports a list of mobility traces as a csv file to be drawn in R
            listMobTraces : list of mobility traces to export
            path : the path whe the file will be created
        """
        with open(path, 'wb') as csvfile:
            oWriter = csv.writer(csvfile, delimiter=';', quotechar=' ', quoting=csv.QUOTE_ALL)
            oWriter.writerow(["poi","lat","lon"])
            size = len(listMobTraces)
            i = 0
            while (i < size):
                mt=listMobTraces[i]
                desc="%s) %s %s"%(str(i),str(mt.timestamp),str(mt.eventType))
                oWriter.writerow([desc,mt.latitude,mt.longitude])
                i+=1

    @classmethod
    def compute_avg_speed(cls, trailMobilityTraces):
        """
            This method returns a float representing the average speed of a set of mobility traces
            @param: The trail of mobility traces
            @result: the average speed
        """
        speed = 0
        size = len(trailMobilityTraces)
        if (size > 1):
            i = 0
            while (i < size-1):
                speed += float( trailMobilityTraces[i].speed(trailMobilityTraces[i+1]))
                i += 1
        return (speed/float(size))

    @classmethod
    def compute_cumulated_distance(cls, trailMobilityTraces):
        """
            This method returns a float representing the cumulated distance of a trail of
            mobility trace
            @param: The trail of mobility traces
            @result: the acumulated distance
        """
        distance = 0
        size = len(trailMobilityTraces)
        if (size > 1):
            i = 0
            while (i < size-1):
                distance += float( trailMobilityTraces[i].distance(trailMobilityTraces[i+1]))
                i += 1
        return (distance)

    @classmethod
    def compute_acumulated_time(cls, trailMobilityTraces):
        """
            This method returns a float representing the cumulated time of a trail of
            mobility trace in seconds
            @param: The trail of mobility traces
            @result: the acumulated time in seconds
        """
        distance = 0
        size = len(trailMobilityTraces)
        if (size > 1):
            i = 0
            while (i < size-1):
                distance += float( trailMobilityTraces[i].time_distance(trailMobilityTraces[i+1]))
                i += 1
        return (distance)

#########################
#       TEST            #
#########################
if __name__ == "__main__":
    mt1=MobilityTrace(1172969203,42,9.42,56.13,"CELL_ATTACH")
    mt2=MobilityTrace(1172969213,41,9.44,56.21,"CELL_ATTACH")
    mt3=MobilityTrace(1172969303,43,9.45,56.23,"CELL_ATTACH")
    mt4=MobilityTrace(1172969503,44,9.47,56.3,"CELL_ATTACH")
    mt5=MobilityTrace(1172969903,41,9.48,56.23,"CELL_ATTACH")
    mylist = [mt1,mt2,mt3,mt4,mt5]
    mylist2= [mt1,mt2]
    dist=mt1.distance(mt2)
    speed=mt5.speed(mt4)
    print ("'{0}' distance {1}, speed: {2}".format(mt1,dist,speed))
    mtm = MobilityTrace.computeMediod(mylist)
    mtc = MobilityTrace.computeCentroid(mylist)
    print "medioid: '{0}' centroid: '{1}'".format(mtm,mtc)

    filtered = MobilityTrace.filterSpeed(mylist,22)
    print "filterd list: {0}".format(filtered)

    frec_t, freq_d = MobilityTrace.compute_frequency_update ( mylist )
    print("frequency: '{0} sec' and '{1} Km'".format(frec_t, freq_d))

    dist_matrix = MobilityTrace.distance_matrix(mylist)
    print "{0}".format(dist_matrix)

    spatial_filtered = MobilityTrace.spatial_filter(mylist)
    print "{0}".format(spatial_filtered,10)

    unique_antennas= MobilityTrace.count_number_of_unique_antennas(mylist)
    print "{0}".format(unique_antennas)

    print "{0}".format((mylist2[1] in mylist))
