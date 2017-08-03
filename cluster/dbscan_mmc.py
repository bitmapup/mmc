import logging
import numpy
from mmc.mobilitytrace import MobilityTrace
from cluster import Cluster
class Dbscan_mmc (Cluster):

    def __init__ (self, minPts, eps, listModels):
        self._minPts = minPts
        self._eps = eps
	self._listModels = listModels
        super(Dbscan_mmc,self).__init__()
######################
#   Properties
######################

    @property
    def minPts (self):
        return self._minPts
    @minPts.setter
    def minPts (self,value):
        self._minPts = value

    @property
    def eps  (self):
        return self._eps
    @eps.setter
    def eps (self,value):
        self._eps = value

    @property
    def listModels (self):
        return self._listModels
    @listModels.setter
    def minPts (self,value):
        self._listModels = value



######################
#   Methods
######################


    def doCluster(self,path,pDistance):
        """
            @path = "models/test/*.mmc"
        """
        matrix = Mmc.buildDistanceMatrix(pathToListOfModels = path ,distance= pDistance)
        self.doClusterFromDistanceMatrix(matrix)

    def doClusterFromDistanceMatrix(self,dMatrix):
        """
            generate groups from a distance matrix
            @dMatrix is the distance matrix based on coverage distance
        """
        size = len(self._listModels)
        i = 0
        j = 0
        aux_list = list()
        idCluster = 0

        to = -1
        while (i < size-1):

            row = dMatrix[i,:]
            sorted_row_index = numpy.argsort(row)
            #print (self._listModels[i])
            #print(row)
            #print(sorted_row_index)
            j = 0
            while (j < size):
                index = sorted_row_index[j]

                if (i != index):
                    #print "test: {0} > {1} : {2}".format(dMatrix[i,index],self._eps,(dMatrix[i,index]>self._eps))
                    if ( dMatrix[i,index] > self._eps ):
                        to = j
                        j = size

                j += 1

            aux_group = self.__getGroup__(to,self._listModels,sorted_row_index)
            #print "group: {0}".format(aux_group)
            #verifies if a group exist
            if (len(aux_group) >= self._minPts):
                hasIntersection = False

                for key in self._dict_cluster:
                    if (self.__hasIntersection__(self._dict_cluster[key],aux_group)):
                        self._dict_cluster[key]= self.__merge__(self._dict_cluster[key],aux_group)
                        hasIntersection =  True
                        break

                if (not hasIntersection):
                   self._dict_cluster[idCluster] = aux_group
                   idCluster += 1

            aux_group = list()
            i += 1


    def __getGroup__ (self, to, listModels, indexArray ):
        """
            return a list of users
        """
        size = to
        i = 0
        group = list()

        while (i < size):
            index = indexArray[i]
            group.append(listModels[index])
            i += 1

        return group

    def __hasIntersection__(self,a,b):
         return len(list(set(a) & set(b))) > 0

    def __merge__(self,a,b):
        return list(set(a) | set(b))

    def clusterToCsv(self,pathFile):
         with open(pathFile,'wb') as f:
             f.write("group,user\n")
             for key in  self._dict_cluster:
                 for item in self._dict_cluster[key]:
                     f.write("{0},{1}\n".format(key,item))
         f.close()

    #end clusterToCsv():

    @classmethod
    def csvToCluster(cls,pathFile):
         """ Build a object Dbscan_mmc from a csv file having "group"(int)
         and "user" (file name string) columns
         """
         labels=["group,user"]
         dict_cluster = dict()
         with open(pathFile,'rb') as tsvin:
             logging.info("Open: {0}".format(pathFile))
             tsvin = csv.reader(tsvin, delimiter=',')
             tsvin.next()

             for row in  tsvin:
                group = row[labels.index("group")]
                user = row[labels.index("user")]
                if group in dict_cluster:
                    dict_cluster[group].append(user)
                else:
                    dict_cluster[group] = [user]
         oDbscan_mmc = Dbscan_mmc(0,0,list())
         oDbscan_mmc.dict_clusters = dict_cluster
         return oDbscan_mmc
    #end clusterToCsv():
