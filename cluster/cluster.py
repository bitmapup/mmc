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


