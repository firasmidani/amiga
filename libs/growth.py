#!/usr/bin/env python

'''
DESCRIPTION library for growth-centric objects.
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS


import pandas as pd


class GrowthPlate(object):

    def __init__(self,data=None,key=None,time=None):

        '''
        Data structure for handling growth data of microtiter plate assays. This however 
            can be generalized to any depenent observations with a single independent observation
            (e.g. time).

        Attributes:

            data (pandas.DataFarme): n x (p+1) DataFrmae (for n timepoints by p wells) that stores
                the raw optical density (or absorbance) measurements. The first column is assumed 
                to be Time. Each succeeding column corresponds to observations for a specific sample
                (i.e. well).
            key (pandas.DataFrame): p x k) DataFrame (for k experimental variables or descriptors) 
                that storees the experimental variables for each sample. One of the columns
                is a unique sample (i.e. well) identifier ('Sample_ID'). You can use Well Locations 
                (e.g. A1) here but if a GrowthPlate object can combine data from multiple plates, 
                thus some wells might share the same well location.
            time (pandas.DataFrame): t x 1 DataFrame (for n timepoints) that stores the raw time measurments.
        '''

        if time is None:
            self.data = data.iloc[:,1:]
            self.time = pd.DataFrame(data.iloc[:,0])
        else:
            self.data = data.copy()
            self.time = time.copy()

        self.key = key.copy()

        assert type(key) == pd.DataFrame, "key must be a pandas DataFrame"
        assert (self.data.shape[1]) == (self.key.shape[0]), "key must contain mapping data for all samples"
