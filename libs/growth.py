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

        self.input_time = self.time.copy()
        self.input_data = self.data.copy()

        assert type(key) == pd.DataFrame, "key must be a pandas DataFrame"
        assert (self.data.shape[1]) == (self.key.shape[0]), "key must contain mapping data for all samples"


    def computeFoldChange(self):
        '''
        '''

        mapping = self.key.copy()

        df = self.input_data.copy()  # timepoints by wells, input data that remains unmodified
        #df_max = df.max(0)  # max by column (i.e. well)
        #df_min = df.min(0)  # min by column (i.e. well)

        # subtrat first time point from each column (i.e. wells) 
        df = df.apply(lambda row: row - df.iloc[0,:],axis=1)

        # compute fold change, maximum measurement (i.e. OD) at any tie-point relative to negative control?
        #df_fc = df_max / df_max.loc[0] 

        # find all unique groups
        plate_groups = mapping.loc[:,['Plate_ID','Group']].drop_duplicates()
        plate_groups = [tuple(x) for x in plate_groups.values]
        for plate_group in plate_groups:
            
            pid,group = plate_group

            controls = {'Plate_ID':[pid],'Group':[group],'Control':[1]}
            controls = mapping[mapping.isin(controls).sum(1)==len(controls)]
            controls = controls.index.values  # list of Sample_ID values

            cases = {'Plate_ID':[pid],'Group':[group],'Control':[0]}
            cases = mapping[mapping.isin(cases).sum(1)==len(cases)]
            cases = cases.index.values  # list of Sample_ID values

            print(cases,controls)

            df_controls = df.loc[:,controls]
            # max by column (i.e. well) then average all control
            df_controls_means = df_controls.max(0).mean(0)
            df_controls_fc = df_controls.max(0) / df_controls_means

            df_cases = df.loc[:,cases]
            df_cases_fc = df_cases.max(0) / df_controls_means

            mapping.loc[controls,'Fold_Change'] = df_controls_fc
            mapping.loc[cases,'Fold_Change'] = df_cases_fc

        print(mapping)

