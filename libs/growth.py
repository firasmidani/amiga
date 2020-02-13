#!/usr/bin/env python

'''
DESCRIPTION library for growth-centric objects.
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS

from libs import aux

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


    def computeBasicSummary(self):
        '''
        Computes basic characteristics of each column in the object's input_data attribute. In particular,
            + OD_Baseline is the first measurment (0-row)
            + OD_Max is the maximum measurment (any row)
            + OD_Min is the miimum measurement (any row)
        '''

        df = self.input_data.copy()  # timepoints by wells, input data that remains unmodified
        df_max = df.max(0)  # max by column (i.e. well)
        df_min = df.min(0)  # min by column (i.e. well)
        df_baseline = df.iloc[0,:]

        joint_df = pd.concat([df_baseline,df_min,df_max],axis=1)
        joint_df.columns = ['OD_Baseline','OD_Min','OD_Max']

        self.key = self.key.join(joint_df)


    def computeFoldChange(self,subtract_baseline=True):
        '''
        Computes the fold change for all wells using the object's unmodified raw data. The object's key
            must have the following columns ['Plate_ID','Gropu','Control']. Control values must be {0,1}.
            The fold change is computed using measurements that have had the first measurment (first time-
            point subtracted, first. The maximum measurement in controls are averaged to get the scaler 
            (i.e. the average maximum OD of control wells) which divides the maximum OD of all cases.
            Fold-changes are normalized to controls belonging to the same group, all wells in a Biolog plate
            will belong to the same group and have the same control (A1 well).
        '''

        mapping = self.key.copy()
        df = self.input_data.copy()  # timepoints by wells, input data that remains unmodified

        # subtract first time point from each column (i.e. wells) 
        if subtract_baseline:
            df = df.apply(lambda row: row - df.iloc[0,:],axis=1)

        # find all unique groups
        plate_groups = mapping.loc[:,['Plate_ID','Group']].drop_duplicates()
        plate_groups = [tuple(x) for x in plate_groups.values]

        for plate_group in plate_groups:
            
            pid,group = plate_group

            # grab lists of Sample_ID of wells corresponding to control and cases
            controls = aux.subsetDf(mapping,{'Plate_ID':[pid],'Group':[group],'Control':[1]}).index.values
            cases = aux.subsetDf(mapping,{'Plate_ID':[pid],'Group':[group],'Control':[0]}).index.values

            df_controls = df.loc[:,controls.index.values]
            df_cases = df.loc[:,cases.index.values]

            # for denominator, max by control column (i.e. well), then average all controls
            df_controls_fc = df_controls.max(0) / df_controls.max(0).mean(0)
            df_cases_fc = df_cases.max(0) / df_controls.max(0).mean(0)

            mapping.loc[controls,'Fold_Change'] = df_controls_fc
            mapping.loc[cases,'Fold_Change'] = df_cases_fc

        self.key = mapping


    def addLocationVarbs(self):
        '''
        '''

        if all(x in self.key.columns for x in ['Row','Column']):
            return None

        row_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'H':7,'G':8}

        if 'Well' in self.key.columns:
            self.key = self.key.join(parseWellLayout(),on='Well')
        else:
            self.key = self.key.join(parseWellLayout().reset_index())
