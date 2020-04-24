#!/usr/bin/env python

'''
DESCRIPTION library for growth-centric objects.
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS

from libs import agp,aio,misc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

class GrowthPlate(object):

    def __init__(self,data=None,key=None,time=None,input_time=None,input_data=None,mods=None):
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

        self.mods = pd.DataFrame(columns=['logged','floored','controlled'],index=['status'])
        self.mods = self.mods.apply(lambda x: False) 

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


    def subtractControl(self,to_do=False,drop=True):
        '''
        Subtract from each treatment sample's growth curve, the growth curve of its corresponding control sample.

        Args:
            to_do (boolean): if False, do not subtract control wells and return None.
            drop (boolean): if True, drop control samples from data.
        '''

        if not to_do:
            return None

        data = self.data.copy()
        mapping = self.key

        # find all unique groups
        plate_groups = mapping.loc[:,['Plate_ID','Group']].drop_duplicates()
        plate_groups = [tuple(x) for x in plate_groups.values]

        for plate_group in plate_groups:

            pid,group = plate_group

            # grab lists of Sample_ID of wells corresponding to control and cases
            controls = misc.subsetDf(mapping,{'Plate_ID':[pid],'Group':[group],'Control':[1]}).index.values
            cases = misc.subsetDf(mapping,{'Plate_ID':[pid],'Group':[group]}).index.values  # includes controls

            data_controls = data.loc[:,controls]
            data_cases = data.loc[:,cases]

            # for each case, divide data by mean controls (if log-transformed), o.w. subtract mean controls
            data_controls = data_controls.mean(1)
            data_cases = (data_cases.T - data_controls).T
            data.loc[:,cases] = data_cases.values

            if drop:
                data = data.drop(controls,axis=1)

        self.data = data
        self.mods.controlled = True


    def thinMeasurements(self,step=11):
        '''
        Thin the number of measurements (for both object's time and data).

        Args:
            step (int)
        '''

        matrix = (self.time).join(self.data)
        select = np.arange(0,self.data.shape[0],step)
        matrix = matrix.iloc[select,:]

        self.time = pd.DataFrame(matrix.iloc[:,0])
        self.data = matrix.iloc[:,1:] 


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

        # if mapping lacks group and control columns skip
        if ('Group' not in mapping.keys()) or ('Control' not in mapping.keys()):
            return None            

        df = self.input_data.copy()  # timepoints by wells, input data that remains unmodified

        # subtract first time point from each column (i.e. wells) 
        if subtract_baseline:
            baseline = df.iloc[0,:]
            df = df.apply(lambda row: row - baseline,axis=1)

        # find all unique groups
        plate_groups = mapping.loc[:,['Plate_ID','Group']].drop_duplicates()
        plate_groups = [tuple(x) for x in plate_groups.values]

        for plate_group in plate_groups:
            
            pid,group = plate_group

            # grab lists of Sample_ID of wells corresponding to control and cases
            controls = misc.subsetDf(mapping,{'Plate_ID':[pid],'Group':[group],'Control':[1]}).index.values
            cases = misc.subsetDf(mapping,{'Plate_ID':[pid],'Group':[group],'Control':[0]}).index.values

            # if group does not have a control, skip
            if len(controls)==0:
                continue

            df_controls = df.loc[:,controls]
            df_cases = df.loc[:,cases]

            # for denominator, max by control column (i.e. well), then average all controls
            df_controls_fc = df_controls.max(0) / df_controls.max(0).mean(0)
            df_cases_fc = df_cases.max(0) / df_controls.max(0).mean(0)

            mapping.loc[controls,'Fold_Change'] = df_controls_fc
            mapping.loc[cases,'Fold_Change'] = df_cases_fc

        self.key = mapping


    def convertTimeUnits(self,input,output):
        '''
        Converts object's time pandas.DataFrame values between different time intervals. 
            It can convert between seconds, minutes, hours, and days.

        Args:
            input (str): either 'seconds','minutes','hours','days' (current unit of time)
            output (str): either 'seconds','minutes','hours','days' (desired unit of time)
        '''

        seconds = 1.0
        minutes = 60.0  # seconds
        hours = 3600.0  # seconds
        days  = 3600.0 * 24  # second

        time_dict = {'seconds':seconds,'minutes':minutes,'hours':hours,'days':days}

        scaler = time_dict[input] / time_dict[output]
        self.time = self.time.astype(float) * scaler


    def logData(self):
        '''
        Transform with a natural logarithm all values in object's data (pandas.DataFrame).
        '''

        self.data = self.data.apply(lambda x: np.log(x))
        self.mods.logged = True


    def subtractBaseline(self):
        '''
        Subtract the first value in each column from all values of the column.

        WARNING: if performed after natural-log-transformation of data, this is equivalent to 
             scaling relative to OD at T0 (i.e. log[OD(t)] - log[OD(0)] = log[ OD(t) / OD(0) ] )
        '''

        baseline = self.data.iloc[0,:]
        self.data = self.data.apply(lambda x: x - baseline,axis=1)
        self.mods.floored = True


    def isSingleMultiWellPlate(self):
        '''
        Checks if object describes a 96-well plate using several criteria. 

        Returns:
            (boolean)
        '''

        # must have only one Plate_ID associated with all samples
        if len(self.key.Plate_ID.unique()) != 1:
            return False	

        # Well must be a column, values woudl be well locations (e.g. A1)
        if 'Well' not in self.key.columns:
            return False

        # makes sure that all 96 well locations are described in key and thus data
        expc_wells = set(aio.parseWellLayout().index.values)  # expected list of wells 
        list_wells = set(self.key.Well.values)  # actual list of wells

        if len(expc_wells.intersection(list_wells)) == 96:
            return True


    def saveKey(self,save_path):
        '''
        Saves the current instance of the object's key (i.e. mapping file). 

        Args:
            save_path (str): file path to location where key should be stored, includes file name. 

        Actions:
            Saves a tab-separated file in desired location (argument).
        x'''

        self.key.to_csv(save_path,sep='\t',header=True,index=True)

    def plot(self,save_path='',plot_fit=False,plot_derivative=False,plot_raw=False):
        '''
        Creates a 8x12 grid plot (for 96-well plate) that shows the growth curves in each well.
            Plot aesthetics require several parameters that are saved in config.py and pulled using 
            functions in misc.py. Plot will be saved as a PDF to location passed via argument. Index
            column for object's key should be Well IDs but object's key should also have a Well column.

        Args:
            save_path (str): file path: if empty, plot will not be saved at all.
            plot_fit (boolean): whether to plot GP fits on top of raw OD.
            plot_derivative (boolean): if True, plot only the derivative of GP fit instead. 

        Returns:
            fig,axes: figure and axis handles.

        Action:
            if user passes save_path argument, plot will be saved as PDF in desired location 
        '''

        # make sure plate is 96-well version, otherwise skip plotting
        if not self.isSingleMultiWellPlate():
            msg = 'USER ERROR: GrowthPlate() object for {} is not a 96-well plate. '.format(self.key.Plate_ID[0])
            msg += 'AMiGA can not plot it.\n'
            print(msg)
            return None

        self.addLocation()

        if plot_derivative: 
            data = self.derivative_prediction
        elif plot_raw:
            data = self.input_data
        elif plot_fit:
            data = self.floored_real_input
        else:
            data = self.data

        time = self.time
        key = self.key

        fig,axes = plt.subplots(8,12,figsize=[12,8])

        # define window axis limits
        ymax = np.ceil(data.max(1).max())
        ymin = np.floor(data.min(1).min())

        if plot_fit:
            ymin = 0

        xmin = 0
        xmax = time.values[-1]
        xmax_up = int(np.ceil(xmax)) # round up to nearest integer

        for well in data.columns:

            # select proper sub-plot
            r,c = key.loc[well,['Row','Column']] -1
            ax = axes[r,c]
            
            # get colors based on fold-change and uration parameters
            color_l,color_f = misc.getPlotColors(key.loc[well,'Fold_Change'])

            # set window axis limits
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])

            # define x-data and y-data points
            x = np.ravel(time.values)
            y = data.loc[:,well].values

            # plot line and fill_betwen, if plotting OD estimate
            ax.plot(x,y,color=color_l,lw=1.5,zorder=10)
            if not plot_derivative:
                ax.fill_between(x=x,y1=[ax.get_ylim()[0]]*len(y),y2=y,color=color_f,zorder=7)

            # add fit lines, if desired
            if plot_fit:
                y_fit = self.floored_real_prediction.loc[:,well].values
                ax.plot(x,y_fit,color='yellow',alpha=0.65,ls='--',lw=1.5,zorder=10)

            # show tick labels for bottom left subplot only, so by default no labels
            if plot_derivative:
                plt.setp(ax,yticks=[ymin,0,ymax],yticklabels=[])  # zero derivative indicates no instantaneous growth
            else:
                plt.setp(ax,yticks=[ymin,ymax],yticklabels=[])
            plt.setp(ax,xticks=[xmin,xmax],xticklabels=[])

            # add well identifier on top left of each sub-plot
            well_color = misc.getTextColors('Well_ID')
            ax.text(0.,1.,key.loc[well,'Well'],color=well_color,
                ha='left',va='top',transform=ax.transAxes)

            # add Max OD value on top right of each sub-plot
            ax.text(1.,1.,"{0:.2g}".format(key.loc[well,'OD_Max']),color=misc.getTextColors('OD_Max'),
                ha='right',va='top',transform=ax.transAxes)

       # show tick labels for bottom left sub-plot only
        plt.setp(axes[7,0],xticks=[0,xmax],xticklabels=[0,xmax_up])
        plt.setp(axes[7,0],yticks=[ymin,ymax],yticklabels=[ymin,ymax])

        # add x- and y-labels and title
        ylabel_base = misc.getValue('grid_plot_y_label')
        ylabel_mod = ['ln ' if self.mods.logged else ''][0]

        if plot_derivative:
            ylabel_text = 'd[{}]/dt'.format(ylabel_base)
        else:
            ylabel_text = ylabel_mod + ylabel_base

        # add labels and title 
        fig.text(0.512,0.07,'Time ({})'.format(misc.getTimeUnits('output')),fontsize=15,
            ha='center',va='bottom')
        fig.text(0.100,0.50,ylabel_text,fontsize=15,
            ha='right',va='center',rotation='vertical')
        fig.suptitle(x=0.512,y=0.93,t=key.loc[well,'Plate_ID'],fontsize=15,
            ha='center',va='center')

        if save_path!='': # if no file path passed, do not save 
            plt.savefig(save_path)

        plt.close()

        return fig,axes

    def addLocation(self):
        '''
        Expands object key to include following columns: Row and Column, which desribe
            the location of each sample in a multi-well plate (e.g. 96-well plate). However,
            locations here are numerical (i.e Rows 1-8 will correspond to A-H, respectively. 

        Action:
            self.key will have two additional columns, if they were missing prior to execution.
        '''

        if all(x in self.key.columns for x in ['Row','Column']):
            return None

        if 'Well' in self.key.columns:
            self.key = self.key.join(aio.parseWellLayout(),on='Well')
        else:
            self.key = self.key.join(aio.parseWellLayout().reset_index())

        row_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}

        self.key.Row = self.key.Row.replace(row_map)
        self.key.Column = self.key.Column.replace(int)


    def extractGrowthData(self,args_dict={},unmodified=False):
        '''
        Creates a GrowthData() object that is equal or smaller based on user-passed criteria.

        Args:
            args_dict (dictionary): keys are experimental variables and values are instances of these variables
            unmodified (boolean): whether to extract data and only pass the input unmodified
                data and time attributes from parent. 

        Returns:
            obj (GrowthPlate object): see definition of parent class for more details.
        '''

        # make sure criteria for selecting data based on experimental variables is not empty
        if not bool(args_dict):
            msg = "USER ERROR: To extract selected growth data, "
            msg += "you must pass criteria for selection to GrwothPlate().extractGrowthData()."
            print(msg)
            return None

        # make sure that criteria for selecting data is formatted as a dictionary of lists or np.arrays
        for dict_key,dict_value in args_dict.items():
            if isinstance(dict_value,(list,np.ndarray)):
                continue
            elif isinstance(dict_value,(int,float,str)):
                args_dict[dict_key] = [dict_value]

        sub_key = self.key
        sub_key = sub_key[sub_key.isin(args_dict).sum(1)==len(args_dict)]
        sub_mods = self.mods

        sub_input_data = self.input_data.loc[:,sub_key.index]
        sub_input_time = self.input_time

        if unmodified:
            sub_data = sub_input_data
            sub_time = sub_input_time
        else:
            sub_data = self.data.loc[:,sub_key.index]
            sub_time = self.time

        # package as a GrowthPlate object
        obj = GrowthPlate(sub_data,sub_key,sub_time,sub_input_time,sub_input_data,sub_mods)

        return obj


    def copy(self):
        '''
        Creates a copy of class instance.

        Returns:
            obj (GrowthPlate object): see definition of parent class for more details.
        '''

        return deepcopy(self)


    def model(self,store=False,dx_ratio_min=0.25,dx_fc_min=1.5):
        '''
        Infers growth parameters of interest (including diauxic shifts) by Gaussian Process fitting of data.

        Args:
            store (boolean): if True, certain data will be store as object's attributes
            diauxie (float): ratio of peak height (relative to maximum) used to call if diauxie occured or not

        Actions:
            modifies self.key, and may create self.prediction and self.derivative_prediction objects
        '''

        data_ls, diauxie_ls = [], []

        # initialize dataframe to store resutls of GP fitting
        ls_params = ['auc','k','gr','dr','td','lag']
        df_params = pd.DataFrame(index=self.key.index,columns=ls_params)

        for sample_id in self.key.index:

            #print(self.key.loc[sample_id,['Well','Plate_ID']].values)

            # extract sample
            args_dict = self.key.loc[sample_id,['Well','Plate_ID']].to_dict()
            sample_growth = self.extractGrowthData(args_dict)

            # create GP object and analyze
            gp = agp.GP(sample_growth.time,sample_growth.data,sample_growth.key)
            gp.describe(dx_ratio_min=dx_ratio_min,dx_fc_min=dx_fc_min)
            gp_params = gp.params

            # directly record select parameters in dataframe
            df_params.loc[sample_id,ls_params] = [gp_params[ii] for ii in ls_params]

            # passively save to diauxic shift detection resutls, manipulation occurs below
            diauxie_ls.append([sample_id,gp_params['diauxie'],gp_params['peaks']])

            # passively save to data, manipulation occurs below (input OD, GP fit, & GP derivative)
            data_ls.append(gp.data(sample_id))

        # maximum number of growth phases based on diauxic shift detection
        max_n_peaks = np.max([len(ii[2]) for ii in diauxie_ls])

        # initialize dataframe to store results of diauxic shift detection
        columns = ['diauxie'] + ['t_peak_{}'.format(ii) for ii in range(1,max_n_peaks+1)]
        diauxie_df = pd.DataFrame(index=self.key.index,columns=columns)

        # populate diauxic shift dataframe
        for ii,(sample_id,status,peaks) in enumerate(diauxie_ls):
            diauxie_df.loc[sample_id,'diauxie'] = status  # 1 if diauxic shift detected, 0 otherwise
            for jj,peak in enumerate(peaks):
                diauxie_df.loc[sample_id,'t_peak_{}'.format(jj+1)] = peak   # time at which growth rate is at local maxima

        # record results in object's key
        self.key = self.key.join(df_params) 
        self.key = self.key.join(diauxie_df) 

        # plotting needs raw OD & GP fit, and may need GP derivative, save all as obejct's attributes
        if store:
            data_df = pd.concat(data_ls).reset_index(drop=True)
            input_df = data_df.pivot(columns='Sample_ID',index='Time',values='OD')
            pred_df = data_df.pivot(columns='Sample_ID',index='Time',values='Fit')
            derivative_df = data_df.pivot(columns='Sample_ID',index='Time',values='Derivative')
            self.floored_real_input = input_df
            self.floored_real_prediction = pred_df
            self.derivative_prediction = derivative_df

        return None

