#!/usr/bin/env python

'''
AMiGA library for the GrowthPlate class for storing and manipulting growth curve data.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (1 class with 16 sub-functions)

# GrowthPlate (CLASS)
#   __init__
#   computeBasicSummary
#   subtractControl
#   thinMeasurements
#   computeFoldChange
#   convertTimeUnits
#   logData
#   subtractBaseline
#   isSingleMultiWellPlate
#   saveKey
#   plot
#   addLocation
#   extractGrowthData
#   copy
#   compute_k_error
#   model

import sys
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from copy import deepcopy

from .params import initParamDf, mergeDiauxieDfs
from .plot import largeTickLabels
from .model import GrowthModel
from .comm import smartPrint
from .detail import parseWellLayout
from .utils import subsetDf, handle_non_pos, getPlotColors, getTextColors, getValue, getTimeUnits

pd.set_option('future.no_silent_downcasting', True)

class GrowthPlate:

    def __init__(self,data=None,key=None,time=None,input_time=None,input_data=None,mods=None,gp_data=None):
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
            input_time (pandas.DataFrame): time attribute but not subjected to any modifications (see mods)
            input_data (pandas.DataFrame): data attribute but not subject to any modifiecations (see mods)
            mods (pandas.DataFrame): tracks which modifications were applied to data & time attributes
            gp_data (pandas.DataFrame): each row is a specific time-point for a specific sample. Identifying columns
                include Plate_ID, Time, Sample_ID. Data columns include
                    + OD (original data)
                    + OD_Fit (gp model fit of original data)
                    + OD_Derivative (gp model fit of derivative, insensitive to y-value, i.e. whether OD is centered)
                    + GP_Input (input to gp.GP() object), this is usually log-transformed and log-baseline-subtracted
                    + GP_Output (output of gp.GP().predict()), hence also log-trasnformed and log-baseline-subtracted
                    + OD_Growth_Data (GP_Input but converted to real OD and centered at zero)
                    + OD_Growth_Fit (GP_Output but converted to real OD and centered at zero)
        '''

        if time is None:
            self.data = data.iloc[:,1:]
            self.time = pd.DataFrame(data.iloc[:,0])
        else:
            self.data = data.copy()
            self.time = time.copy()

        if input_time is None:
            self.input_time = self.time.copy()
        else:
            self.input_time = input_time
        
        if input_data is None:
            self.input_data = self.data.copy()
        else:
            self.input_data = input_data

        self.key = key.copy()
        self.gp_data = gp_data

        if mods is None:
            self.mods = pd.DataFrame(columns=['logged','floored','controlled','raised'],index=['status'])
            self.mods = self.mods.apply(lambda x: False) 
        else:
            self.mods = mods

        assert isinstance(key,pd.DataFrame), "key must be a pandas DataFrame"
        assert (self.data.shape[1]) == (self.key.shape[0]), "key must contain mapping data for all samples"


    def computeBasicSummary(self):
        '''
        Computes basic characteristics of each column in the object's input_data attribute. In particular,
            + OD_Baseline is the first measurment (0-row)
            + OD_Max is the maximum measurment (any row)
            + OD_Min is the miimum measurement (any row)
        '''

        # compute OD_Min, OD_Max, and OD_Baseline
        df = self.input_data.copy()  # timepoints by wells, input data that remains unmodified
        df_max = df.max(axis=0,numeric_only=True)  # max by column (i.e. well)
        df_min = df.min(axis=0,numeric_only=True)  # min by column (i.e. well)
        df_baseline = df.iloc[0,:]

        # compute OD_Emp_AUC
        time = self.time.values
        dt = np.mean(time[1:,0]-time[:-1,0])
        df_auc = df.apply(lambda x: np.dot(np.repeat(dt,df.shape[0]).T,x),axis=0)

        joint_df = pd.concat([df_baseline,df_min,df_max,df_auc],axis=1)
        joint_df.columns = ['OD_Baseline','OD_Min','OD_Max','OD_Emp_AUC']

        self.key = self.key.join(joint_df)


    def subtractControl(self,to_do=False,drop=True,blank=False):
        '''
        Subtract from each treatment sample's growth curve, the growth curve of its corresponding control sample.

        Args:
            to_do (boolean): if False, do not subtract control wells and return None.
            drop (boolean): if True, drop control samples from data.
        '''

        if not to_do:
            return None

        data = self.data.copy()
        mapping = self.key.copy()

        pid_text = 'Plate_ID'
        grp_text = 'Group'
        ctr_text = 'Control'
        if blank:
            grp_text, ctr_text = f'Blank{grp_text}', f'Blank{ctr_text}'

        # find all unique groups
        plate_groups = mapping.loc[:,[pid_text,grp_text]].drop_duplicates()
        plate_groups = [tuple(x) for x in plate_groups.values]

        for plate_group in plate_groups:

            pid,group = plate_group

            # grab lists of Sample_ID of wells corresponding to control and cases
            controls = subsetDf(mapping,{pid_text:[pid],grp_text:[group],ctr_text:[1],'Flag':[0]}).index.values
            cases = subsetDf(mapping,{pid_text:[pid],grp_text:[group]}).index.values  # includes controls

            if len(controls)==0:
                msg = '\nFATAL ERROR: User requested subtraction of control samples. However, '
                msg+= f'samples belonging to group {group} of plate {pid} lack ' 
                msg+= 'any corresponding control samples in the current working directory.\n'
                sys.exit(msg)

            data_controls = data.loc[:,controls]
            data_cases = data.loc[:,cases]

            # for each case, divide data by mean controls (if log-transformed), o.w. subtract mean controls
            data_controls = data_controls.mean(axis=1,numeric_only=True)
            data_cases = (data_cases.T - data_controls).T
            data.loc[:,cases] = data_cases.values

            mapping.loc[cases,'Adj_OD_Baseline'] = data.loc[:,cases].loc[0,:].values
            mapping.loc[cases,'Adj_OD_Min'] = data.loc[:,cases].apply(np.min).values
            mapping.loc[cases,'Adj_OD_Max'] = data.loc[:,cases].apply(np.max).values

            if drop: 
                data = data.drop(controls,axis=1)
                mapping = mapping.drop(controls,axis=0)

        self.data = data
        self.key = mapping
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


    def dropFlaggedWells(self,to_do=False):

        if not to_do:
            return None

        flagged = self.key[self.key.Flag==1].index.values

        self.key.drop(labels=flagged,axis=0,inplace=True)
        self.data.drop(labels=flagged,axis=1,inplace=True)


    def computeFoldChange(self,subtract_baseline=True):
        '''
        Computes the fold change for all wells using the object's unmodified raw data. The object's key
            must have the following columns ['Plate_ID','Gropu','Control']. Control values must be {0,1}.
            The fold change is computed using measurements that have had the first measurment (first time-
            point subtracted, first. The maximum measurement in controls are averaged to get the 447r 
            (i.e. the average maximum OD of control wells) which divides the maximum OD of all cases.
            Fold-changes are normalized to controls belonging to the same group, all wells in a Biolog plate
            will belong to the same group and have the same control (A1 well).
        '''

        mapping = self.key.copy()


        # if mapping lacks group and control columns skip
        if ('Group' not in mapping.keys()) or ('Control' not in mapping.keys()):

            mapping.loc[:,'Fold_Change'] = [np.nan]*mapping.shape[0]

            self.key = mapping

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
            controls = subsetDf(mapping,{'Plate_ID':[pid],'Group':[group],'Control':[1]}).index.values
            cases = subsetDf(mapping,{'Plate_ID':[pid],'Group':[group],'Control':[0]}).index.values

            # if group does not have a control, skip
            if len(controls)==0:

                mapping.loc[cases,'Fold_Change'] = [np.nan]*len(cases)

                continue

            df_controls = df.loc[:,controls]
            df_cases = df.loc[:,cases]

            # for denominator, max by control column (i.e. well), then average all controls
            df_controls_fc = df_controls.max(axis=0,numeric_only=True) / df_controls.max(axis=0,numeric_only=True).mean(axis=0,numeric_only=True)
            df_cases_fc = df_cases.max(axis=0,numeric_only=True) / df_controls.max(axis=0,numeric_only=True).mean(axis=0,numeric_only=True)

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


    def raiseData(self):
        '''
        Replace non-positive (i.e. negative or zero) values in each sample measurement with
            lowest positive meausrement in sample. This is necessary if data will be later
            log-transformed otherwise errors will occur.
        '''

        # raise any negative or zero values to a pseudo-count or baseline OD. 
        self.data = self.data.apply(lambda x: handle_non_pos(x))
        self.mods.raised = True

        # add OD_Offset to key
        self.key.loc[:,'OD_Offset'] = self.data.iloc[0,:].values


    def logData(self,to_do=True):
        '''
        Transform with a natural logarithm all values in object's data (pandas.DataFrame).
        '''

        if not to_do:
            return None

        self.data = self.data.apply(lambda x: np.log(x))
        self.mods.logged = True


    def subtractBaseline(self,to_do=True,poly=False,groupby=None):
        '''
        Subtract the first value in each column from all values of the column.

        WARNING: if performed after natural-log-transformation of data, this is equivalent to 
             scaling relative to OD at T0 (i.e. log[OD(t)] - log[OD(0)] = log[ OD(t) / OD(0) ] )
        '''

        if not to_do:
            return None

        if poly:
            p,ind = (3,5) #(polynomial degrees, num time points to use)
            if groupby is None:
                groups = {None:tuple(self.key.index)}
            else:
                groups = self.key.groupby(groupby).groups

            time = self.time.iloc[:ind].values.ravel()

            for k,index in groups.items():
                temp = self.data.loc[:,index]
                od = temp.values[:ind,:]
                coeff = np.polyfit(time*od.shape[1],od,p)
                temp = temp - np.polyval(coeff,self.time.values[0])
                self.data.loc[:,index] = temp
        else:
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

        # Well must be a column, values would be well locations (e.g. A1)
        if 'Well' not in self.key.columns:
            return False

        # makes sure that all 96 well locations are described in key and thus data
        expc_wells = set(parseWellLayout(order_axis=1).index.values)  # expected list of wells 
        list_wells = set(self.key.Well.values)  # actual list of wells

        if len(list_wells) != 96:
            return False

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


    def plot(self,save_path='',plot_fit=False,plot_derivative=False,plot_raw_with_fit=False):
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

        sns.set_style('whitegrid')
        fontsize=15

        time = self.time

        # make sure plate is 96-well version, otherwise skip plotting
        if not self.isSingleMultiWellPlate():
            msg = f'WARNING: GrowthPlate() object for {self.key.Plate_ID.iloc[0]} is not a 96-well plate. '
            msg += 'AMiGA can not plot it.\n'
            print(msg)
            return None

        self.addLocation()

        #key = self.key
        cols =['Sample_ID','Plate_ID','Well','Row','Column','Fold_Change','OD_Max','OD_Baseline','Flag']
        key = self.key.reindex(cols,axis='columns',)
        key = key.dropna(axis=1,how='all')
        if 'Sample_ID' in key.columns:
            key = key.drop_duplicates().set_index('Sample_ID')
        
        if plot_derivative: 
            base_y = self.gp_data.pivot(columns='Sample_ID',index='Time',values='GP_Derivative')
        elif plot_fit:
            base_y = self.gp_data.pivot(columns='Sample_ID',index='Time',values='OD_Growth_Data')
            overlay_y = self.gp_data.pivot(columns='Sample_ID',index='Time',values='OD_Growth_Fit')
        elif plot_raw_with_fit:
            base_y = self.gp_data.pivot(columns='Sample_ID',index='Time',values='OD_Data')
            overlay_y = self.gp_data.pivot(columns='Sample_ID',index='Time',values='OD_Fit')
        else:
            base_y = self.data#gp_data.pivot(columns='Sample_ID',index='Time',values='OD_Data')

        fig,axes = plt.subplots(8,12,figsize=[12,8])

        # define window axis limits
        ymax = np.ceil(base_y.max(axis=1,numeric_only=True).max(numeric_only=True))
        if plot_derivative:
            ymin = np.floor(base_y.min(axis=1,numeric_only=True).min(numeric_only=True))
        else:
            ymin = 0

        if plot_fit:
            ymin = 0

        xmin = 0
        xmax = np.ravel(time)[-1]
        xmax_up = int(np.ceil(xmax)) # round up to nearest integer

        for well in base_y.columns:

            # select proper sub-plot
            r,c = key.loc[well,['Row','Column']] -1
            ax = axes[r,c]

            # highlight flagged well
            if key.loc[well,'Flag'] and getValue('plot_flag_wells')=='empty':

                [ax.spines[ii].set(lw=0) for ii in ['top','bottom','right','left']]
                plt.setp(ax,xticks=[],yticks=[])

                continue

            # get colors based on fold-change and uration parameters
            if 'Fold_Change' in key.keys():
                color_l,color_f = getPlotColors(key.loc[well,'Fold_Change'])
            else:
                color_l = getValue('fcn_line_color')
                color_f = getValue('fcn_face_color')

            # set window axis limits
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])

            # define x-data and y-data points
            x = np.ravel(time.values)
            y = base_y.loc[:,well].values

            # plot line and fill_betwen, if plotting OD estimate
            ax.plot(x,y,color=color_l,lw=1.5,zorder=10)
            if not plot_derivative:
                ax.fill_between(x=x,y1=[ax.get_ylim()[0]]*len(y),y2=y,color=color_f,zorder=7)

            # add fit lines, if desired
            if plot_fit or plot_raw_with_fit:
                y_fit = overlay_y.loc[:,well].values
                ax.plot(x,y_fit,color=getValue('gp_line_fit'),alpha=0.65,ls='--',lw=1.5,zorder=10)

            # show tick labels for bottom left subplot only, so by default no labels
            if plot_derivative:
                plt.setp(ax,yticks=[ymin,0,ymax],yticklabels=[])  # zero derivative indicates no instantaneous growth
            else:
                plt.setp(ax,yticks=[ymin,ymax],yticklabels=[])
            plt.setp(ax,xticks=[xmin,xmax],xticklabels=[])

            # add well identifier on top left of each sub-plot
            well_color = getTextColors('Well_ID')
            ax.text(0.,1.,key.loc[well,'Well'],fontsize=10,color=well_color,ha='left',va='top',transform=ax.transAxes)

            # add Max OD (or Max dOD/dt) value on top right of each sub-plot
            if plot_derivative:
                od_max = np.max(y)
            elif self.mods.floored:
                od_max = key.loc[well,'OD_Max'] - key.loc[well,'OD_Baseline']
            else:
                od_max = key.loc[well,'OD_Max']
            ax.text(1.,1.,f"{od_max:.2f}",fontsize=10,
                color=getTextColors('OD_Max'),ha='right',va='top',transform=ax.transAxes)
            
            if key.loc[well,'Flag'] and getValue('plot_flag_wells')=='cross':
            
                kwargs = {'color':'red','lw':2,'ls':'-','zorder':5}
                (xminf,xmaxf), (yminf,ymaxf) = ax.get_xlim(), ax.get_ylim()
                ax.plot([xminf,xmaxf],[yminf,ymaxf],**kwargs)
                ax.plot([xminf,xmaxf],[ymaxf,yminf],**kwargs)
                plt.setp(ax,xticks=[],yticks=[])


        # show tick labels for bottom left sub-plot only
        plt.setp(axes[7,0],xticks=[0,xmax],xticklabels=[0,xmax_up])
        plt.setp(axes[7,0],yticks=[ymin,ymax],yticklabels=[ymin,ymax])
        largeTickLabels(axes[7,0],fontsize=fontsize)
        
        # add x- and y-labels and title
        ylabel_base = getValue('grid_plot_y_label')
        #ylabel_mod = ['ln ' if self.mods.logged else ''][0]
        ylabel_mod = ''

        if plot_derivative:
            ylabel_text = f'd[ln{ylabel_base}]/dt'
        else:
            ylabel_text = ylabel_mod + ylabel_base

        # add labels and title 
        fig.text(0.512,0.07,'Time ({})'.format(getTimeUnits('output')),fontsize=fontsize,ha='center',va='bottom')
        fig.text(0.100,0.50,ylabel_text,fontsize=fontsize,ha='right',va='center',rotation='vertical')
        fig.suptitle(x=0.512,y=0.93,t=key.loc[well,'Plate_ID'],fontsize=fontsize,ha='center',va='center')

        # if no file path passed, do not save 
        if save_path!='':
            plt.savefig(save_path, bbox_inches='tight')

        self.key.drop(labels=['Row','Column'],axis=1,inplace=True)

        plt.close()

        return fig,axes


    def addLocation(self):
        '''
        Expands object key to include following columns: Row and Column, which describe
            the location of each sample in a multi-well plate (e.g. 96-well plate). However,
            locations here are numerical (i.e Rows 1-8 will correspond to A-H, respectively. 

        Action:
            self.key will have two additional columns, if they were missing prior to execution.
        '''

        cond_1 = all(x in self.key.columns for x in ['Row','Column'])
        if cond_1:
            cond_2 = all([isinstance(ii,int) for ii in self.key.Column.values])
            cond_3 = all([isinstance(ii,int) for ii in self.key.Row.values])
            if cond_2 and cond_3: 
                return None
            else:
                self.key = self.key.drop(labels=['Row','Column'],axis=1)

        if 'Well' in self.key.columns:
            self.key = self.key.join(parseWellLayout(order_axis=1),on='Well')
        else:
            self.key = self.key.join(parseWellLayout(order_axis=1).reset_index())

        row_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
        col_map = {ii:int(ii) for ii in self.key.Column.values}

        self.key.Row = self.key.Row.replace(row_map).infer_objects(copy=False)
        self.key.Column = self.key.Column.replace(col_map).infer_objects(copy=False)


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
        sub_key = sub_key[sub_key.isin(args_dict).sum(axis=1,numeric_only=True)==len(args_dict)]
        sub_mods = self.mods

        sub_input_data = self.input_data.loc[:,sub_key.index]
        sub_input_time = self.input_time

        if unmodified:
            sub_data = sub_input_data
            sub_time = sub_input_time
        else:
            sub_data = self.data.loc[:,sub_key.index]
            sub_time = self.time

        # if any row in data contains at least one np.nan value, it will be dropped
        sub_data = sub_data.dropna(axis=0,how='any')
        sub_time = sub_time.loc[sub_data.index,:]

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


    def compute_k_error(self):
        '''Compute K error for checking quality of fit'''

        def foo(x,thresh):

            if x['expected'] == 0 and x['predicted'] == 0:
                kerr =  0
            elif x['expected'] == 0 and x['predicted'] != 0:
                kerr = np.inf
            else:
                kerr = abs((x['predicted']/x['expected'])-1) * 100

            if kerr > thresh:
                return 'TRUE'
            else:
                return 'FALSE'
        
        # whetehr to use use raw or adjusted OD
        if 'Adj_OD_Max' in self.key.columns:
            refs = ['Adj_OD_Max','Adj_OD_Baseline']
        else:
            refs = ['OD_Max','OD_Baseline']
        thresh = getValue('k_error_threshold')

        # compute K_Error and flag ones that deviate above threshold
        sub_df = pd.DataFrame(index=self.key.index,columns=['expected','predicted'])
        sub_df.loc[:,'expected'] = (self.key.loc[:,refs[0]] - self.key.loc[:,refs[1]]).values
        if self.mods.logged:
            sub_df.loc[:,'predicted'] = self.key.k_lin.values
        else:
            sub_df.loc[:,'predicted'] = self.key.k_log.values
        sub_df.loc[:,'K_Error'] = sub_df.apply(lambda x: foo(x,thresh),axis=1)
        
        self.key.loc[:,f'K_Error > {thresh}%'] = sub_df.loc[:,'K_Error']


    def model(self,nthin=1,store=False,verbose=False):
        '''
        Infers growth parameters of interest (including diauxic shifts) by Gaussian Process fitting of data.

        Args:
            store (boolean): if True, certain data will be store as object's attributes
            diauxie (float): ratio of peak height (relative to maximum) used to call if diauxie occured or not

        Actions:
            modifies self.key, and may create self.latent and self.dlatent_dt objects
        '''

        # initialize variables for storing parameters and data
        data_ls, diauxie_dict = [], {}
        gp_params = initParamDf(self.key.index,0)
        mse_df = pd.DataFrame(index=self.key.index,columns=['MSE'])

        for sample_id in self.key.index:

            pid,well = self.key.loc[sample_id,['Plate_ID','Well']].values

            smartPrint(f'Fitting {pid}\t{well}',verbose)

            # extract sample
            args_dict = self.key.loc[sample_id,['Well','Plate_ID']].to_dict()
            sample = self.extractGrowthData(args_dict)

            df = sample.time.join(sample.data)
            df.columns = ['Time','OD']

            # create GP object and analyze
            gm = GrowthModel(df=df,
                             baseline=sample.key.OD_Offset.values,
                             ARD=False,heteroscedastic=False,nthin=nthin,logged=self.mods.logged)

            curve = gm.run(name=sample_id)

            mse_df.loc[sample_id,:] = curve.compute_mse()
            diauxie_dict[sample_id] = curve.params.pop('df_dx')
            gp_params.loc[sample_id,:] = curve.params

            # passively save data, manipulation occurs below (input OD, GP fit, & GP derivative)
            if store:
                data_ls.append(curve.data())

        smartPrint('',verbose)

        diauxie_df = mergeDiauxieDfs(diauxie_dict)

        # record results in object's key
        self.key = self.key.join(gp_params)
        self.key = self.key.join(mse_df)
        self.key = pd.merge(self.key,diauxie_df,on='Sample_ID')

        # check quality of fit with K_Error
        self.compute_k_error()

        # plotting needs transformed (or real) OD & GP fit, & may need GP derivative, save all as obejct attributes
        if store:
            self.gp_data = pd.concat(data_ls).reset_index(drop=True)

        return None

