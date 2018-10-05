''' 
======================================================================================================================
----------------------------------------------------  DATA PREPARATION  ----------------------------------------------
======================================================================================================================
'''
import math
import numpy as np
import pandas as pd
from scipy import stats

# To allow pandas dataframes to display more columns
pd.set_option("display.max_columns",50)

class DataPreparation(object):
    '''
    Class for DataPreparation
    Loads, clean, partition, binn, regroup,replace (incidence)
    ----------------------------------------------------
    Author: Jan Benisek, Python Predictions
    Date: 14/02/2018
    ----------------------------------------------------
    ***PARAMETERS***
    :partition_train:           Size of training set as int <0;1>
    :partition_select:          Size of selection set as int <0;1>
    :partition_valid:           Size of validation set as int <0;1>
    :sampling_1:                Size of sampling of target class
    :sampling_0:                Size of sampling of non-target class
    :discret_nbins:             ???
    :regroup_sign:              Significance level for regrouping categorical variables
    :rseed:                     Random seed for reproducibility (partitioning). None or a number
    
    ***ATTRIBUTES***
    :_headers_dict:             Dict of 4 lists with header names (object, numeric, bool, other)
    :_partitioning_settings:    Dict with train/sel/valid sets with their size
    :_sampling_settings:        Dict with sampling settings (how many 1's and 0's we will take)
    ---------------------------------------------------- 
    '''
    def __init__(self, data_path, data_types_path, partition_train, partition_select, partition_valid,
                 sampling_1, sampling_0, discret_nbins, regroup_sign, rseed):
        ''' ***PARAMETERS*** '''
        self.data_path = data_path
        self.data_types_path = data_types_path
        self.partition_train = partition_train
        self.partition_select = partition_select
        self.partition_valid = partition_valid
        self.sampling_1 = sampling_1
        self.sampling_0 = sampling_0
        self.discret_nbins = discret_nbins
        self.regroup_sign = regroup_sign
        self.rseed = rseed
        ''' ***ATTRIBUTES*** '''
        # Instance attributes = Each instance has its own version self.XY
        # Not everyone is initialized here
        
        # Class attributes = Shared accross all instances DataPreparation.XY
        DataPreparation._partitioning_settings = {'train':self.partition_train,
                                                  'selection':self.partition_select, 
                                                  'validation':self.partition_valid}
        DataPreparation._sampling_settings = {1:self.sampling_1, 
                                              0:self.sampling_0}
        
        #Set seed for testing
        #partitioning will be affected
        if rseed:
            np.random.seed(rseed)
        

        
    def transform(self):
        '''
        Method transforms given csv
        Returns DF
        ----------------------------------------------------
        data_path: path to the csv with data file to be transformed
        data_types_path: path to the csv with data types of the above csv dataset
        ---------------------------------------------------- 
        '''
        key_clmns = ["ID","TARGET","PARTITION"]
        
        ##Load csv
        df_transformed, df_types = self._loadCSVs(self.data_path, self.data_types_path)
        
        ##Clean headers
        df_transformed = self._cleanHeaders(df_transformed)
        self._getHeaderNames(df_transformed, df_types, key_clmns)
        
        ##Partitioning
        df_transformed = self._addPartitionColumn(df_transformed)
        
        ##Sample
        df_transformed = self._sampling(df_transformed)
        
        ##Preprocessing
        #Continuous 2 bins
        df_cont = self._prepNumVars(df_transformed, key_clmns)
        #Regrouping categoricals
        df_cat = self._prepCatVars(df_transformed, key_clmns)
        #Rename booleans
        df_bool = self._prepBoolVars(df_transformed, key_clmns)
        
        ##Merge together Preprocessing DFs
        df_prep = pd.concat([
                            df_cont, 
                            df_cat[list(set(df_cat.columns) - set(key_clmns))],
                            df_bool[list(set(df_bool.columns) - set(key_clmns))]
                            ],
                            axis=1)
        
        ##Replace groups by incidence rate
        df_inc = self._replaceByIncidenceRate(df_prep, key_clmns)
        
        ##Merge Preprocessing and Incidence DFs
        df_out = pd.concat([
                            df_prep, 
                            df_inc[list(set(df_inc.columns) - set(key_clmns))]
                            ],
                            axis=1)
        
        ##Cleaning
        del df_transformed, df_cont, df_cat, df_bool, df_prep, df_inc
        
        return df_out
        
    def _loadCSVs(self, data_path, data_types_path):
        '''
        Function loads csv and if no datatype csv is present, 
          guesses the datatypes.
        Returns raw DataFrame
        ----------------------------------------------------
        data_path: path to the data csv file
        data_types_path: path to the datatypes csv file
        ---------------------------------------------------- 
		Return also data types? watch out if it is not given!
        '''
        #Loads Data types
        types_exist = True   

        #load data_types
        try:
            df_types = pd.read_csv(data_types_path, header=None)
            df_types.columns = ['variable','data_type']
        except FileNotFoundError:
            types_exist = False
            df_types = pd.DataFrame()
          
        #load data
        df = pd.read_csv(data_path, header=0, sep=None, engine='python')
            
        #change datatypes
        if types_exist: 
            for row in df_types.itertuples(): #0:index, 1:variable, 2:data_type        
                if row[2] == 'int':
                	#Nan is stored as float, hence the dtype.
                	#Won't work when converting to int with nans
                    df[row[1]] = df[row[1]].astype(np.float64)
                if row[2] in ['str', 'bool']:
                    df[row[1]] = df[row[1]].apply(str)
        
        return df, df_types
    
    def _cleanHeaders(self, df):
        '''
        Method cleans headers in given DataFrame.
        Returns cleaned DF
        ----------------------------------------------------
        df: input dataframe to be modified
        ---------------------------------------------------- 
        '''
        #Define functions
        def strip_quot(x_in):
            '''Function to remove quotes from variable names and/or variable values'''
            try:
                x_out = x_in.strip().strip('"').strip("'")
            except:
                x_out=x_in
            return x_out
        
        def lower_upper(x_in):
            '''Function to put 'id' and 'target' variable names in uppercase, 
               all other variable names are put in lowercase'''
            if ((x_in.lower() == 'id')|(x_in.lower() == 'target')):
                x_out = x_in.upper()
            else:
                x_out = x_in.lower()
            return x_out
        
        #Apply functions
        df = df.rename(columns=strip_quot)
        df = df.rename(columns=lower_upper)
        df = df.applymap(strip_quot)
        
        return df
    
    def _getHeaderNames(self, df, _df_types, key_clmns):
        '''
        Method returns lists with header names (int and obj).
        Does not return anything, only initialize the self._headers_dict variable for later use
        ----------------------------------------------------
        df: input dataframe from which the headers are retrieved
        _df_types: dataframe with types
        key_columns: list with colum names of keys
        ---------------------------------------------------- 
        '''
        #Define function
        def get_headers(dataframe,type): 
            '''Function to group variable names based on the data type of the variable'''
            return dataframe.select_dtypes(include=[type]).columns.values
        
        #Get header names into a list
        other_headers = key_clmns[:2]
        
        if len(_df_types) != 0:
            bool_mask = _df_types[_df_types['data_type'] != 'bool']
            try:
                bool_headers = [n for n in _df_types.loc[bool_mask==False,0].values if n not in other_headers]
            except:
                bool_headers = []
        else:
            bool_headers = []
            
        object_headers = [n for n in get_headers(df,'object') if n not in other_headers + bool_headers]
        numeric_headers = [n for n in get_headers(df,'number') if n not in other_headers + bool_headers]
        
        self._headers_dict = {'string':object_headers, 'numeric':numeric_headers, 'bool':bool_headers, 'other':other_headers}
        
    def _addPartitionColumn(self, df):
        '''
        Method shuffle DF and create a column PARTITIONING with train/selection/validation categories
        Returns DF with new column PARTITIONING
        ----------------------------------------------------
        df: input dataframe which will be parittioned
        ---------------------------------------------------- 
        '''
        #Shuffle and sort target
        df = df.iloc[np.random.permutation(len(df))].sort_values(by='TARGET', ascending=False).reset_index(drop=True)
        
        partition = []
        sorted_target=df['TARGET'] #Just the target since it is allready sorted (see above)
        for target in [sorted_target.iloc[0],sorted_target.iloc[-1]]:
            target_length = (sorted_target==target).sum()
            
            for part, size in DataPreparation._partitioning_settings.items():
                partition.extend([part]*math.ceil(target_length*size))
                
        df["PARTITION"] = partition[:len(df)]
        
        return df
    
    def _sampling(self, df):
        '''
        Method takes sample for the dataframe. If no sampling is specified, all data are taken.
        Returns sampled DF.
        ----------------------------------------------------
        df: input dataframe which will be sampled
        ---------------------------------------------------- 
        '''
        drop_index = []
        for target, size in DataPreparation._sampling_settings.items():
            if size < 1:
                sample_length = int(round((df['TARGET']==target).sum() * size))
                
                for part, size in DataPreparation._partitioning_settings.items():
                    part_length = int(round(sample_length * size))
                    drop_index_part = df[(df['TARGET']==target) & (df['PARTITION']==part)].index[part_length:]
                    drop_index.extend(drop_index_part)
                    
        df.drop(drop_index,inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _prepNumVars(self, df, key_columns):
        '''
        Method converts numerical variables into bins. 10 bins is always used
        Returns DF with binned columns
        ----------------------------------------------------
        df: input dataframe (with all variables!)
        key_columns: list with colum names of keys
        ---------------------------------------------------- 
        '''
        
        df_out = df.loc[:,key_columns].copy()
        
        for clmn in self._headers_dict['numeric']:
            result = DataPreparation.__eqfreq(var=df[clmn],
                                   train=df["PARTITION"]=="train",
                                   autobins=True,
                                   nbins=self.discret_nbins,
                                   precision=0,
                                   twobins=True,
                                   # TRUE OPTION STILL PRODUCES ERROR IN SORTNUMERIC function AND SCORING procedure !!!!!!!!!
                                   catchLarge=False) 
            df_out = pd.concat([df_out,result[0]], axis=1)
            
        return df_out
    
    def _prepCatVars(self, df, key_columns):
        '''
        Method regroup categorical variables based on significance.
        If the incidence rate in particular group is not signicantly different 
          from the average incidence rate, then the variable is regrouped 
          (will be pushed to 'Non-significants' category)
        Returns DF with regrouped categorical columns
        ----------------------------------------------------
        df: input dataframe (with all variables!)
        key_columns: list with colum names of keys
        ---------------------------------------------------- 
        '''
        
        df_out = df.loc[:,key_columns].copy()
        
        for clmn in self._headers_dict['string']:
            # We label missing and empty values for categorical variables as 'Missing'
            # Note the interaction with the 'keep' parameter of the regroup function.
            mask = DataPreparation.__maskmissing(df[clmn])
            df.loc[mask,clmn]='Missing'
            # Perform regrouping function
            result = DataPreparation.__regroup(var=df[clmn],
                                target=df.loc[:,'TARGET'],
                                train=df['PARTITION']=='train',
                                pval_thresh=self.regroup_sign,
                                dummy=True,
                                keep='Missing',
                                rename='Non-significants')
            df_out = pd.concat([df_out,result[0]],axis=1)
            
        return df_out
    
    def _prepBoolVars(self, df, key_columns):
        '''
        Method just passes the variables.
          In order to be consistent, there is this special method. Otherwise they could be renamed whenever
        Returns DF with renamed bool variables
        ----------------------------------------------------
        df: input dataframe (with all variables!)
        key_columns: list with colum names of keys
        ---------------------------------------------------- 
        '''
        
        df_out = df.loc[:,key_columns].copy()
        
        def passvar(var):
            var_pass = var.copy()
            var_pass.name = "B_"+var.name
            info = ("Passing "+var.name)
            return var_pass, info
        
        for clmn in self._headers_dict['bool']:
            # We label missing and empty values for boolean variables as 'Missing'
            mask = DataPreparation.__maskmissing(df[clmn])
            df.loc[mask,clmn]='Missing'
            # Perform the passvar function
            result = passvar(var=df[clmn])
            df_out = pd.concat([df_out,result[0]],axis=1)
            
        return df_out
    
    def _replaceByIncidenceRate(self, df, key_columns):
        '''
        Method to replace the groups with average incidence rate (the "secret sauce"). 
            The variables will start with "D_"
        ----------------------------------------------------
        df: input dataframe (with all variables!)
        key_columns: list with colum names of keys
        ---------------------------------------------------- 
        '''
        df_out = df.loc[:,key_columns].copy()
        
        headers_for_incidrep = [h for h in df.columns if ((h not in key_columns) & (h[:2]=="B_"))]
        
        for clmn in headers_for_incidrep:
            # Perform increp function
            result = DataPreparation.__increp(b_var=df[clmn],
                               target=df['TARGET'],
                               train=df['PARTITION']=="train")
            df_out = pd.concat([df_out,result], axis=1)
            
        return df_out
    
    '''
    ====================================================================
    ====================  AUXILIARY STATIC METHODS  ====================
    ====================================================================
    '''
    
    @staticmethod
    def __eqfreq(var, train, autobins=True, nbins=10, precision=0, twobins=True, catchLarge=True):
        '''
        Special method for binning continuous variables into bins
        ----------------------------------------------------
        var: input pd.Serie with continuous columns
        train: mask with rows which belongs to train
        autobins: adapts number of bins
        nbins: number of bins
        precision: precision to form meaningful bins
        twobins: if only two bins are found, iterate to find more
        catchLarge: check when groups are too big
        ---------------------------------------------------- 
        - This function is a reworked version of pd.qcut to satisfy our particular needs
        - Takes for var a continuous pd.Series as input and returns a pd.Series with bin-labels (e.g. [4,6[ )
        - Train takes a series/list of booleans (note: we define bins based on the training set)
        - Autobins reduces the number of bins (starting from nbins) as a function of the number of missings
        - Nbins is the wished number of bins
        - Precision=0 results in integer bin-labels if possible
        - twobins=True forces the function to output at least two bins
        - catchLarge tests if some groups (or missing group) are very large, and if so catches and outputs two groups
        - note: catchLarge makes twobins irrelevant
        '''
    
        # Test for large groups and if one exists pass them with two bins: Large_group,Other
        if catchLarge:
            catchPercentage=1-(1/nbins)
            groupCount = var[train].groupby(by=var[train]).count()
            maxGroupPerc = groupCount.max()/len(var[train])
            missingPerc = sum(var[train].isnull())/len(var[train])
            if maxGroupPerc>=catchPercentage:
                largeGroup = groupCount.sort_values(ascending=False).index[0]
                x_binned = var.copy()
                x_binned.name = 'B_'+var.name
                x_binned[x_binned!=largeGroup]='Other'
                cutpoints=None
                info = (var.name+": One large group, outputting 2 groups")
                return x_binned, cutpoints, info
            elif missingPerc>=catchPercentage:
                x_binned = var.copy()
                x_binned.name = 'B_'+var.name
                x_binned[x_binned.isnull()]='Missing'
                x_binned[x_binned!='Missing']='Other'
                cutpoints=None
                info = (var.name+": One large missing group, outputting 2 groups")
                return x_binned, cutpoints, info
        # Adapt number of bins as a function of number of missings
        if autobins:
            length = len(var[train])
            missing_total = var[train].isnull().sum()
            missing_perten = missing_total/length*10
            nbins = max(round(10-missing_perten)*nbins/10 ,1)
        # Store the name and index of the variable
        name = var.name
        series_index = var.index
        # Transform var and train to a np.array and list respectively, which is needed for some particular function&methods
        x = np.asarray(var)
        train = list(train)
        # First step in finding the bins is determining what the quantiles are (named as cutpoints)
        # If the quantile lies between 2 points we use lin interpolation to determine it
        cutpoints = var[train].quantile(np.linspace(0,1,nbins+1),interpolation = 'linear')
        # If the variable results only in 2 unique quantiles (due to skewness) increase number of quantiles until more than 2 bins can be formed
        if twobins:
            extrasteps = 1
            # Include a max. extrasteps to avoid infinite loop
            while (len(cutpoints.unique())<=2) & (extrasteps<20):
                cutpoints = var[train].quantile(np.linspace(0,1,nbins+1+extrasteps),interpolation = 'linear')
                extrasteps+=1
        # We store which rows of the variable x lies under/above the lowest/highest cutpoint 
        # Without np.errstate(): x<cutpoints.min() or x>cutpoints.max() can give <RuntimeWarning> if x contains nan values (missings)
        # However the function will result in False in both >&< cases, which is a correct result, so the warning can be ignored
        with np.errstate(invalid='ignore'):
            under_lowestbin = x < cutpoints.min()
            above_highestbin= x > cutpoints.max()
    
    
        def _binnedx_from_cutpoints(x, cutpoints, precision, under_lowestbin, above_highestbin):
        ### Attributes the correct bin ........................
        ### Function that, based on the cutpoints, seeks the lowest precision necessary to have meaningful bins
        ###  e.g. (5.5,5.5] ==> (5.51,5.54]
        ### Attributes those bins to each value of x, to achieve a binned version of x   
            
            # Store unique cutpoints (e.g. from 1,3,3,5 to 1,3,5) to avoid inconsistensies when bin-label making
            # Indeed, bins [...,1], (1,3], (3,3], (3,5], (5,...] do not make much sense
            # While, bins  [...,1], (1,3],        (3,5], (5,...] do make sense
            unique_cutpoints = cutpoints.unique()
            # If there are only 2 unique cutpoints (and thus only one bin will be returned), 
            # keep original values and code missings as 'Missing'
            if len(unique_cutpoints) <= 2:
                cutpoints = None
                x_binned = pd.Series(x)
                x_binned[x_binned.isnull()] = 'Missing'
                info = (var.name+": Only one resulting bin, keeping original values instead")
                return x_binned, cutpoints, info
            # Store info on whether or not the number of resulting bins equals the desired number of bins
            elif len(unique_cutpoints) < len(cutpoints):
                info = (var.name+": Resulting # bins < whished # bins")
            else:
                info = (var.name+": Resulting # bins as desired")
            # Finally, recode the cutpoints (which can have doubles) as the unique cutpoints
            cutpoints = unique_cutpoints
            
            # Store missing values in the variable as a mask, and create a flag to test if there are any missing in the variable
            na_mask = np.isnan(x)
            has_nas = na_mask.any()
            # Attribute to every x-value the index of the cutpoint (from the sorted cutpoint list) which is equal or higher than
            # the x-value, effectively encompasing that x-value.
            # e.g. for x=6 and for sorted_cutpoint_list=[0,3,5,8,...] the resulting_index=3    
            ids = cutpoints.searchsorted(x, side='left')
            # x-values equal to the lowest cutpoint will recieve a ids value of 0
            # but our code to attribute bins to x-values based on ids (see end of this subfunction) requires a min. value of 1
            ids[x == cutpoints[0]] = 1
            # Idem as previous: x-values below the lowest cutpoint should recieve a min. value of 1
            if under_lowestbin.any():
                ids[under_lowestbin] = 1
            # Similar as previous: x-values above the highest cutpoint should recieve the max. allowed ids
            if above_highestbin.any():
                max_ids_allowed = ids[(above_highestbin == False) & (na_mask==False)].max()
                ids[above_highestbin] = max_ids_allowed
            # Maximal ids can now be defined if we neglect ids of missing values
            max_ids = ids[na_mask==False].max()
            
            # Based on the cutpoints create bin-labels
            # Iteratively go through each precision (= number of decimals) until meaningful bins are formed
            # If theoretical bin is ]5.51689,5.83654] we will prefer ]5.5,5.8] as output bin
            increases = 0
            original_precision = precision
            while True:
                try:
                    bins = _format_bins(cutpoints, precision)
                except ValueError:
                    increases += 1
                    precision += 1
                    #if increases >= 5:
                        #warnings.warn("Modifying precision from "+str(original_precision)+" to "+str(precision)+" to achieve discretization")
                        #print("Modifying precision from "+str(original_precision)+" to "+str(precision)+" to achieve discretization")
                else:
                    break
            
            # Make array of bins to allow vector-like attribution
            bins = np.asarray(bins, dtype=object)
            # If x has nas: for each na-value, set the ids-value to max_ids+1
            # this will allow na-values to be attributed the highest bin which we define right below
            if has_nas:
                np.putmask(ids, na_mask, max_ids+1)
            # The highest bin is defined as 'Missing'
            bins = np.append(bins,'Missing')
            # ids-1 is used as index in the bin-labels list to attribute a bin-label to each x. Example:
            # x=6   sorted_cutpoint_list=[0,3,5,8,...]   ids=3   levels=[[0,3],(3,5],(5,8],...]
            # The correct bin level for x is (5,8] which has index 2 which is equal to the ids-1
            x_binned = bins[ids-1]
            return x_binned, cutpoints, info
            
    
        def _format_bins(cutpoints, prec):
        # Based on the quantile list create bins. Raise error if values are similar within one bin.
        # On error _binnedx_from_cutpoints will increase precision
            
            fmt = lambda v: _format_label(v, precision=prec)
            bins = []
            for a, b in zip(cutpoints, cutpoints[1:]):
                fa, fb = fmt(a), fmt(b)
                
                if a != b and fa == fb:
                    raise ValueError('precision too low')
                    
                formatted = '(%s, %s]' % (fa, fb)
                bins.append(formatted)
            
            bins[0] = '[...,' + bins[0].split(",")[-1]
            bins[-1] = bins[-1].split(",")[0] + ',...]'
            return bins
    
    
        def _format_label(x, precision):
        # For a specific precision, returns the value formatted with the appropriate amount of numbers after comma and correct brackets
        
            if isinstance(x,float):
                frac, whole = np.modf(x)
                sgn = '-' if x < 0 else ''
                whole = abs(whole)
                if frac != 0.0:
                    val = '{0:.{1}f}'.format(frac, precision)
                    val = _trim_zeros(val)
                    if '.' in val:
                        return sgn + '.'.join(('%d' % whole, val.split('.')[1]))
                    else: 
                        if '0' in val:
                            return sgn + '%0.f' % whole
                        else:
                            return sgn + '%0.f' % (whole+1)
                else:
                    return sgn + '%0.f' % whole
            else:
                return str(x)
    
    
        def _trim_zeros(x):
        # Removes unnecessary zeros and commas
            while len(x) > 1 and x[-1] == '0':
                x = x[:-1]
            if len(x) > 1 and x[-1] == '.':
                x = x[:-1]
            return x
        
        x_binned, cutpoints, info = _binnedx_from_cutpoints(x, cutpoints, precision=precision, under_lowestbin=under_lowestbin, above_highestbin=above_highestbin)
        x_binned = pd.Series(x_binned, index=series_index, name="B_"+name)
        return x_binned, cutpoints, info
    
    @staticmethod
    def __maskmissing(df):
        '''
        Method checks which values of a var are empty strings or null values
        Returns DF mask
        ----------------------------------------------------
        df: input dataframe
        ---------------------------------------------------- 
        '''
        # Check if values are null
        crit1 = df.isnull()
        # Check if values are empty strings
        modvar = pd.Series([str(value).strip() for value in df])
        crit2 = modvar==pd.Series(['']*len(df))
        return crit1 | crit2
    
    @staticmethod
    def __regroup(var,target,train,pval_thresh=0.01,dummy=True,keep='Missing',rename='Other'):
        '''
        Method regroups categorical variables
        Returns DF mask
        ----------------------------------------------------
        var: input pd.Serie with cat column
        target: pd.Serie with target variable
        train: pd.Serie with parition variable
        pval_thresh: threshold for regrouping
        dummy: scale of booleans (?)
        keep: keep specific groups (?)
        rename: rename the insignificant category
        ---------------------------------------------------- 
        - Each group is tested with a chi² for relevant incidence differences in comparison to a rest-group
        - The rest group has the size of the remaining groups and an 'overall average incidence' (if dummy=True) or 
        - remaining groups average incidence' (if dummy=False)
        - Groups with a pvalue above the threshold are relabled to a single group
        '''
        
        # Define the chi² test condition
        # Groups that do not meet the condition are not analyzed and will be unconditionally relabled
        def _chi2cond_(var=var,target=target,train=train):
            varcounts = var[train].groupby(by=var).count()
            train_inc = target[train].sum()/len(target[train])
            factor = max(train_inc, 1-train_inc)
            analyze_mask = (varcounts*factor)>5
            analyze_groups = analyze_mask.index[analyze_mask].values
            return analyze_groups
        
        # Compute overal incidence mean
        incidence_mean = target[train].mean()
        # Create container of which groups will be kept, compared to the groups which will be relabled
        keepgroups = []
        # Cycle and test each group that meets the chi² condition
        for group in _chi2cond_():
            # Container for target 0/1 observations of the group under scrutiny
            obs_group = []
            # Counts of the target 0/1 occurences for the group under scrutiny
            obs_group.append(((target[train]==0)&(var[train]==group)).sum())
            obs_group.append(((target[train]==1)&(var[train]==group)).sum())
            obs_group = np.array(obs_group)
            # Container for target 0/1 observations of the remaining groups together
            obs_other = []
            # Counts of the target 0/1 occurences for the remaining groups together
            obs_other.append(((target[train]==0)&(var[train]!=group)).sum())
            obs_other.append(((target[train]==1)&(var[train]!=group)).sum())
            obs_other = np.array(obs_other)
            # If dummy=True, we scale the two groups of target 0/1 occurences such that the incidence is equal to the overall incidence
            # The size of the two groups of target 0/1 occurences is still equal to the size of the remaining groups
            if dummy:
                obs_other_size = obs_other.sum()
                obs_other[0]=(1-incidence_mean)*obs_other_size # 0(1) index coincides with target = 0(1)
                obs_other[1]=(  incidence_mean)*obs_other_size
            obs = np.array([obs_group,obs_other])
            # Place at least 1 observation to avoid error in chi2 test
            obs[obs==0] = 1
            # Perform chi² test
            pval = stats.chi2_contingency(obs, correction=False)[1]
            # If pval outperforms threshold, append the group in the keepgroups list
            if pval<=pval_thresh:
                keepgroups.append(group)
            #elif group==keep:
            #    keepgroups.append(group)
        # If the specific group to be kept (e.g. 'Missing') didn't pass the test, append it to the keepgroups list
        if keep not in keepgroups:
            keepgroups.append(keep)
        # Makes a list of all groups not in the keepgroups list
        regroup_mask = [val not in keepgroups for val in var.values]
        var_regroup = var.copy()
        # Rename those groups
        var_regroup[regroup_mask] = rename
        var_regroup.name = "B_"+var.name
        info = (var.name+": from "+str(len(var.unique()))+" to "+str(len(var_regroup.unique())))
        return var_regroup, info
    
    
    @staticmethod
    def __increp(b_var, target, train):    
        '''
        Method for incidence replacement
        Returns replaced pd.Serie
        ----------------------------------------------------
        b_var: input pd.Serie to be replaced
        target: pd.Serie with target variable
        train: pd.Serie with parition variable
        ---------------------------------------------------- 
        '''
        
        #get variable name
        name = b_var.name
        #get overall incidence 
        incidence_mean = target[train].mean()
        #get incidence per group
        incidences = target[train].groupby(b_var).mean()
        #construct dataframe with incidences
        idf = pd.DataFrame(incidences).reset_index()
        #get values that are in the data but not in the labels
        bin_labels = incidences.index
        newgroups = list(set(b_var.unique()) ^ set(bin_labels))
        #if newgroups, add mean incidence to incidence dataframe for each new group
        if len(newgroups)>0:
            #make dataframe:
            ngdf = pd.DataFrame(newgroups)
            ngdf.columns = [name]
            ngdf["TARGET"] = incidence_mean
            #dataframe with incidences:    
            idf = idf.append(ngdf)
        #dataframe with the variable
        vdf = pd.DataFrame(b_var)
        #discretized variable by merge
        d_var = pd.merge(vdf,idf,how='left',on=name)["TARGET"]
        return pd.Series(d_var, name="D_"+name[2:]) 
    





















    



