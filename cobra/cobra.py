''' 
======================================================================================================================
---------------------------------------------------------  COBRA  ----------------------------------------------------
======================================================================================================================
'''
#
import cobra.data_preparation as dpc
import cobra.univariate_selection as us
import cobra.model_selection as ms
#
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class COBRA(object):
    '''
    Wrapper class for all the child classes for easier usage
    ----------------------------------------------------
    Author: Jan Benisek, Python Predictions
    Date: 21/02/2018
    ----------------------------------------------------
    ***PARAMETERS***
    :data_path:                 Path to .csv file which contains the data
    :data_types_path:           Path to .csv files which contains the metadata
    :partition_train:           Size of training set as int <0;1>
    :partition_select:          Size of selection set as int <0;1>
    :partition_valid:           Size of validation set as int <0;1>
    :sampling_1:                Size of sampling of target class
    :sampling_0:                Size of sampling of non-target class
    :discret_nbins:             ???
    :regroup_sign:              Significance level for regrouping categorical variables
    :rseed:                     Random seed for reproducibility (partitioning). None or a number
    
    ***ATTRIBUTES***
    :_partition_dict:           Dict with partitioned DFs X/Y train/selection/validation
    ----------------------------------------------------
    __init__: contains variables which are established with the object. 
              If some of them is changed, then the whole process must be redone(call the class again),
              because the model comparison wont make sense
    transform: For the reasons before, transform has no parameters
    fit_univariate: there I can change stuff when trying different modelling ideas
                    i.e. what variables will I get if AUC threshold is changed
    fit_model: Here I want try many things, so the parametes are changeble in the method.
    '''
    
    def __init__(self, 
                 data_path, 
                 data_types_path, 
                 partition_train=0.5, 
                 partition_select=0.3, 
                 partition_valid=0.2,
                 sampling_1=1, 
                 sampling_0=1, 
                 discret_nbins=5, 
                 regroup_sign=0.001,
                 rseed=None):
        
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
        
    
    def transform(self):
        '''
        Method transforms given csv
        ----------------------------------------------------
        only self
        ---------------------------------------------------- 
        '''
        dtrans = dpc.DataPreparation(self.data_path,
                                     self.data_types_path,
                                     self.partition_train,
                                     self.partition_select,
                                     self.partition_valid,
                                     self.sampling_1,
                                     self.sampling_0,
                                     self.discret_nbins,
                                     self.regroup_sign,
                                     self.rseed)

        df_trans = dtrans.transform()
        
        return df_trans
        
    
    def fit_univariate(self, df_t, preselect_auc=0.53, preselect_overtrain=5):
        '''
        Method transforms given csv
        Returns univariate selection and correlation matrix
        ----------------------------------------------------
        df_t: dataframe with transformed data
        ---------------------------------------------------- 
        '''
        
        unisel = us.UnivariateSelection(preselect_auc, 
                                        preselect_overtrain)
        df_sel, df_corr = unisel.fit(df_t)
        
        return df_sel, df_corr
    
    def fit_model(self, df_t, df_us, modeling_nsteps=30, forced_vars=None, excluded_vars=None, name=None):
        '''
        Method fits and finds best model
        Returns dataframe with all the info - forward selection, AUC, importance...
        ----------------------------------------------------
        df_t: dataframe with transformed data
        df_us: dataframe with univariate selection
        modeling_nsteps: how many steps in modelling
        forced_vars: list with variables to be forced in the model
        excluded_vars: list with variables to be excluded in the model
        name: name of the model
        ---------------------------------------------------- 
        '''
        modsel = ms.ModelSelection()
        
        df_models = modsel.fit(df_t, 
                               df_us,
                               modeling_nsteps=modeling_nsteps,
                               forced_vars=forced_vars,
                               excluded_vars=excluded_vars,
                               name=name)
        
        self._partition_dict = modsel._partition_dict
        
        return df_models
    
    '''
    ====================================================================
    ================  STATIC METHODS FOR VISUALIZATION  ================
    ====================================================================
    '''
    @staticmethod
    def plotPredictorQuality(df, dim=(12,8)):
        '''
        Method plots Univarite quality of predictors
        Returns plot
        ----------------------------------------------------
        df: dataframe with univariate selection
        dim: tuple with width and lentgh of the plot
        ---------------------------------------------------- 
        '''
        plt.style.use('seaborn-darkgrid')
        
        #----------------------------------
        #------  Prepare the data  --------
        #----------------------------------
        df_uq = df[['variable','AUC train','AUC selection']][df['preselection'] == True].sort_values(by='AUC train', ascending=False)
        df_uq.columns = ['variable name','AUC train','AUC selection']
        df_uq = pd.melt(df_uq, id_vars=['variable name'], value_vars=['AUC train', 'AUC selection'], var_name='partition', value_name='AUC')
        
        #----------------------------------
        #-------  Plot the bars  ----------
        #----------------------------------
        fig, ax = plt.subplots(figsize=dim)
        
        ax = sns.barplot(x="AUC", y="variable name", hue="partition", data=df_uq)
        ax.set_title('Univariate Quality of Predictors')
        plt.show()
        
    @staticmethod
    def plotCorrMatrix(df, dim=(12,8)):
        '''
        Method plots Correlation matrix among predictors
        Returns plot
        ----------------------------------------------------
        df: dataframe with correlation data
        dim: tuple with width and lentgh of the plot
        ---------------------------------------------------- 
        '''
        fig, ax = plt.subplots(figsize=dim)
        ax = sns.heatmap(df, cmap='Blues') 
        ax.set_title('Correlation Matrix')
        plt.show()
    
    @staticmethod
    def plotIncidence(df, variable, dim=(12,8)):
        '''
        Method plots Incidence plot on train partition
        Returns plot
        ----------------------------------------------------
        df: dataframe with cleaned, binned, partitioned and prepared data
        variable: variable for which the incidence plot will be shown`
        dim: tuple with width and lentgh of the plot
        ---------------------------------------------------- 
        '''
        plt.style.use('seaborn-darkgrid')
        
        #----------------------------------
        #------  Prepare the data  --------
        #----------------------------------
        #Set up the variable and dataframe
        var_prefix = 'B_' + variable
        df_plt = df[['TARGET', var_prefix]][df['PARTITION'] == 'train'].copy()
        
        #Aggregate the data
        avg_inc_rate = df_plt['TARGET'].mean()
        
        aggregations = {
                        'bin_inc_rate': 'mean',
                        'bin_size': 'count'
                        }
        df_plt = df_plt.groupby(var_prefix, as_index=False)['TARGET'].agg(aggregations)
        df_plt['avg_inc_rate'] = avg_inc_rate
        
        #----------------------------------
        #-----  Plot the incidence  -------
        #----------------------------------
        fig, ax = plt.subplots(figsize=dim)
        ##First Axis
        #Bin size
        y_pos = np.arange(len(df_plt[var_prefix]))
        plt.bar(y_pos, df_plt['bin_size'].values.tolist(), align='center', color="cornflowerblue")
        plt.xticks(y_pos, df_plt[var_prefix])
        plt.ylabel('Bin Size')
        plt.xlabel(variable + ' Bins')
        
        ##Second Axis
        ax2 = ax.twinx()
        #incidence rate per bin
        plt.plot(df_plt['bin_inc_rate'], color="darkorange", marker=".", markersize=20, linewidth=3, label='incidence rate per bin')
        plt.plot(df_plt['avg_inc_rate'], color="dimgrey", linewidth=4, label='average incidence rate')
        ax2.plot(np.nan, "cornflowerblue", linewidth=6, label = 'bin size') #dummy line to have label on second axis from first
        ax2.set_yticklabels(['{:3.1f}%'.format(x*100) for x in ax2.get_yticks()])
        plt.ylabel('Incidence')
        
        ##Set Axis
        sns.despine(ax=ax, right=True, left=True)
        sns.despine(ax=ax2, left=True, right=False)
        ax2.spines['right'].set_color('white')
        
        #remove white line from second grid axes
        #the white lines are reguler, Spyder sometimes fails to visualize it (try to export the pic!)
        ax2.grid(False)
        
        ##Description
        fig.suptitle('Incidence Plot - ' + variable, fontsize=20)
        ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=1, mode="expand", borderaxespad=0.)
        plt.show()
    
    @staticmethod
    def plotAUC(df, dim=(12,8)):
        '''
        Method plots AUC for train/selection/validation and number of selected variables
        Returns plot
        ----------------------------------------------------
        df: dataframe with models performance
        dim: tuple with width and lentgh of the plot
        ---------------------------------------------------- 
        AUC on optimal number of vars
        '''
        plt.style.use('seaborn-darkgrid')
        
        #----------------------------------
        #------  Prepare the data  --------
        #----------------------------------
        df_plt = df[['last_var_added','auc_train','auc_selection','auc_validation']]
        df_plt.columns = ['variable name', 'AUC train','AUC selection','AUC validation']
        
        #----------------------------------
        #--------  Plot the AUC  ----------
        #----------------------------------
        fig, ax = plt.subplots(figsize=dim)
        
        plt.plot(df_plt['AUC train'], marker=".", markersize=20, linewidth=3, label='AUC train')
        plt.plot(df_plt['AUC selection'], marker=".", markersize=20, linewidth=3, label='AUC selection')
        plt.plot(df_plt['AUC validation'], marker=".", markersize=20, linewidth=3, label='AUC validation')
        #Set xticks
        ax.set_xticks(np.arange(len(df_plt['variable name'])+1))
        ax.set_xticklabels(df_plt['variable name'].tolist(), rotation = 40, ha='right')
        #Make Pretty
        ax.legend(loc='lower right')
        fig.suptitle('Multivariate Model AUC - ' + df.name, fontsize=20)
        plt.ylabel('AUC')
        plt.show()
    
    @staticmethod
    def plotVariableImportance(df, step, dim=(12,8)):
        '''
        Method plots variable importance for given model
        Returns plot
        ----------------------------------------------------
        df: dataframe with models performance
        step: for which model the importance will be shown
        dim: tuple with width and lentgh of the plot
        ---------------------------------------------------- 
        Importance on optimal number of vars
        '''
        plt.style.use('seaborn-darkgrid')
    
        #----------------------------------
        #------  Prepare the data  --------
        #----------------------------------
        #dict_plt = df['importance'].iloc[model_row]
        dict_plt = df['importance'][df['step'] == step]
        df_plt = pd.DataFrame.from_dict(dict_plt.iloc[0], orient='index')
        df_plt.reset_index(level=0, inplace=True)
        df_plt.columns = ['variable name','importance']
        
        #----------------------------------
        #-------  Plot the bars  ----------
        #----------------------------------
        fig, ax = plt.subplots(figsize=dim)
            
        ax = sns.barplot(x="importance", y="variable name", data=df_plt)
        ax.set_title('Variable Importance in model ' + df.name)
        plt.show()
    
    @staticmethod
    def plotCumulatives(model_list, df_trans, dim=(12,8)):
        '''
        Method plots cumulative response and gains in one plot for multiple models
        Returns plot
        ----------------------------------------------------
        model_list: list of tuples with model DF and step number (which model to take from the DF)
        df_trans: dataframe with cleaned, binned, partitioned and prepared data
        dim: tuple with width and lentgh of the plot
        ---------------------------------------------------- 
        Max 5 models is allowed, on train partition
        '''
        plt.style.use('seaborn-darkgrid')
        
        if len(model_list) >5:
            raise ValueError('The maximum number of input models is 5')
    
        colors = ['cornflowerblue','forestgreen','firebrick','darkmagneta','orange']
        
        avg_incidence = df_trans['TARGET'][df_trans['PARTITION'] == 'train'].mean()
        
        #----------------------------------
        #-------  Plot the data  ----------
        #----------------------------------
        fig, (ax_cresp, ax_cgains) = plt.subplots(1, 2, sharey=False, figsize=dim)
        #
        #Cumulative Response
        #
        for i, model in enumerate(model_list):
            #------  Prepare the data  --------
            cum_resp = model[0]['cum_response'][model[0]['step'] == model[1]].tolist()[0]
            #------  Plot line for each model  --------
            ax_cresp.plot(cum_resp, color=colors[i], linewidth=3, label='cumulative response - ' + model[0].name)
            
        ax_cresp.axhline(y=np.round(avg_incidence*100), color="darkorange", linewidth=3, ls="--", label='average incidence rate')
        ax_cresp.set_title('Cumulative Response', fontsize=20)
        #Format axes
        ax_cgains.set_xlim([0,100])
        ax_cgains.set_ylim([0,100])
        #Format ticks
        ax_cresp.set_yticklabels(['{:3.0f}%'.format(x) for x in ax_cresp.get_yticks()])
        ax_cresp.set_xticklabels(['{:3.0f}%'.format(x) for x in ax_cresp.get_xticks()])
        #Legend
        ax_cresp.legend(loc='upper right')
        #
        #Cumulative Gains
        #
        for i, model in enumerate(model_list):
            #------  Prepare the data  --------
            cum_gains = model[0]['cum_gains'][model[0]['step'] == model[1]].tolist()[0]
            #------  Plot line for each model  --------
            ax_cgains.plot(cum_gains, color=colors[i], linewidth=3, label='cumulative gains - ' + model[0].name)
            
        ax_cgains.plot(ax_cgains.get_xlim(), ax_cgains.get_ylim(), linewidth=3, ls="--", color="darkorange", label='random selection')
        ax_cgains.set_title('Cumulative Gains', fontsize=20)
        #Format axes
        ax_cgains.set_xlim([0,100])
        ax_cgains.set_ylim([0,100])
        #Format ticks
        ax_cgains.set_yticklabels(['{:3.0f}%'.format(x) for x in ax_cgains.get_yticks()])
        ax_cgains.set_xticklabels(['{:3.0f}%'.format(x) for x in ax_cgains.get_xticks()])
        #Legend
        ax_cgains.legend(loc='lower right')
        
        #Make pretty
        plt.tight_layout()
        
        plt.show()
        
    @staticmethod
    def plotAUCComparison(model_list, dim=(12,8)):
        '''
        Method plots AUC comarison on train/selection/validation
        Returns plot
        ----------------------------------------------------
        model_list: list of tuples with model DF and step number (which model to take from the DF)
        dim: tuple with width and lentgh of the plot
        ---------------------------------------------------- 
        '''
        plt.style.use('seaborn-darkgrid')

        #----------------------------------
        #------  Prepare the data  --------
        #----------------------------------
        df_plt = pd.DataFrame() 
        for model, step in model_list:
            df_aux = pd.DataFrame(model[['auc_train','auc_selection','auc_validation']][model['step'] == step])
            df_aux['model'] = model.name
            df_plt = pd.concat([df_plt, df_aux])
            
        df_plt.reset_index(inplace=True, drop=True) 
        
        df_plt.columns = ['AUC train','AUC selection','AUC validation','model']
        df_plt = pd.melt(df_plt, id_vars=['model'], value_vars=['AUC train','AUC selection','AUC validation'], 
                         var_name='partition', value_name='AUC')
        
        #----------------------------------
        #-------  Plot the bars  ----------
        #----------------------------------
        fig, ax = plt.subplots(figsize=(12,8))
        
        ax = sns.barplot(x="AUC", y="partition", hue="model", data=df_plt)
        
        ax.set_xlim([0,1.2])
        ax.set_title('AUC comparison')
        plt.show()