''' 
======================================================================================================================
--------------------------------------------------  MODEL SELECTION  --------------------------------------------
======================================================================================================================
'''
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

class ModelSelection(object):
    '''
    Class for Model Selection
    Finds best model using forward selection
    ----------------------------------------------------
    Author: Jan Benisek, Python Predictions
    Date: 19/02/2018
    ----------------------------------------------------
    ***PARAMETERS***
    :modeling_nsteps:             Size of training set as int <0;1>
    :forced_vars:                 Force variables to be used in forward selection
    :excluded_vars:               List with variables to be excluded
    
    ***ATTRIBUTES***
    :_partition_dict:             Dict with partitioned DFs X/Y train/selection/validation
    :_optimal_nvars:              Optimal number of variables
    ---------------------------------------------------- 
    '''
    
    def __init__(self):
        pass
        
    def fit(self, df_trans, df_unisel, modeling_nsteps, forced_vars, excluded_vars, name):
        '''
        Method fits (=performs) Model Selection
        Returns DF with model performance and list
        ----------------------------------------------------
        df_trans: transformed dataset
        df_unisel: dataframe with univariate selection
        modeling_nsteps: how many variables will be used for modelling
        forced_vars: variables forced to be used in the modelling, list
        excluded_vars: variables to be excluded
        ---------------------------------------------------- 
        '''
        self.modeling_nsteps = modeling_nsteps
        
        ##Create partition
        self._partition_dict = self._getTrainSelectValidXY(df_trans)
        
        ##Perform forward selection
        df_fsel = self._forwardSelection(df_unisel, forced_vars, excluded_vars)
        
        ##Find optimal number of variables and add it into df_fsel
        self._optimalModelCriterion(df_fsel)
        
        ##Cumulative respone/gain and adds it into df_fsel
        self._cumulatives(df_fsel)
        
        ##Calclates importance and adds it into df_fsel
        self._calcImportance(df_fsel)
        
        ##Give name
        df_fsel.name = name
        
        return df_fsel
        
    def _getTrainSelectValidXY(self, df):
        '''
        Method split given DF into train/test/validation set in respect to X and Y.
        Returns dictionary with DFs
        ----------------------------------------------------
        df: transformed dataset
        ---------------------------------------------------- 
        '''
        
        dvars = [n for n in df.columns if n[:2] == 'D_']
        
        mask_train = df['PARTITION']=="train"
        mask_selection = df['PARTITION']=="selection"
        mask_validation = df['PARTITION']=="validation"
        
        y_train = df.loc[mask_train,'TARGET']
        y_selection = df.loc[mask_selection,'TARGET']
        y_validation = df.loc[mask_validation,'TARGET']
        
        x_train = df.loc[mask_train,dvars]
        x_selection = df.loc[mask_selection,dvars]
        x_validation = df.loc[mask_validation,dvars]
        
        dict_out = {'y_train':y_train, 'y_selection':y_selection, 'y_validation':y_validation, 
                    'x_train':x_train, 'x_selection':x_selection, 'x_validation':x_validation}
        
        return dict_out
    
    def _forwardSelection(self, df_sel, forced_vars, excluded_vars):
        '''
        Method performs forward selection
        Returns DF with performance
        ----------------------------------------------------
        df_sel: DF with selection from Univariate Selection
        forced_vars: list with varibels forced to be in the forward selection
        excluded_vars: list with variables to be excluded
        ---------------------------------------------------- 
        '''    
        df_best_models = pd.DataFrame(columns=["modelfit",
                                               "predictor_names",
                                               "predictor_lastadd",
                                               "auc_train",
                                               "auc_selection",
                                               "auc_validation",
                                               "pred_train",
                                               "pred_selection",
                                               "pred_validation"])
        df_sel_aux = df_sel.copy()
        #Set variables to be used for the selection
        if forced_vars:
            #Flag those which user chose to be in the model
            df_sel_aux['forced'] = np.where(df_sel_aux['variable'].isin(forced_vars), True, False)
            #Sort the variables so the one which user chose are first, then the ones from preselection
            # and those are sorted by train AUC
            df_sel_aux.sort_values(['forced','preselection', 'AUC train'], ascending=[False, False, False], inplace=True)
            #Flag for which variables will be used
            df_sel_aux['final_vars'] = np.where(((df_sel_aux['preselection'] == True) | (df_sel_aux['forced'] == True)),
                                                  True,
                                                  False)
            selected_variables = df_sel_aux['variable'][df_sel_aux['final_vars'] == True].tolist()
        else:
            df_sel_aux['forced'] = False
            df_sel_aux.sort_values(['preselection', 'AUC train'], ascending=[False, False], inplace=True)
            selected_variables = df_sel_aux['variable'][df_sel_aux['preselection'] == True].tolist()
            
        if excluded_vars:
            selected_variables = [var for var in selected_variables if var not in excluded_vars]
        
        
        selected_variables = ['D_' + var for var in selected_variables]
        
        n_steps = min(self.modeling_nsteps,len(selected_variables))
        predictors = []
        
        for i in range(1,n_steps+1):
            try:
                result = self.__forward(current_predictors=predictors,
                                        pool_predictors=selected_variables,
                                        positive_only=True)
                df_best_models.loc[i] = result
                predictors = df_best_models.loc[i].predictor_names
                
            except Exception as e:
                if e.args[-1]=='NormalStop':
                    pass
                else:
                    break
                 
        return df_best_models
    
    def _optimalModelCriterion(self, df):
        '''
        Method finds optimal number of variables in given model
        Returns nothing, sets attribute self._optimal_nvars and add a column to the input DF
        ----------------------------------------------------
        df: df with best models 
        ---------------------------------------------------- 
        Here I kept all the functions inside, since they are shorter
        '''    
        
        def comparefit(p,g=2):
            # We fit a second degree (g=2) polyline through our auccurve 
            # This serves as a starting base for finding our optimal stopping point
            z = np.polyfit(p.index, p, g)
            f = np.poly1d(z)
            y_new = f(p.index)
            return pd.Series(y_new,index=p.index)
        
        def slopepoint(p,p_fit,thresh_ratio=0.2):
            # We take the polyline from comparefit and look for the point of which the slope lies just below some percentage of the max. slope
            slopes = [p_fit[i+1]-p_fit[i] for i in range(1,len(p_fit))]
            slopes = pd.Series(slopes, index=range(1,len(p_fit)))
            thresh = slopes.max()*thresh_ratio
            p_best_index = (slopes[slopes>thresh])[-1:].index
            p_best = p.loc[p_best_index]
            return p_best
        
        def moveright(p,p_fit,p_best,n_steps=5,dampening=0.01):
            # We look nsteps right on the polyline (starting from the slopepoint) and take the point with largest difference with real line
            # We move to that point if that difference is larger than some multiplication of the difference at the slopepoint
            # That multiplication gets larger as current the current difference gets smaller with a certain amount of dampening. 
            # The rationale behind this is as follows: 
            #  if the current difference is already large than the larger difference will definitely be noteworthy
            #  if however the current difference is near zero than there needs to be much larger difference to be noteworthy
            in_index = p_best.index.values[0]
            lower = (in_index-1)
            upper = (in_index+n_steps-1)
            p_diff = p[lower:upper]-p_fit[lower:upper]
            out_index = p_diff.idxmax() 
            factor = 1/abs(p_diff[in_index])
            if (p_diff[out_index]>p_diff[in_index]+(abs(p_diff[in_index])*factor*dampening)):
                p_best_new = pd.Series(p[out_index],index=[out_index])
            else:
                p_best_new = p_best
            return p_best_new
        
        def moveleft(p,p_fit,p_best,rangeloss=0.1, diffshare=0.8): #diff_min=0.005):
            # Starting from whatever point we end up with (either the slopepoint or a move to the right)
            # We look left on the polyline and take the point for which the real line is largest (current point included)
            # We move left if we stay within [a specific % loss of range] AND [a minimum % of current difference]
            # i.e. we don't won't to go to low compared to the overall real line
            #      and we don't won't to move to a point that does not make a significant increase in AUC (i.e. difference between polyline and real line)
            p_left = p[:p_best.index.values[0]]
            p_best = p_left[p_left==p_left.max()]
            p_diff = p-p_fit
            p_range = p.max()-p.min()
            s = p[(p >= p_best.values[0]-(rangeloss*p_range)) 
                  & (p.index <= p_best.index.values[0]) 
                  & (p_diff>=diffshare*p_diff[p_left.index[-1]])
                 ]
            p_best_new = s[s.index == s.index.values.min()]
            return p_best_new
        
        points = df['auc_selection']
        points_fit = comparefit(p=points, g=2)
        points_slope = slopepoint(p=points, p_fit=points_fit, thresh_ratio=0.2)
        points_right = moveright(p=points, p_fit=points_fit, p_best=points_slope, n_steps=5, dampening=0.01)
        points_left = moveleft(p=points, p_fit=points_fit, p_best=points_right, rangeloss=0.1, diffshare=0.8)
        
        #Set the variable
        self._optimal_nvars = points_left.index.values[0]
        
        #Add it to the models dataframe
        df['opt_var'] = False
        df.loc[df.index == self._optimal_nvars, 'opt_var'] = True
        
    def _cumulatives(self, df):
        '''
        Method calculates cumulative gains/response
        Returns nothing, adds cgains/response into the dataframe
        ----------------------------------------------------
        df: df with best models 
        ---------------------------------------------------- 
        '''    
        
        def cumulatives(y,yhat,perc_as_int=False,dec=2):
            nrows = len(y)
            npositives = y.sum()
            y_yhat = pd.DataFrame({"y":y, "yhat":yhat}).sort_values(by='yhat', ascending=False).reset_index(drop=True)
            cresp = []
            cgains = [0]
            for stop in (np.linspace(0.01,1,100)*nrows).astype(int):
                cresp.append(round(y_yhat.loc[:stop,'y'].mean()*max(100*int(perc_as_int),1),dec))
                cgains.append(round(y_yhat.loc[:stop,'y'].sum()/npositives*max(100*int(perc_as_int),1),dec))
            return cresp,cgains
        
        cresp_all = [None]
        cgains_all = [None]
        for i in range(1,len(df)+1):
            out = cumulatives(y=self._partition_dict['y_selection'],
                              yhat=df['pred_selection'][i][:,0],
                              perc_as_int=True,
                              dec=2)
            cresp_all.append(out[0]) 
            cgains_all.append(out[1])
         
        #Add it to the models dataframe
        df['cum_response'] = cresp_all[1:]
        df['cum_gains'] = cgains_all[1:]
        
    def _calcImportance(self, df):
        '''
        Method calculates importance of each variable
        Returns nothing, adds importnace into the dataframe
        ----------------------------------------------------
        df: df with best models 
        ---------------------------------------------------- 
        ''' 
        
        def getImportance(model):            
            predictors = [pred[2:] for pred in model.predictor_names]
            importance_dict = {}
            for predictor in predictors: 
                pearsonr = stats.pearsonr(self._partition_dict['x_selection'].loc[:,'D_'+predictor].values, model.pred_selection[:,0])
                importance_dict.update({predictor:np.round(pearsonr[0])})
            return importance_dict
        
        importance_all=[None]
        for i in df.index:
            importance_all.append(getImportance(df.loc[i,:]))
            
        #Add it to the models dataframe
        df['importance'] = importance_all[1:]
                 
    '''
    ====================================================================
    =======================  AUXILIARY METHODS  ========================
    ====================================================================
    '''        
    def __buildModel(self, predictors_subset):
        '''
        Method buils Logistic Regression with the given subset of variables.
        Returns trained model and fit
        ----------------------------------------------------
        predictors_subset: list of variables to be used in the model
        ---------------------------------------------------- 
        '''
        
        # Fit model on predictors_subset and retrieve performance metric
        model = LogisticRegression(fit_intercept=True, C=1e9, solver = 'liblinear')
        modelfit = model.fit(y=self._partition_dict['y_train'], X=self._partition_dict['x_train'][predictors_subset])
        
        # Position of the TARGET==1 class
        pos = [i for i,h in enumerate(modelfit.classes_) if h==1]
        # Prediction probabilities for the TARGET==1
        y_pred = modelfit.predict_proba(self._partition_dict['x_train'][predictors_subset])[:,pos]
        auc = metrics.roc_auc_score(y_true=self._partition_dict['y_train'], y_score=y_pred)
        
        model_score_dict = {"modelfit":modelfit,
                            "auc":auc,
                            "predictor_names":predictors_subset,
                            "predictor_lastadd":predictors_subset[-1]}
        
        return model_score_dict
    
    def __calculateAUC(self, df):
        '''
        Method for computing AUC of all sets (train, selection & validation)
        Returns DF with the AUC
        ----------------------------------------------------
        df: df without AUC
        partition: dictionary with DFs of partitions (train/sel/valid X/Y)
        ---------------------------------------------------- 
        '''
        df = df[:]
        for x,y,part in [(self._partition_dict['x_train'], self._partition_dict['y_train'], 'train'),
                         (self._partition_dict['x_selection'], self._partition_dict['y_selection'], 'selection'),
                         (self._partition_dict['x_validation'], self._partition_dict['y_validation'], 'validation')]:
            pos = [i for i,h in enumerate(df.modelfit.classes_) if h==1]
            y_pred = df.modelfit.predict_proba(x[df['predictor_names']])[:,pos]
            df["auc_"+part] = metrics.roc_auc_score(y_true=y, y_score=y_pred)
            df["pred_"+part] = y_pred
            
        return df
    
    def __forward(self, current_predictors, pool_predictors, positive_only=True):
        '''
        Method for forward selection
        Returns best model
        ----------------------------------------------------
        current_predictors: current predictors to be used
        pool_predictors: remaining predictors
        positive_only: predictors must be positivee (???)
        ---------------------------------------------------- 
        '''
        #Pull out predictors we still need to process
        remaining_predictors = [p for p in pool_predictors if p not in current_predictors]
        # If there are no more predictors left to use, raise an error we can easily identify as normal
        if len(remaining_predictors)==0:
            raise ValueError("No more predictors left to use","NormalStop")
        
        #Create a model for each combination of: current predictor(s) + one of the remaining predictors
        #Keep track of the submodels and their performance
        #If error skip to next and do not include in comparison table
        results = []
        errorcount = 0
        for p in remaining_predictors:
            try:
                results.append(self.__buildModel(current_predictors+[p]))
            except:
                errorcount += 1 
        df_results = pd.DataFrame(results)
        
        # If we require all coefficients to be positive...
        if positive_only:
            #Create a flag for each submodel to test if all coefficients are positive 
            all_positive = pd.Series(None, index=df_results.index)
            for i in range(0,len(df_results)):
                all_positive[i] = (df_results.modelfit[i].coef_ >= 0 ).all()
                
            # if no model exist with only positive coefficients raise error we can easily identify as normal
            if (all_positive==0).all():
                raise ValueError("No models with only positive coefficients","NormalStop")
                
            #Choose model with best performance and only positive coefficients
            best_model = df_results.loc[df_results[all_positive==1].auc.idxmax()]
            best_model = self.__calculateAUC(best_model)
            
        # If we don't require all coefficients to be positive...   
        else:
            #Choose model with best performance
            best_model = df_results.loc[df_results.auc.idxmax()]
            best_model = self.__calculateAUC(best_model)
        
        return best_model
    
