# COBRA :snake: <img src="https://github.com/JanBenisek/Pytho/blob/master/pythongrey%20large.png" width="100" align="right">

**Cobra** here on GitHub is refactored web-based cobra originally developed by Guillaume. The goal is to wrap the back-end into easy to use Python package.

If you wish to modify the code, the best is to fork the repository or create another branch!

:heavy_exclamation_mark: Still lots of :bug: and under construction, keep that in mind:heavy_exclamation_mark:

## What can Cobra 1.0 do:
  * Transform given .csv to be ready to use for prediction modelling
    * _Clense the headers, partition into train/selection/validation sets, sample, bins and regroups variables and add columns with incidence rate per categories._
  * Perform univariate selection based on AUC
  * Find best model by forward selection
  * Visualize the results
  * Allow iteration among each step for the analyst
  
## Installation
  *  Clone this repository to your local PC (use GitHub Desktop). This assumes that the cloned repository will be in this directory `C:\Local\pers\Documents\GitHub\cobra`
  * Open Powershell and navigate to that folder
  * Once you are in the folder, execute `python setup.py install`. This is how the line should look like:
  `PS C:\Local\pers\Documents\GitHub\cobra> python setup.py install`
  * Restart kernel and you are ready to go
  * For example of use, see the Jupyter Notebook in `examples` folder
  
  
## Questions
### BUSINESS
  * Do we want to sort the output of univariate selection? If yes, based on what?
  * What does forcing a variable in the model means? I can force the variable to be used in the forward selection, even if it failed the other criterias, but still the forward selection does not have to pick it up.
  * Do we really need the process to find the optimal number of variables?
### TECHNICAL
**Data Preparation**
  * When importing the .csv, isn't better to use C engine in pandas and give explici info about the separator? Since parsing .csv is slow in Python, it can help speed up the process a bit
  * There should be input checks - how many missing, are the required columns present, partitioning params must equal to 1
  * The parametes sample_1/sample_0 feels confusing. I assume we always take all 1's, isn't better simply specify the ration amongst 1 and 0? Like 10 would mean take 10xmore 0's in respect to 1's
  * I rewrote the .csv loading, `getattr` was raising error when imported (its a global variable which when using in other scripts, behaves strangely)
  * The eqfreq and regroup functions are still a bit mistery, so I only copy-pasted them.
  
**Univariate Selection**

**Model Selection**
  * The whole forward selection is really hard to read and I dont I still fully understand it.
  * Whats the parameter positive only? And why it is a parameter if it can never be False?
  * How does it force variables? I could not find it in the code?
  * The procedure to choose best number of variables thorws an error. I am not brave enough to go into it.
  
**Design + Other**
  * Would it make sense to have the input/output DF's self? So the exchange between classes/procedure can be more smooth and more OOP? Sometimes I modify input DF and thats not the way we should go (data_preparation.py - addPartitionColumn())
  * Do you prefer to have extra library with auxiliary functions, or have those functions with double underscores in the classes (this is now the case). Or have them as static methods? But thats not always feasible
  * What do you think about the overall design - 3 classes wrapped into one main?
  * And how about the idea returning DFs?
  * Did we test the speed? I assume its gonna explode with bigger datasets.
  * It would be nice if someone checks the comments withing the functions. I had to guess a bit sometimes what each param means.
  * Sometimes inside the classes I return the DF, sometimes and modify it. Not consistent. Any idea how to make it better?
  * What do you think about the visualization?
  * How to improve import so I don't have to do cobra.cobra?
  


