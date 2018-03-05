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
