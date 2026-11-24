# How to run the experiments:

- ### Execute `preprocessing.py` to process the data and build the required training, test and validation sets. Make sure the main dataset `ChCh-Miner_durgbank-chem-chem.tsv` is at the `data/` directory.
- ### For executing the first experiment (ablation on different groups of features), run `ablation_by_group.py`
- ### For executing the second experiment (singular feature analysis and permutation importance), run `features_analysis.py`.

### **Flags:**
|Flag|Description|Options|
|:----------:|:---------:|:------:|
|`--dataset`|Mandatory flag for choosing which dataset to preprocess or analyze. Available for all scripts|`baseline`, `markov`, `kernel`|
|`--analysis`|Optional for choosing which feature analysis to reproduce. Available only for `features_analysis.py`|`all` (default), `correlation`, `auc`, `models`, `permutation`|

### **Examples:**
- ### `python preprocessing.py --dataset baseline` runs the preprocessing pipeline for the topology features group
- ### `python features_analysis.py --dataset markov --analysis correlation` runs the pearson's correlation analysis for the topoly + markov features groups 
