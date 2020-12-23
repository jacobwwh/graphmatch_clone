# Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree
Code for paper "Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree", SANER 2020  
Requires:   
pytorch    
javalang  
pytorch-geometric  

## Data
Google Code Jam snippets in googlejam4_src.zip  
Google Code Jam clone pairs in javadata.zip  
BigCloneBench snippets and clone pairs in BCB.zip  

## Running
Run experiments on Google Code Jam:  
python run_java.py  
For BigCloneBench:  
python run_bcb.py  

This operation include training, validation, testing and writing test results to files.   

Arguments:  
nextsib, ifedge, whileedge, foredge, blockedge, nexttoken, nextuse: whether to include these edge types in FA-AST  
data_setting: whether to perform data balance on training set  
  '0': no data balance  
  '11': pos:neg = 1:1  
  '13': pos:neg = 1:3  
  '0'/'11'/'13'/+'small': use a smaller version of the training set
