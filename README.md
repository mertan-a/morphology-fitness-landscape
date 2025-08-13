# Morphology-fitness landscape

This is the code and data for the paper "Evolutionary Brain-Body Co-Optimization Consistently Fails to Select for Morphological Potential". (The links to the paper will be available soon...)

## Instructions

Detailed instructions to inspect the morphology-fitness landscape will be available soon. For now, please do

```
cd results
tar -xvzf compressed_data.tar.gz
```

to extract the "raw_results.npy" -- which contains the initial morphology-fitness landscape, and the "updated_results.pkl" -- which contains the updated morphology-fitness landscape. Both are Python dictionaries with keys being morphology IDs (see "search_space.py" for converting morphology IDs to their grid representations). In "raw_results.npy", the values are length 300 lists showing the fitness trajectory of 300 generations long controller evolution for that morphology. In "updated_results.pkl", values are numbers showing the estimated true fitness of that morphology. 
