# Persistence Landscapes in Time Series Data Augmentation

### Project Description

Often, the coordinates of data points themselves tell
little about the object. To this end, we need to account
for how they correlate in a metric space. This is where TDA
(topological data analysis) comes into play.  

In this work, we apply augmentation techniques on time-series datasets
from [UEA & UCR repository](https://timeseriesclassification.com/).
Along with [the earlier proposed method](https://arxiv.org/pdf/1702.05538.pdf)
to perform simple transformations on object representations in a learned
feature space, we explore novel approaches involving the concept of
[persistence landscapes](https://arxiv.org/pdf/1207.6437.pdf).

### Augmentation

Most datasets are large enough for a convolutional classifier to
achieve test accuracy of 100%. So, we pick just a small number of
training samples to see the effect of applying the data augmentation
procedure, which comprises:

* Fitting
 - calculating persistence landscapes from time series
 - training the inverse map modeled by a network to reconstruct
 the original time series from persistence landscapes
* Predicting
 - interpolating and/or extrapolating between persistence landscapes
 - generating new time series from the modified landscapes
 (with the learned inverse map)

Then, we train the classifier on the augmented dataset and predict on the test.
We also propose the other augmentation scheme which is close to [DeVries and
Taylor's work](https://arxiv.org/pdf/1702.05538.pdf) with one additional
feature: we pass the perturbed code (transformed repr. of time series)
together with persistence landscape as input to the decoder, which learns
how to use this auxiliary information.

### Results & Conclusions

Experiment metadata:
Dataset "Coffee"  
Reduced dataset size - 8 samples  
Augmented dataset size - 72 samples  
Transformation used - interpolation

![Classifier train-validation loss on the reduced
Coffee dataset (left) and its augmented counterpart (right)](results.png)

We have shown the possibility of using persistence landscapes
for data augmentation. However, the method effectiveness remains
questionable. Since we didn't have time to carry out the full testing.

### Contributions

* Maxim Velikanov (not a github user at the time) -
TDA part (TDA-time-series.ipynb)
* [Vladislav Lukoshkin](https://github.com/lukoshkin) - the rest

### References

It is more about concepts and materials I liked rather than
those that became central in the project

* [Weight Normalization](https://arxiv.org/pdf/1602.07868.pdf)
* [Dilated Causal Convolutions](https://arxiv.org/pdf/1803.01271.pdf)
* [Intro to TDA for Physicists](https://arxiv.org/pdf/1904.11044.pdf)
