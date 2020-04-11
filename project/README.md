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
feature: we take into account the information contained in the corresponding
to encoded input vector persistence landscape by applying to them the same
transformations and passing further to the decoder.

In the experiments, in addition to the reduction of an original dataset,
we weaken the classifier to make the difference even more sensible

### Results & Conclusions

Experiment metadata:
Dataset "Coffee"  
Reduced dataset size - 8 samples  
Augmented dataset size - 72 samples  
Transformation used - interpolation

(a bit outdated picture)
![Classifier train-validation loss on the reduced
Coffee dataset (left) and its augmented counterpart (right)](results.png)

We have shown the possibility of using persistence landscapes
for data augmentation. In our experiments (it is about 15
different settings: univariate and multivariate datasets - 6 in total,
~ 3 sub-samples of each one), any of the three augmentation methods
(DeVries et al., our scheme #1 and #2) have become the best. So, it is
difficult to single out a better approach here, since none of them are
universal. To determine when and which method to apply, one needs to carry
out a separate study on this matter - what is beyond our project.

### Contributions

* Maxim Velikanov (not a github user at the time) - TDA part
* [Vladislav Lukoshkin](https://github.com/lukoshkin) - the rest

### References

It is more about concepts and materials I liked rather than
those that became central in the project

* [Weight Normalization](https://arxiv.org/pdf/1602.07868.pdf)
* [Dilated Causal Convolutions](https://arxiv.org/pdf/1803.01271.pdf)
* [Intro to TDA for Physicists](https://arxiv.org/pdf/1904.11044.pdf)
