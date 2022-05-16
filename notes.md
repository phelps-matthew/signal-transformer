## ConvEncoder
* In group norm paper, given fixed number of 32 groups, He finds that best number of channels per groups n_channels / 2
* Performance is nearly invariant to group number itself, peaking at 32 groups in ResNets. 
* GELU weighs input by its sign and magnitude, rather than sign as with RELU. Allows bounded (negatative) activation in negative regions. Positive activation is unbounded.
* GELUS are the expectation of modified Adaptive Dropout. See paper, interesting formulation as the activation function is derived from idea computing expectation of dropout and zoneout of neurons. GELU's appear to perform *much* better than RELU when either are accompanied by dropout.
* GELU weights its input depending on how much greater it is than other inputs. 
