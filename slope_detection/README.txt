Prior to actually seeing the data, we did some exploration with slope-finding neural nets.

Methylation is detected by examining the 'upstream' and 'downstream' slopes in the methylation density along the DNA strand.

If the data were precise enough to see these slopes directly, then the neural net would need to be able to locate and measure these slopes.

So, as a warmup excercise, I created and tested an algorithm to find those slopes.

datagen.py creates these window-like functions and saves them as numpy arrays, while keras_deriv.py teaches a neural net how to find them.

I also experimented with several other architectures in the keras_conv.py and keras_comb.py scripts.


As an even simpler warmup (included since everything is in the code), there is also a test to teach a neural net how to do a random linear transformation.

This is practically trivial, but serves as a simple, checkable result to ensure that learning is happening.
