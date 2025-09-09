# Is it enough to steer just "one" concept?

It is impossible to determine the correct granularity of the learnt representation: we do not know the number of ground truth concepts present, and enforcing each dimension to encode a sufficiently distinct concept seems like a huge assumption to make.

So, let's instead identify clusters of inter-connected concepts in the model's supposedly disentangled representation obtained through S/SAEs and steer all of them together. 

Step 1: Train SSAEs on your favourite dataset.

Step 2: Learn a linear SCM on the learnt steering vectors (which will be the same as the binary concepts the vectors steer).

Specifics TBD:

Step 3: Pass the steering vector **s** for a concept A through the SCM **M** to get its downstream effects as well and steer using **Ms** instead of **s**.
