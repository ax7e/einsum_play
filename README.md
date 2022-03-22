# H3_play
- We tested several methods to optimize einsum
- The raw code resides in H3.
# Methods Include:
		- opt_einsum with cupy as backend
		- cutensor-einsum
		- manually reduce tensor using cutensor functions(best)
		- manually reduce tensor using searched order(slightly) 
		- numpy.einsum(extremly slow)
