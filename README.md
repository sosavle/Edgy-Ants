## About Ant Colony Optimization Algorithms
The ant colony optimization algorithm (ACO) is a probabilistic technique for solving computational problems which can be reduced to finding good (but not necessarilly analytically optimal) paths through graphs. Artificial ant behavior inspiration was taken from the behavior of real ants. An ant repeatedly hops from one location to another to ultimately reach the destination (food). Ants deposit organic compounds called pheromones while tracing a path. The intensity of the pheromone is an indicator that the particular path is more favorable. More ants traverse through the different paths and the pheromone intensity gets updated each time, with each individual ant preferring a path that has a higher pheromone intensity.

## Program Parameters
The behavior of the ants colony edge detection algorithm is mediated by a variety of parameters. There are a total of nine adjustable parameters in this implementation of the algorithm: 
* number of ants
* pheromone weight factor (alpha)
* pheromone evaporation factor (rho)
* minimum pheromone intensity (tau_min)
* edge visibility weight factor (beta)
* visibility threshold value (b)
* ant memory length
* cycles
* steps per cycle.

## Parameter effects on resulting edgemap

### Alpha and beta
![Effects of alpha and beta parameters on edge resolution](./alpha-beta.PNG)
As each ant explores the image surface, there are two different pieces of information that may guide the ant's next move: the amount of pheromone in a spot (i.e. following the global maximum established so far) and what the visibility value of a spot (what the ant individually considers would be the best move). The relative weighting of these two parameters is controlled by factors alpha and beta. As the picture above shows, higher values of alpha tend to result in a greater number of edges, while higher values of beta tend to result in sharper edges. This effect only tepends on the ratio between alpha and beta, so increasing or decreasing both parameters proportionally will not affect edge resolution.


### Evaporation rate
![Effect of evaporation rate on edge resolution](./evaporation.PNG)
After every step within a cycle, a small percentage of deposited pheromones can be made to evaporate, which may be used to erase extraneous edges. As shown above, however, this parameter should be used only to make minor adjustments after other parameters have been properly set. Ignoring evaporation (i.e. setting it to zero) will still result in a good edgemap, whereas increasing evaporation excessively may make your image fail to resolve (i.e. there is not a significant enough difference in pheromone values in the image).


### Number of cycles and steps per cycle
![Effects of number of steps and cycles on edge resolution](./steps-cycles.PNG)
The program can be run in multiple cycles, where the first one starts with a blank slate with minimal pheromone values uniformly distributed across the image, and subsequent cycles restart the process over the pheromone map developed by previous ants. During each cycle, each individual ant can make up to a set number of steps. Increasing the amount of steps per cycle can increase the maximum number and length of the edges found, while increasing the amount of cycles will usually result in a reduction of the less defining edges.


### Number of ants
![Effects of number of ants on edge resolution](./numAnts.PNG)
The number of ants is stronlgy related to the performance of the algorithm, both in terms of the resulting edgemap as well as the time taken to execute. The number of ants recommended by literature is the square root of the number of pixels in the image.

## Sources and further reading
* Health M, Sarkar S, Sanocki T, Bowyer KW (1998) Comparison of edge detectors: a methodology and initial study. Comput Vis Image Understanding 69 :38–54
* Jing Tian, Weiyu Yu and Shengli Xie, "An ant colony optimization algorithm for image edge detection," 2008 IEEE Congress on Evolutionary Computation (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 751-756, doi: 10.1109/CEC.2008.4630880.
