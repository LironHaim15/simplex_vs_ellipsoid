
# Simplex vs Interior Point (Ellipsoid)

## About
A small Python program that compares results of finding optimal solution in Simplex and Interior Point (Ellipsoid) methods.


### Requirements
* Libraries:
	* numpy 
	* matplotlib
	* scipy (for optimizations methods)
  

### Instructions
The main class in the program is called `Model`. An instance of this class holds the following fields:
* `method` - indicates the optimization method: string of `simplex` or `interior-point`
* `constraints` -  defines the problems constraints: 2D array where each row is holds the coefficients of one 		  constraint.
* `resources` - defines the resources for each constraint: an array where each cell is the resource of the constraint in the row at the same index in `constraints` array.
* `objective` - defines the target function of the problem (for minimization or maximization): an array of the coefficients of the objective function. 
* `error` - a string for errors.

An example of a problem:
>Maximize: f = 2*x[0] + 4*x[1]
>
>Subject to: -5*x[0] + 3*x[1] <= 7
>
>1*x[0] + 4*x[1] <= 6

the problem above will be entered as:
>`objective` = [2, 4]
> 
>`constraints` = [[-5, 3], [1, 4]]
> 
>`resources` = [7, 6]

In order to run the optimizer - call the class method `run()`.

There are some predefined tests in the code, commented in the main function, for example:
* Calculating the average iterations of 100 random problems from dimension `dim`.
* Calculating the amount of iterations of solving Klee-Minty problem from dimension `dim`.
* Calculating the amount of iterations of solving Klee-Minty problem from dimension `dim` after scaling the problem and converting its constraints to a slightly different porblem (reduction).

The above are for both Simplex & Interior Point.
