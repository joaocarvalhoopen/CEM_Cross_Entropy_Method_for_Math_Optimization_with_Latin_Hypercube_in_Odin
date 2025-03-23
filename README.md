# CEM Cross Entropy Method for Math Optimization with Latin Hypercube in Odin
A simple but fast mathematical optimizer for no derivatives black-box optimization. 

## Description
This is a Odin implementation of the Cross Entropy Method for mathematical optimization with Latin Hypercube Sampling for the initialization of the first iteration ( that is good for high dimension optimization ), then afterwords, Gaussian sampling with updated mean and variance of the elite sub set. This is a no derivative optimization method that can be used for float64 black-box optimization problems.

## Note
This method can optimize Rosenbrock with 5000 variables.

## License
MIT Open Source License

## References
- Wikipedia - Cross-entropy method <br> 
  [https://en.wikipedia.org/wiki/Cross-entropy_method](https://en.wikipedia.org/wiki/Cross-entropy_method)

- Those two videos shows pretty well what CEM is doing, <br>
  Video - Global Minimisation via Cross Entropy Method 2D <br>
  [https://www.youtube.com/watch?v=l7fS1KDMaOo](https://www.youtube.com/watch?v=l7fS1KDMaOo) <br>
  Video - Stochastic Optimization - Cross Entropy Visualization 3D <br>
  [https://www.youtube.com/watch?v=tNAIHEse7Ms](https://www.youtube.com/watch?v=tNAIHEse7Ms)

- A Tutorial on the Cross-Entropy Method <br>
  [https://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf](https://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf)

- Book - The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation, and Machine Learning <br>
  by Rubinstein, R.Y. and Kroese, D.P. 2004

- Wikipedia - Rosenbrock function <br>
  [https://en.wikipedia.org/wiki/Rosenbrock_function](https://en.wikipedia.org/wiki/Rosenbrock_function) 

- Virtual Library of Simulation Experiments <br>
  Test Functions and Datasets <br>
  [https://www.sfu.ca/~ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html)

## Have fun!
Best regards, <br>
Joao Carvalho
