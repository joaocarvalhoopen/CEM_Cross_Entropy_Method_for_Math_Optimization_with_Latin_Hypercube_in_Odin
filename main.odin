// Lib : CEM Cross Entropy Method for Math Optimization with Latin Hypercube in Odin
//
// Description : Odin Implementation of the Cross Entropy Method for mathematical
//               optimization with Latin Hypercube Sampling for the initialization
//               of the first iteration ( that is good for high dimension optimization ),
//               then Gaussian sampling with updated mean/variance of the elite sub set. 
//               This is a no derivative optimization method that can be used for
//               black-box optimization problems.
//
// Date : 2025.03.23
//
// Author: Joao Carvalho
//
// License: MIT Open Source License
//
// References:
//
//  - Wikipedia - Cross-entropy method 
//    [https://en.wikipedia.org/wiki/Cross-entropy_method](https://en.wikipedia.org/wiki/Cross-entropy_method)
//
//  - A Tutorial on the Cross-Entropy Method
//    [https://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf](https://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf)
//
//  - Book - The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation, and Machine Learning
//    by Rubinstein, R.Y. and Kroese, D.P. 2004
//
//  - Wikipedia - Rosenbrock function
//    [https://en.wikipedia.org/wiki/Rosenbrock_function](https://en.wikipedia.org/wiki/Rosenbrock_function) 
//
//  - Virtual Library of Simulation Experiments:
//    Test Functions and Datasets
//    [https://www.sfu.ca/~ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html)
//
//

package main

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:os"

import cem "./cem_cross_entropy_method_optimizer"


main :: proc ( ) {
    fmt.println( "Begin CEM - Cross-Entropy Method for Mathematical Optimization...\n" )

    cem.test_main()

    fmt.println( "\n... end CEM - Cross-Entropy Method for Mathematical Optimization." )
}