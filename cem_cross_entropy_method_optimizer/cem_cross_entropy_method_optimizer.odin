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

package cem_cross_entropy_method_optimizer

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:os"

MODE_DECREASING_POPULATION :: false

// IMPORTANT: This normally is worse than the normal mode.
//
// MODE_DECREASING_POPULATION :: true

// CEM - Cross-Entropy Optimization
//
// Parameters:
//   objective_fn            : Function pointer f( x: f64) -> f64 returning a score.
//   bounds_low, bounds_high : Slice arrays of length D.
//   D                       : Number of dimensions.
//   num_samples             : Population size.
//   num_elite               : Top fraction to guide distribution.
//   max_iterations          : Maximum iterations
//   eps                     : Variance threshold for early stopping.
//                             NOTE: This process diminish variance in the
//                                   population at each iteration.
//
// Output:
//   out_best_point : Slice of array of length D, the best point found.
//   out_best_value : Pointer to f64, the best objective value found
//   out_final_mean : Slice of the array of length D, the final mean.
// 
// The function returns after final iteration or if var < eps in all dims.
//
cem_optimize :: proc ( objective_fn      : proc ( x : [ ]f64 ) -> f64,
                       bounds_low        : [ ]f64,
                       bounds_high       : [ ]f64,
                       starting_point_x0 : [ ]f64,
                       D                 : int,
                       num_samples       : int,
                       num_elite         : int,
                       max_iterations    : int,
                       eps               : f64 ) ->
                     ( out_best_point    : [ ]f64,
                       out_best_value    : f64,
                       out_final_mean    : [ ]f64 ) { 


    assert( D > 0 )
    assert( num_samples > 1 )
    assert( num_elite > 1 )
    assert( num_elite <= num_samples )
    assert( max_iterations > 0 )
    assert( eps > 0.0 )
    assert( len( bounds_low ) == D )
    assert( len( bounds_high ) == D )

    if ! check_if_point_in_bounds( starting_point_x0,
                                   bounds_low,
                                   bounds_high ) {
    
        fmt.printfln( "Error: Starting point X0 not in bounds." )
        os.exit( -1 )
    }

    out_best_point = make( [ ]f64, D )
    out_best_value = min( f64 )
    out_final_mean = make( [ ]f64, D )

    i : int = 0
    d : int = 0

    num_func_evals : int = 0

    // 1. Allocate the population array of size num_samples.
    population : [ ]Individual = make( [ ]Individual, num_samples )
    for i in 0 ..< num_samples {

        population[ i ].x = make( [ ]f64, D )
    }

    defer {

        for i in 0 ..< num_samples {

            delete( population[ i ].x )
        }
        delete( population )
    }

    // Print the bounds_low and bounds_high.
    print_slice( "bounds_low",
                 bounds_low )

    print_slice( "bounds_high",
                 bounds_high )

/*
    // 2. Use Latin Hypercube Sampling to initialize population.
    //    The LatinHypercube sampling will diminish the maximum distance
    //    between the points of all samples in the dimensions.  
    //    We'll store the result in population[ i ].x.
    lhs_temp : [ ][ ]f64 = latin_hypercube_sampling(
                                    num_samples,
                                    D,
                                    bounds_low,
                                    bounds_high )

    // Copy LHS result into population.
    for i in 0 ..< num_samples {

        for d in 0 ..< D {

            population[ i ].x[ d ] = lhs_temp[ i ][ d ]
        }

    }

    // DEBUG : only for debugging.
    // print_population( population )

    // Deallocate the lhs_temp.
    for i in 0 ..< num_samples {

        delete( lhs_temp[ i ] )
    }
    delete( lhs_temp )
*/

    // 2. Use Latin Hypercube Sampling to initialize population.
    //    The LatinHypercube sampling will diminish the maximum distance
    //    between the points of all samples in the dimensions.  
    //    We'll store the result in population[ i ].x.
    latin_hypercube_sampling_fast( num_samples,
                                   D,
                                   bounds_low,
                                   bounds_high,
                                   population,
                                   starting_point_x0 )


    // DEBUG : only for debugging.
    // print_population( population )


    fmt.printf( "Generated population 0 : %d\n", num_samples )

    // 3. Evaluate objective for each sample, then sort descending.
    for i in 0 ..< num_samples {

        population[ i ].score = objective_fn( population[ i ].x )
        num_func_evals += 1
    }

    // Sort the population in descending order.
    slice.sort_by( population, compare_individual_desc )

    // DEBUG : only for debugging.
    // print_population( population )


    // 4. Elite set is top num_elite. Compute initial mean / var.
    elite : [ ]Individual = population         // Top portion is 'elite' after sort.
    // Mean and variance of the elite set.
    mu  : [ ]f64 = make( [ ]f64, D )
    var : [ ]f64 = make( [ ]f64, D )

    // Calculate the mean of the elite set.
    for d in 0 ..< D {

        sum_d : f64 = 0.0
        for i in 0 ..< num_elite {

            sum_d += elite[ i ].x[ d ]
        }
        mu[ d ] = sum_d / f64( num_elite )
    }

    // Calculate the variance of the elite set.
    for d in 0 ..< D {

        sum_sq : f64 = 0.0
        for i in 0 ..< num_elite {

            diff : f64 = elite[ i ].x[ d ] - mu[ d ]
            sum_sq += diff * diff
        }
        var[ d ] = sum_sq / f64( num_elite )
    }

    // Track the overall best.  After the first generation, that is population[ 0 ].    
    out_best_value = population[ 0 ].score

    for d in 0 ..< D {

        out_best_point[ d ] = population[ 0 ].x[ d ]
    }

    fmt.printfln( "Generated scores 0 : best_eval=%e, best_point=%e",
                  out_best_value, population[ 0 ].score )

    // So we can update the num_samples and num_elite in the next iterations.
    num_samples := num_samples
    num_elite   := num_elite

    // 5. Iterate subsequent generations.
    for iteration in 1 ..= max_iterations {

        // Check if variance is below threshold in all dims => converge.
        all_small : bool = true
        for d in 0 ..< D {

            if var[ d ] >= eps {

                all_small = false
                break
            }
        }

        if all_small {
        
            fmt.printfln( "Converged at iteration %d.", iteration )
            break
        }


        // Calculate the square root of the variance only once.
        for d in 0 ..< D {

            var[ d ] = var[ d ] > 1e-30 ? math.sqrt( var[ d ] ) : 1e-15
        }


        // 5a. Sample a new population from N( mu, var ).
        // Generate the new individuals in the population.
        // We will maintain the best of the previous iteration in the population,
        // so it always counts to the means and to the variance.
        // But we are not going to evaluate it again.
        // Don't forget that there is a descending sort for the score
        // after evaluation in the main loop.

        // With this scheme it goes from the total number of samples or individuals
        // to the new number of individuals - num_elite.
        
        // Calculate the new number of samples ( iter ) and the new num elite ( iter ).

        when MODE_DECREASING_POPULATION {

            iter_num_samples : int = num_samples - int ( f64( num_samples - num_elite ) *
                                                        ( f64( iteration ) / f64( max_iterations ) )
                                                    )

            ratio_num_elite := f64( num_elite ) / f64( num_samples )

            iter_num_elite : int = int ( f64( iter_num_samples ) * ratio_num_elite )

            // With trimmed of end inefficient samples, because it end in a Dirac impulse,
            // and in this way we can give more sample to the first iterations, by enlarging
            // the initial population.

            num_samples = iter_num_samples
            
            num_elite   = iter_num_elite
        }


        for ind_index in 1 ..< num_samples {

            for d in 0 ..< D {

                // std_d : f64 = var[ d ] > 1e-30 ? math.sqrt( var[ d ] ) : 1e-15
                
                std_d : f64 = var[ d ]

                // val : f64 = #force_inline rand.float64_normal( mu[ d ], std_d )
                val : f64 = #force_inline my_float64_normal( mu[ d ], std_d )
                
                // Clamp to [ bounds_low[ d ], bounds_high[ d ] ] if needed.
                val = #force_inline clamp( val, bounds_low[ d ], bounds_high[ d ] )
                population[ ind_index ].x[ d ] = val
            }

            // Evaluate the objective.
            population[ ind_index ].score = objective_fn( population[ ind_index ].x )
            num_func_evals += 1
        }


        // Sort the population in descending order.
        slice.sort_by( population[ 0 : num_samples ], compare_individual_desc )

        // 5c.  Elite is top num_elite of new population. Recompute mu / var.      
        elite = population[ 0 : num_elite ]  // Top portion is 'elite' after sort.


/*
        // Calculate the mean of the elite set.
        for d in 0 ..< D {

            sum_d : f64 = 0.0
            for i in 0 ..< num_elite {

                sum_d += elite[ i ].x[ d ]
            }
            mu[ d ] = sum_d / f64( num_elite )
        }

        // Calculate the variance of the elite set.
        for d in 0 ..< D {

            sum_sq : f64 = 0.0
            for i in 0 ..< num_elite {

                diff : f64 = elite[ i ].x[ d ] - mu[ d ]
                sum_sq += diff * diff
            }
            var[ d ] = sum_sq / f64( num_elite )
        }

*/

        
        // ===>>> Calculate the mean of the elite set in a more cache friendly way.

        // Set mean to zero.
        for i in 0 ..< D {

            mu[ i ] = 0.0
        }

        // More cache friendly, because mu is a small vector in can fit on L2 cache.
        for i in 0 ..< num_elite {

            for d in 0 ..< D {

                mu[ d ] += elite[ i ].x[ d ]
            }
        }

        for d in 0 ..< D {

            mu[ d ] /= f64( num_elite )
        }




        // ===>>> Calculate the variance of the elite set in a more cache friendly way.

        // Set mean to zero.
        for i in 0 ..< D {

            var[ i ] = 0.0
        }

        // More cache friendly, because mu is a small vector in can fit on L2 cache.
        for i in 0 ..< num_elite {

            for d in 0 ..< D {

                diff : f64 = elite[ i ].x[ d ] - mu[ d ]
                var[ d ] += diff * diff
            }
        }

        for d in 0 ..< D {

            var[ d ] /= f64( num_elite )
        }




        // Track the overall best.  After the first generation, that is population[ 0 ].
        if population[ 0 ].score > out_best_value {

            out_best_value = population[ 0 ].score
            for d in 0 ..< D {

                out_best_point[ d ] = population[ 0 ].x[ d ]
            }
        }

        // Print iteration info.
        fmt.printfln( "Iteration %d: best_score=%.6e, current_best=%.6e",
                       iteration, out_best_value, population[ 0 ].score )
    }

    fmt.printfln( "\nFinished CEM - Cross Entropy Method optimization." )

    fmt.printfln( "Total function evaluations = %d", num_func_evals )

    // Prepare mean output.
    // The other outputs are already set.
    for d in 0 ..< D {

        out_final_mean[ d ] = mu[ d ]
    }

    return out_best_point, out_best_value, out_final_mean
}

check_if_point_in_bounds :: proc ( point       : [ ]f64,
                                   bounds_low  : [ ]f64,
                                   bounds_high : [ ]f64 ) ->
                                   bool {

    D : int = len( point )

    for d in 0 ..< D {

        if point[ d ] < bounds_low[ d ] || point[ d ] > bounds_high[ d ] {

            return false
        }
    }

    return true
}

clamp :: #force_inline proc ( val : f64,
                              low : f64,
                              high : f64 ) ->
                              f64 {

    return val < low ? low : (val > high ? high : val)
}

// The following function is from the standard library.
my_float64_normal :: #force_inline proc( mean, stddev: f64, gen := context.random_generator ) -> f64 {

    return #force_inline rand.norm_float64( gen ) * stddev + mean
}


// Latin Hypercube Sampling
//   num_samples : Number of points
//   D           : Dimension
//   bounds      : Array of ( low, high ) pairs, length = D
//   out_samples : 2D array, out_samples[ i ][ d ], i in [ 0..num_samples - 1 ]
//                 must be preallocated by caller.
//
//    1. Cut the interval [ 0,1 ] into num_samples segments.
//    2. For each dimension, randomize the points within segments.
//    3. Shuffle along each dimension.
//    4. Scale to the user-specified [ low, high ]
//
latin_hypercube_sampling :: proc ( num_samples : int,
                                   D           : int,
                                   bounds_low  : [ ]f64,
                                   bounds_high : [ ]f64  ) ->
                                 ( out_samples : [ ][ ]f64 ) {

    // Step 0: Allocate the population array of size num_samples.
    out_samples = make( [ ][ ]f64, num_samples )
    for i in 0 ..< num_samples {

        out_samples[ i ] = make( [ ]f64, D )
    }

    // Step 1: Create an array 'cut' of size ( num_samples + 1 ) in [ 0, 1 ].
    cut : [ ]f64 = make( [ ]f64, num_samples + 1 )

    for i in 0 ..= num_samples {

        cut[ i ] = f64( i ) / f64( num_samples ) // linspace( 0, 1, num_samples + 1 )
    }
    
    defer delete( cut )

    // We'll create rdpoints = a + ( b - a ) * u, where 
    // a[ i ] = cut[ i ], b[ i ] = cut[ i + 1 ], i = 0 ..( num_samples - 1 ).
    for i in 0 ..< num_samples {

        a_i : f64 = cut[ i ]
        b_i : f64 = cut[ i + 1 ]
        for d in 0 ..< D {

            u : f64 = #force_inline rand.float64_range( 0, 1 )
            val_01 : f64 = a_i + ( b_i - a_i ) * u
            // We'll store these in out_samples[ i ][ d ] initially.
            out_samples[ i ][ d ] = val_01
        }
    }

    // Shuffle each dimension's column independently. 
    // We shuffle row indices but only for the dimension 'd'. 
    // This is effectively rng.shuffle( rdpoints[ :, d ] ) in Python.
    for d in 0 ..< D {

        for i in 0 ..< num_samples - 1 {

            // Pick a random index from i ..( num_samples - 1 ) .
            swap_idx : int = i + int( #force_inline rand.uint64( ) % ( u64( num_samples - i ) ) )
            temp : f64 = out_samples[ i ][ d ]
            out_samples[ i ][ d ] = out_samples[ swap_idx ][ d ]
            out_samples[ swap_idx ][ d ] = temp
        }
    }

    // Finally, scale from [ 0, 1 ] to the dimension's [ low, high ] .
    for i in 0 ..< num_samples {
    
        for d in 0 ..< D {
    
            low  : f64 = bounds_low[ d ]
            high : f64 = bounds_high[ d ]
            out_samples[ i ][ d ] = low + out_samples[ i ][ d ] * ( high - low )
        }
    }

    return out_samples
}


// Latin Hypercube Sampling fast version.
//
//   num_samples : Number of points
//   D           : Dimension
//   bounds      : Array of ( low, high ) pairs, length = D
//   population  : 1D array of individuals, each with an x slice field
//                 with the dimensions, population[ individual_index ].x[ d ],
//                 i in [ 0 .. num_samples - 1 ] must be preallocated by caller.
//
latin_hypercube_sampling_fast :: proc ( num_samples       : int,
                                        D                 : int,
                                        bounds_low        : [ ]f64,
                                        bounds_high       : [ ]f64,
                                        population        : [ ]Individual,
                                        starting_point_x0 : [ ]f64         ) {


    // Seed the random number generator. 
    // For reproducible results, replace time(NULL) with a fixed seed.
    // srand((unsigned int) time(NULL));

    // Temporary array to hold a shuffled index permutation for each dimension.
    idx : [ ]int = make( [ ]int, num_samples )
    if idx == nil {

        fmt.printfln( "Error: latin_hypercube_sampling_fast - Could not allocate temporary index array." )
        os.exit( -1 )
    }
    defer delete( idx )

    // Iterate over each dimension
    for d in 0 ..< D {

        // 1. Fill idx with [ 0 .. num_samples - 1 ]
        for i in 0 ..< num_samples {

            idx[ i ] = i
        }

        // 2. Shuffle idx using Fisher-Yates
        for i : int = num_samples - 1; i > 0; i -= 1 {

            j : int = int( #force_inline rand.uint64( ) % u64( (i + 1) ) )
            temp : int = idx[ i ]
            idx[ i ] = idx[ j ]
            idx[ j ] = temp
        }

        // 3. Fill out_samples: a random offset + scale to [bounds_low[d], bounds_high[d]]
        range : f64 = bounds_high[ d ] - bounds_low[ d ]
        for i in 0 ..< num_samples {

            // Random offset in [0,1)
            // r : f64 = f64( rand.uint( ) ) / (f64(  (double)RAND_MAX + 1.0);
            r : f64 = #force_inline rand.float64_range( 0, 1 )
            // Fraction = ( shuffled_index + offset ) / num_samples
            fraction : f64 = ( f64( idx[ i ] ) + r ) / f64( num_samples )
            // Scale to [ bounds_low[ d ], bounds_high[ d ] ]
            population[ i ].x[ d ] = bounds_low[ d ] + fraction * range;
        }
    }

    // Add the starting point x0 to the population.
    for d in 0 ..< D {

        population[ 0 ].x[ d ] = starting_point_x0[ d ]
    } 

}

// We'll define a struct for storing individuals: 
//   x     : Slice pointer to array of length D
//   score : fitness score value
Individual :: struct {

    x     : [ ]f64,    // Slice of length D.
    score : f64, 
}

// We want a descending sort by .score
compare_individual_desc :: #force_inline proc ( a : Individual,
                                                b : Individual ) ->
                                                bool {

    return a.score > b.score
}

print_slice :: proc ( slice_name : string,
                      slice_data : [ ]f64  ) {

    D : int = len( slice_data )

    fmt.printf( "%s = [ ", slice_name )
    for d in 0 ..< D {

        fmt.printf( "%f", slice_data[ d ] )
        if d < D - 1 {
        
            fmt.printf( ", " )
        }
    }
    fmt.printfln( " ]" )
}

print_population :: proc ( population : [ ]Individual ) {

    for i in 0 ..< len( population ) {

        slice_name := fmt.aprintf( "population[ %d ] - score : %f ",
                                    i, population[ i ].score )
        print_slice( slice_name,
                     population[ i ].x )
    }
}

// Example fitness or objective functions

// 3D toy objective: 
// A "bowl" - like shape near ( 2, -1, -10 ). We want to MAXIMIZE it.
my_3d_objective :: #force_inline proc ( point : [ ]f64 ) -> f64 {

    x : f64 = point[ 0 ]
    y : f64 = point[ 1 ]
    z : f64 = point[ 2 ]
    
    /* -( ( x - 2 ) ^ 2 ) - 0.5 * ( ( y + 1 ) ^ 2 ) - 0.1 * ( ( z + 10 ) ^ 2 ) */

    return -( (x - 2.0 ) * ( x - 2.0 ) ) -
           0.5 * ( ( y + 1.0 ) * ( y + 1.0 ) ) -
           0.1 * ( ( z + 10.0 ) * ( z + 10.0 ) )
}

// Rosenbrock in 30D or 100D or any dimension. We do the "negative" 
// so that we can maximize ( rather than minimize ).
rosenbrock_neg :: #force_inline proc ( x : [ ]f64 ) -> f64 {

    D : int = len( x )

    s : f64 = 0.0
    for i in 0 ..< D - 1 {

        term1 : f64 = x[ i + 1 ] - x[ i ] * x[ i ]
        term2 : f64 = 1.0 - x[ i ]
        s += 100.0 * term1 * term1 + term2 * term2
    }

    // Negative ( multiplied by -1 ) to make it a maximization objective
    return -s
}

// Sum of Different Powers Function
// https://www.sfu.ca/~ssurjano/sumpow.html
sum_of_different_power_functions_neg :: #force_inline proc ( x : [ ]f64 ) -> f64 {

    D : int = len( x )

    s : f64 = 0.0

    for i in 0 ..< D - 1 {

        s += math.pow( math.abs( x[ i ] ), f64( i + 1 ) )
    }

    // Negative ( multiplied by -1 ) to make it a maximization objective
    return -s
}

// Sum of 2 Powers Function
sum_of_2_powers_neg :: proc ( x : [ ]f64 ) -> f64 {

    D : int = len( x )

    s : f64 = 0.0

    for i in 0 ..< D {

        s += math.pow( x[ i ], 2 )
    }

    return -s
}

// Test #1: 3D example
test_3_dimensions :: proc ( ) {

    // Num dimensions.
    D : int = 3;
    bounds_low  := [ ? ]f64{ -10.0, -20.0, -40.0 }
    bounds_high := [ ? ]f64{  10.0,  30.0,   0.0 }

    // If you don't put nothing is inicialized with point zero in all dimensions.
    start_point_x0 : [ ]f64 = make( [ ]f64, D )


    // num_samples    : int = 100
    // num_elite      : int =  10
    // max_iterations : int =  50
    // eps            : f64 =   1e-6


    num_samples    : int = 100
    num_elite      : int =  20
    max_iterations : int =  50
    eps            : f64 =   1e-6


    // num_samples    : int = 10
    // num_elite      : int =  3
    // max_iterations : int =  2
    // eps            : f64 =  1e-6


    // Call CEM Optimizer
    best_point : [ ]f64
    best_value : f64
    final_mean : [ ]f64
    
    best_point,
    best_value,
    final_mean = cem_optimize( my_3d_objective,
                               bounds_low[ : ],
                               bounds_high[ : ], 
                               start_point_x0,
                               D,
                               num_samples,
                               num_elite,
                               max_iterations,
                               eps )

    defer {

        delete( best_point )
        delete( final_mean )
    }


    fmt.printfln( "\n=== 3D Example ===" )

    slice_name := "Final mean"
    print_slice( slice_name,
                 final_mean ) 

    fmt.printfln( "Best value found = %e", best_value )

    slice_name = "Best point found"
    print_slice( slice_name,
                 best_point ) 

}

// Test #2: Rosenbrock in N dimensions 
// By default let's do 30D. You can increase to 100D or 200D if desired.
test_rosenbrock_N_dimensions :: proc ( ) {

    // Adjust dimension as needed. For large D, consider bigger 
    // population or more iterations.
    
    // D : int = 2

    // D : int = 3    
    
    // D : int = 30
    
    // D : int = 100

    // D : int = 200

    D : int = 600

    // D : int = 5000

    // D : int = 10000


    bounds_low  : [ ]f64 = make( [ ]f64, D )
    bounds_high : [ ]f64 = make( [ ]f64, D )
    
    defer {
        delete( bounds_low )
        delete( bounds_high )
    }

    for d in 0 ..< D {

        bounds_low[ d ]  = -5.0
        bounds_high[ d ] =  5.0
    }

    // If you don't put nothing is inicialized with point zero in all dimensions.
    start_point_x0 : [ ]f64 = make( [ ]f64, D )


    start_point_x0[ 0 ] = -3.0
    start_point_x0[ 1 ] =  4.0

    // start_point_x0[ 0 ] = -1.0
    // start_point_x0[ 1 ] =  1.0


    // num_samples    : int = 200    // Possibly large for high dimension.
    // num_elite      : int =  40    // 20000
    // max_iterations : int =  10    //    10       // 200
    // eps            : f64 =   1e-9 //     1e-30   // 1e-6


    // WITHOUT POP Deacreasing
    // Total function evaluations = 6019700
    // Iteration 300: best_score  = -5.921227e+02
    // real	0m21,927s     Much faster than the next one.

    num_samples    : int = 20000     // Possibly large for high dimension.
    num_elite      : int =  4000     // 20000
    max_iterations : int =   300     //    10       // 200
    eps            : f64 =     1e-12 //     1e-30   // 1e-6


    // WITH POP Deacreasing
    // Total function evaluations = 48_999_756
    // Iteration 244: best_score  = -5.920818e+02
    // real	2m53,175s  Much Worst than the previous one.


    // num_samples    : int = 200000     // Possibly large for high dimension.
    // num_elite      : int =  4000      
    // max_iterations : int =   300      
    // eps            : f64 =     1e-12  // 



    // num_samples    : int = 200  // Possibly large for high dimension.
    // num_elite      : int =  20
    // max_iterations : int = 100
    // eps            : f64 = 1e-6

    
    // num_samples    : int = 200000  // Possibly large for high dimension.
    // num_elite      : int =  40000  // 20000
    // max_iterations : int =    400  // 200
    // eps            : f64 =      1e-6 //  1e-30// 1e-6

    // num_samples    : int = 200000  // Possibly large for high dimension.
    // num_elite      : int =  40000  // 20000
    // max_iterations : int =    500  // 200
    // eps            : f64 =      1e-6 //  1e-30// 1e-6

    
    
    // Call CEM Optimizer.
    best_point : [ ]f64 
    best_value : f64
    final_mean : [ ]f64

    best_point,
    best_value,
    final_mean = cem_optimize( rosenbrock_neg,
                               bounds_low,
                               bounds_high,
                               start_point_x0,
                               D,
                               num_samples,
                               num_elite,
                               max_iterations,
                               eps )

    defer { 
        delete( best_point )
        delete( final_mean )                      
    }

    fmt.printfln( "\n=== %dD Rosenbrock Example ===", D )
   

    slice_name := "Final mean"
    print_slice( slice_name,
                 final_mean ) 
   
    fmt.printfln( "Best Rosenbrock value (neg. of usual) = %.e", best_value )
   
   
    slice_name = "Best point found"
    print_slice( slice_name,
                 best_point ) 
    
}

// Test #3: sum_of_different_power_functions_neg
test_sum_of_different_power_functions_neg :: proc ( ) {

    // Num dimensions.
    D : int = 240;

    // D : int = 80_000;


    bounds_low  : [ ]f64 = make( [ ]f64, D )
    bounds_high : [ ]f64 = make( [ ]f64, D )
    
    defer {
        delete( bounds_low )
        delete( bounds_high )
    }

    for d in 0 ..< D {

        bounds_low[ d ]  = -1.0
        bounds_high[ d ] =  1.0
    }

    // If you don't put nothing is inicialized with point zero in all dimensions.
    start_point_x0 : [ ]f64 = make( [ ]f64, D )

    defer {
        delete( start_point_x0 )
    }

    for d in 0 ..< D {

        start_point_x0[ d ] =  1.0
    }
 
    // 125 evaluations
    //
    // num_samples    : int = 25
    // num_elite      : int = 6
    // max_iterations : int = 4
    // eps            : f64 = 1e-6

    // 250 evaluations
    //
    // num_samples    : int = 50
    // num_elite      : int = 10
    // max_iterations : int =  4
    // eps            : f64 =  1e-6

    // 450 evaluations
    //
    // NOTE: This is a good setting for this function.
    // num_samples    : int = 50
    // num_elite      : int = 10
    // max_iterations : int =  8
    // eps            : f64 =  1e-6

    // 850 evaluations
    // NOTE: Very good setting for this function.
    //
    // num_samples    : int = 50
    // num_elite      : int = 10
    // max_iterations : int = 16
    // eps            : f64 =  1e-6

    // 1650 evaluations
    //
    // num_samples    : int = 50
    // num_elite      : int = 10
    // max_iterations : int = 32
    // eps            : f64 =  1e-6
 
    // 10_200 evaluations
    // 10_150 evaluations   No Decreasing Population    Best value found = -9.663598e-15
    num_samples    : int = 200
    num_elite      : int =  40
    max_iterations : int =  50
    eps            : f64 =   1e-6

    // 11_226 evaluations
    // 11_226 evaluations   With Decreasing Population    Best value found = -9.670331e-06
    //
    // IMPORTANT: Set the constant MODE_DECREASING_POPULATION to TRUE

    // num_samples    : int = 1200
    // num_elite      : int =  200
    // max_iterations : int =   50
    // eps            : f64 =    1e-6



    // num_samples    : int = 10000
    // num_elite      : int =  2000
    // max_iterations : int =   100
    // eps            : f64 =  1e-6


    // 900 evaluations
    //
    // num_samples    : int = 100
    // num_elite      : int =  10
    // max_iterations : int =   8
    // eps            : f64 =   1e-6


    // 2_550 evaluations
    //
    // num_samples    : int = 50
    // num_elite      : int = 10
    // max_iterations : int = 50
    // eps            : f64 =  1e-6

    // 5_100 evaluations
    //
    // num_samples    : int = 100
    // num_elite      : int =  20
    // max_iterations : int =  50
    // eps            : f64 =   1e-6

    // 101_000 evaluations
    //
    // num_samples    : int = 1000
    // num_elite      : int =  200
    // max_iterations : int =  100
    // eps            : f64 =    1e-6


    // Call CEM Optimizer
    best_point : [ ]f64
    best_value : f64
    final_mean : [ ]f64
    
    best_point,
    best_value,
    final_mean = cem_optimize( sum_of_different_power_functions_neg,
                               bounds_low[ : ],
                               bounds_high[ : ], 
                               start_point_x0,
                               D,
                               num_samples,
                               num_elite,
                               max_iterations,
                               eps )

    defer {

        delete( best_point )
        delete( final_mean )
    }


    fmt.printfln( "\n=== %d D sum_of_different_power_functions_neg ===", D )

    slice_name := "Final mean"
    print_slice( slice_name,
                 final_mean ) 

    fmt.printfln( "\nBest value found = %e\n", best_value )


    slice_name = "Best point found"
    print_slice( slice_name,
                 best_point ) 

}

// Test #4: sum_of_2_powers_neg
test_sum_of_2_powers_neg :: proc ( ) {

    // Num dimensions.
    D : int = 240;

    bounds_low  : [ ]f64 = make( [ ]f64, D )
    bounds_high : [ ]f64 = make( [ ]f64, D )
    
    defer {
        delete( bounds_low )
        delete( bounds_high )
    }

    for d in 0 ..< D {

        bounds_low[ d ]  = -1.0
        bounds_high[ d ] =  1.0
    }

    // If you don't put nothing is inicialized with point zero in all dimensions.
    start_point_x0 : [ ]f64 = make( [ ]f64, D )

    defer {
        delete( start_point_x0 )
    }

    for d in 0 ..< D {

        start_point_x0[ d ] =  1.0
    }
 
    // num_samples    : int = 50
    // num_elite      : int = 10
    // max_iterations : int = 50
    // eps            : f64 =  1e-6

    // 5000 evaluations
    // num_samples    : int = 100
    // num_elite      : int =  20
    // max_iterations : int =  50
    // eps            : f64 =   1e-6

    // 1000 evaluations
    num_samples    : int =  100
    num_elite      : int =   20
    max_iterations : int =   10
    eps            : f64 =    1e-6



    // Call CEM Optimizer
    best_point : [ ]f64
    best_value : f64
    final_mean : [ ]f64
    
    best_point,
    best_value,
    final_mean = cem_optimize( sum_of_2_powers_neg,
                               bounds_low[ : ],
                               bounds_high[ : ], 
                               start_point_x0,
                               D,
                               num_samples,
                               num_elite,
                               max_iterations,
                               eps )

    defer {

        delete( best_point )
        delete( final_mean )
    }

    fmt.printfln( "\n=== %d D sum_of_2_powers_neg ===", D )

    slice_name := "Final mean"
    print_slice( slice_name,
                 final_mean ) 

    fmt.printfln( "\nBest value found = %e\n", best_value )

    slice_name = "Best point found"
    print_slice( slice_name,
                 best_point ) 
}

// run the two test demos
test_main :: proc ( ) {
 
    // test_3_dimensions( )
    test_rosenbrock_N_dimensions( )

    // test_sum_of_different_power_functions_neg( )

    // test_sum_of_2_powers_neg( )
}



