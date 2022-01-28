import itertools
import multiprocessing as mp
import numpy as np
import random
import tensorflow as tf
import time
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import linear as keras_linear
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from scipy.spatial import ConvexHull

import sys
sys.path.append('./../../../Marabou/')
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
from hitman_cost import HitmanCost

import adversarial_attacks as adv

def logger(msg, 
           verbose=False, log_type=''):
    """
    Print stuff nicely.
    """
    if verbose is False:
        pass
    else:
        print("[logger{}]: {}".format('-'+log_type if log_type!='' else '', msg))

def idx2word(indices, window_size, input_len):
    """
    When a feature is selected by MinimumHS, the entire window is selected
    """
    res = []
    for idx in indices:
        for r in range(0, input_len, window_size):
            if idx<r+window_size and idx>=r and (idx not in res):
                res.extend([el for el in range(r, min(r+window_size, input_len))])
    return res

def freeze_session(session, 
                   keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def MinimumHS(subsets,costs, 
              htype='smallest', **kwargs):
    """
    Return a list that contains the minimum-hitting-set of the input, i.e. given a list of lists
     it finds the minimum set that intersects all the others and whose dimension is minimal
    Example: subsets = [[1], [], [2,1], [33]], returns [1,33] (it ignores empty sets)
    Uses pysat Hitman, but includes the possibility to extract the smallest, the largest and a random
     minimum hitting set.
    Input:
        subsets:list of lists
            all the "sets" used to find the minimum-hitting set. Can't contain empty set []
        costs: dict
            key is the word and value is it's cost
        htype:string
            (optional) 'smallest' resturn the smallest and largest minimal HS, "random-smallest" choses one of the 
             minimal hitting sets at random, 'random' returns one at random everytime is invoked. 
             'MLC' (Minumum Linear Cost) selects the explanation that minimizes the linear distance among all 
             the features (when sorted). Finally, 'LIME' guarantees that the hitting set is both minimal and
             has as much feature in common with the kwargs `lime_set` as possible
        **kwargs:any
            (optional) `lime_set` is a list of features (from a LIME explanation) that is used to search the 
             MinimumHS of minimal size and with as much feature in common with `lime_set` as possible.
    Output:
        hs:list
            minimum hitting set
    """
    #print(subsets)
    #print('-------------------------------')
    if len(subsets) == 0:
        hs = []
        return hs
    if [] in subsets:
        raise Exception("Subsets variable can't contain [] set (empty)")
    h = HitmanCost(bootstrap_with=subsets, costs=costs,htype='sorted')
    if htype == 'smallest':
        hs = h.get()
        return hs
    elif htype == 'largest':
        for el in h.enumerate():
            hs = el
        return hs
    elif htype == 'random':
        HS = []
        for el in h.enumerate():
            HS += [el]
        hs = random.choice(HS)
        return hs
    elif htype == "random-smallest":
        HS = [h.get()]
        min_size = len(HS[0])
        for el in h.enumerate():
            if len(el) == min_size:
                HS += [el]
        hs = random.choice(HS)
        return hs
    elif htype == 'MLC':
        best_hs, min_cost = [], np.inf
        for el in h.enumerate():
            c = 0
            el.sort()  # sort and calculate linear cost
            for i,_ in enumerate(el[:-1]):
                c += el[i+1] - el[i]
            if c < min_cost:
                best_hs = el
                min_cost = c
        return best_hs
    elif htype == 'LIME' and len(kwargs['lime_set']) > 0:
        lime_set = kwargs['lime_set']
        hs = h.get()
        min_size = len(hs)
        features_in_common = len(set(lime_set).intersection(hs))
        for el in h.enumerate():
            if len(el) == min_size and len(set(lime_set).intersection(el)) > features_in_common:
                hs = el
                features_in_common = len(set(lime_set).intersection(el))
        return hs
    elif htype == "exclude" and len(kwargs['exclude_set']) > 0:
        exclude_set = kwargs['exclude_set']
        hs = h.get()
        min_size = len(hs)
        features_in_common = len(set(exclude_set).intersection(hs))
        for el in h.enumerate():
            if len(el) == min_size and len(set(exclude_set).intersection(el)) < features_in_common:
                hs = el
                features_in_common = len(set(exclude_set).intersection(el))
        return hs
    else:
        raise NotImplementedError("{} is not a vald htype method".format(htype))

def Entails(h, network, input_, input_bounds, output_constraints, weights, window_size, norm,
            verbose=False):
    """
    Implementation of Entails function from "Abduction-Based Explanation for ML Models", AAAI-2019 that
     is solved with Marabou c++ solver.
    Input:
        h:list
            contains the indices of variables that are fixed (i.e., C - h are free vars).
        network:maraboupy.MarabouNetworkTF.MarabouNetworkTF
            marabou network specification
        input_bounds:list
            list of [min, max] lists, one entry for each variable
        output_constraints:list
            list of [1st_output_var:int, 2nd_output_var:int, tolerance:float] s.t. the output constraint is expressed in the
             form (1st_output_var - 2nd_output_var <= tolerance) which is approx. (1st_output_var < 2nd_output_var)
        weights:list
            list with numpy.arrays of weights and biases of the softmax layer
        verbose:boolean
            (optional) verbosity of the logger function
    Output:
        vals:dictionary
            dictionary of values {idx: value} where each index refers to the i-th input variable, while the
             value is indeed the value of a successfull adversarial attack if found. 
             If nothing has been found, vals is equal to {}, hence its length is 0.
    """
    # Ranges for Free params ('f_ranges') and Constants ('c_ranges')
    c_ranges = [n for n in h]
    f_ranges = [n for n in range(0, len(input_)) if n not in c_ranges]
    logger("Free Vars(s) {}".format(f_ranges), verbose, "DEBUG")
    logger("Constant Var(s) {}".format(c_ranges), verbose, "DEBUG")
    # Get the input and output variable numbers
    inputVars = network.inputVars[0][0].flatten()
    outputVars = network.outputVars[0].flatten().tolist()
    # Set input bounds
    y_hat, target_class, eps = output_constraints   
    if norm == '1':  # L-1 norm is vector-wise (i.e., constraints are for an entire feature)
        # Set upper/lower bounds (depends on the L-p norm chosen)
        for n in f_ranges:
            network.setLowerBound(inputVars[n], input_bounds[n][0])
            network.setUpperBound(inputVars[n], input_bounds[n][1]) 
        sign_vars = [network.getNewVariable() for n in inputVars]
        for n in range(0, len(input_), window_size):
            if n in f_ranges:
                for nn in range(n, n+window_size):
                    network.addAbsConstraint(inputVars[n], sign_vars[nn])
                network.addInequality(sign_vars[n:n+window_size], [1 for _ in range(window_size)], eps)  
    elif norm == 'inf':
        # Set upper/lower bounds (depends on the L-p norm chosen)
        for n in f_ranges:
            network.setLowerBound(inputVars[n], input_bounds[n][0])
            network.setUpperBound(inputVars[n], input_bounds[n][1]) 
    elif norm == 'knn':
        # Set constraints for the k-neighbors convex hull
        input_bounds, minmax_input_bounds = input_bounds[0], input_bounds[1] 
        for i, n in enumerate(range(0, len(input_), window_size)):
            for ib in input_bounds[i]:
                p = np.dot(np.array(input_[n:n+window_size]).reshape(1,window_size), np.array(ib[:-1]).reshape(window_size,)) + ib[-1]
                assert p <= eps, print("Inconsistent bound, negative result of xW+b<=tolerance: {}".format(p))
                network.addInequality(inputVars[n:n+window_size], ib[:-1], eps-ib[-1])
        # Set upper/lower bounds (depending on the max-min values of each vertices in the convex-hull)
        for i, n in enumerate(range(0, len(input_), window_size)):
            if n in f_ranges:
                for l, j in enumerate(range(n, n+window_size)):
                    network.setLowerBound(inputVars[j], minmax_input_bounds[i][0][l])
                    network.setUpperBound(inputVars[j], minmax_input_bounds[i][1][l])
    elif norm == 'knn-linf':
        # Set constraints as the maxmin values for the k-neighbors convex hull
        _, minmax_input_bounds = input_bounds[0], input_bounds[1] 
        # Set upper/lower bounds (depending on the max-min values of each vertices in the convex-hull)
        for i, n in enumerate(range(0, len(input_), window_size)):
            if n in f_ranges:
                for l, j in enumerate(range(n, n+window_size)):
                    network.setLowerBound(inputVars[j], minmax_input_bounds[i][0][l])
                    network.setUpperBound(inputVars[j], minmax_input_bounds[i][1][l])
    else:
        raise Exception("Norm {} is not a valid norm, please use 'inf' or '1'".format(norm))
    # Constants (before and after Free Params) are set equal to their original value
    for n in c_ranges:
        equation = MarabouUtils.Equation()
        equation.addAddend(1, n)
        equation.setScalar(input_[n])
        network.addEquation(equation)
    # target_class>=(y_hat-\eps) inequality constraint
    #  (which becomes strict with a small epsilon added to one of the outputs)
    # API: \sum_i vars_i*coeffs_i <= scalar
    W_0, W_1 = weights[0][:,0], weights[0][:,1]
    b_0, b_1 = weights[1][0], weights[1][1]
    if y_hat == 0 and target_class == 1:  # xw0 + b0 < xw1 + b1
        network.addInequality(outputVars, W_0-W_1, b_1-b_0-eps)
    elif y_hat == 1 and target_class == 0:  # xw1 + b1 < xw0 + b0
        network.addInequality(outputVars, W_1-W_0, b_0-b_1-eps)
    else:
        raise Exception("y_hat={}, target_class={} is not valid assignment (0 and 1 values accepted)".format(y_hat, target_class))
    # Call to C++ Marabou solver
    logger("Results for value {}".format(h), verbose, "DEBUG")
    vals, _ = network.solve(verbose=False)
    return vals

def PickFalseLits(C_setminus_h, filename, input_, input_bounds, output_constraints, weights_softmax, norm,
                  window_size=1, randomize=False, verbose=False):
    """
    Search for a subset of free variables in C_setminus_h s.t. the adversarial attack is still effective.
    Input:
        C_setminus_h:list
            contains the indices of variables that are free.
             Naming is terrible but at least we are consistent with the abduction paper notation
        filename:string
            path to the graph in the format maraboupy.MarabouNetworkTF.MarabouNetworkTF
        input_:list
            flatten list of inputs of the adversarial attack
        input_bounds:list
            list of [min, max] lists, one entry for each variable
        output_constraints:list
            list of [1st_output_var:int, 2nd_output_var:int, tolerance:float] s.t. the output constraint is expressed in the
             form (1st_output_var - 2nd_output_var <= tolerance) which is approx. (1st_output_var < 2nd_output_var)
        weights_softmax:list
            list with numpy.arrays of weights and biases of the softmax layer
        window_size:int
            length of the window used to encode a single variable
        randomize:boolean
            (optional) free variables in C_setminus_h are shuffled before being processed
        verbose:boolean
            (optional) verbosity of the logger function
    Output:
        C_prime:list
            subset of the original free variables that successfully brought an adversarial attack
    """
    C_prime, adv = [], []  # free variables that we collect, hopefully reduced wrt C_setminus_h
    input_len = len(input_)
    iterator_ = list(range(0, input_len, window_size))
    fixed_vars = [i for i in iterator_ if i not in C_setminus_h]  # vars that are fixed when input_ is found and the function is called
    free_vars = [i for i in iterator_ if i in C_setminus_h]  # vars that are fixed when input_ is found and the function is called
    # shuffle free variables 
    if randomize is True:
        random.shuffle(free_vars)
    for i in free_vars:
        h = fixed_vars + idx2word([i], window_size, input_len)  # fix i-th window
        network = Marabou.read_tf(filename, modelType='savedModel_v2', savedModelTags=['serving_default'])  # re-initialize the network
        res = Entails(h, network, input_, input_bounds, output_constraints, weights_softmax, window_size, norm)
        if len(res) != 0:  # if there is an attack
            fixed_vars += idx2word([i], window_size, input_len)  # var will be discarded (the attack is still adversarial)
    adv = [i for i in free_vars if i not in fixed_vars]  # keep all free vars that were really adversarial
    for i in adv:
        C_prime += idx2word([i], window_size, input_len)  # expand window
    logger("{}".format(C_prime), verbose, "DEBUG")
    return C_prime

def get_costs(orig_input,window_size,word_cost_func,default_val=100):
    costs = dict()
    c = 0
    for i in orig_input:
        if i in word_cost_func.keys():
            for j in range(window_size):
                costs[c] = word_cost_func[i]
                c += 1
        else:
            for j in range(window_size):
                costs[c] = default_val
                c += 1
    return costs

def knn_smallest_explanation_linf_with_cost(model, filename, orig_input, numpy_input, word_cost_func, eps, y_hat, convex_hull_constraints, output_constraints, window_size, weights_softmax,
                                  adv_attacks=False, adv_args=(None, None), sims=10, randomize_pickfalselits=False, HS_maxlen=100, verbose=True):
    """
    Smallest Explanation API, algorithm 2 paper "Abduction-Based Explanation for ML Models", AAAI-2019.
    Input:
        model:tensorflow.keras.models
            model used to make predictions and extract the Marabou graph
        filename:string
            path to the graph in the format maraboupy.MarabouNetworkTF.MarabouNetworkTF
        orig_input:list
            list of words in the original input
        numpy_input:numpy.array
            input of the adversarial attack in numpy format
        word_cost_func:dict
            the dictionary mapping words to their costs
        eps:float
            radius of the region to find adversarial attacks
        y_hat:int
            integer that specifies the output class
        convex_hull_constraints:list
            list of convex hull constraints, one for each input word
        output_constraints:list
            list of [1st_output_var:int, 2nd_output_var:int, tolerance:float] s.t. the output constraint is expressed in the
             form (1st_output_var - 2nd_output_var <= tolerance) which is approx. (1st_output_var < 2nd_output_var)
        window_size:int
            length of the window used to encode a single variable
        weights_softmax:list
            list with numpy.arrays of weights and biases of the softmax layer
        adv_attacks:boolean
            (optional) exploit PickFalseLiterals with adversarial attack routines where number of attacks is minimized
        adv_args:tuple
            (optional) tuple (func, args) used to launch the advresarial attacks routine
        sims:int
            (optional) number of simulations (per number of variables) in the PGD adversarial attacks routine
        randomize_pickfalselits:boolean
            (optional) PickFalseLits function uses a randomized approach to refine the explanations
        HS_maxlen:int
            (optional) max size of GAMMA
        verbose:boolean
            (optional) verbosity of the logger function.
    Output:
        smallest_expl:list
            variables in the minimal explanation
        exec_time:float
            execution time in seconds from the beginning of the execution of the routine
        GAMMA:list
            list of hitting sets
    """
    start_time, exec_time = time.time(), 0
    input_ = numpy_input.flatten().tolist()
    input_len = len(input_)
    abs_input_bounds = [[abs(input_[i]-eps[int(i/window_size)][0][i%window_size]), abs(eps[int(i/window_size)][1][i%window_size]-input_[i])] for i in range(input_len)]
    GAMMA = []
    costs = get_costs(orig_input,window_size,word_cost_func)
    #print(costs)
    timer = 0 
    while True:
        # Keep length of Gamma fixed + keep only smallest elements
        if len(GAMMA) > HS_maxlen:
            GAMMA.sort(key=len)
            GAMMA = GAMMA[:HS_maxlen]
        if GAMMA != []:
            logger("Calculating HS on set of average len = {}, (size of GAMMA={})".format(sum(map(len, GAMMA))/float(len(GAMMA)), len(GAMMA)), verbose, "DEBUG")
        # Generate Minimum Hitting Set
        h = MinimumHS(GAMMA, costs, htype='smallest')
        h = idx2word(h, window_size, input_len)  # fixed vars
        logger("MinimumHS {}".format(h), verbose, "DEBUG")
        # Check that h is an optimal HS (triggered only when adv_attacks is True)
        if (len(GAMMA) > 5000 and adv_attacks is True) or (timer > 25 and adv_attacks is True):
            network = Marabou.read_tf(filename, modelType='savedModel_v2', savedModelTags=['serving_default'])  # re-initialize the network
            res = Entails(h, network, input_, convex_hull_constraints, output_constraints, weights_softmax, window_size, 'knn-linf', verbose)  # Algorithm, line 5  
            if len(res) == 0:
                break
        # Start procedure
        network = Marabou.read_tf(filename, modelType='savedModel_v2', savedModelTags=['serving_default'])  # re-initialize the network
        # 1. Look for an adversarial attack *before* Entails (at line 5 Alg. 2)
        if adv_attacks is True:
            adv_args = (adv_args[0], adv_args[1][:6] + (h,) + (abs_input_bounds,) + (adv_args[1][-1],))  # set mask on inputs that should not be tested
            found, res = adv_args[0](*adv_args[1])
            if found is False:
                logger("Adversarial attack not found on free vars {}. Run Entails on vars {}".format([f for f in range(input_len) if f not in h], h), verbose, 'DEBUG')
                if len(h) > int(0.5*input_len):
                    res = Entails(h, network, input_, convex_hull_constraints, output_constraints, weights_softmax, window_size, 'knn-linf', verbose)  # Algorithm, line 5  
                else:
                    res = {None}  # res must not be empty to skip Marabou (that is super-expensive for big models)
        else:
            logger("Run Entails on vars {}".format(h), verbose, "DEBUG")
            res = Entails(h, network, input_, convex_hull_constraints, output_constraints, weights_softmax, window_size, 'knn-linf', verbose)  # Algorithm, line 5  
        if len(res) == 0:  # return if there is no attack with h as a smallest explanation
            break
        else:
            logger("Attack found", verbose, "DEBUG")
            C_setminus_h = [c for c in range(input_len) if c not in h]  # free vars used to find an attack, naming consistent with paper notation
            pop_attacks = []
            # 2.1 Search sparse aversarial attacks with sparseRS routine
            if adv_attacks is True:
                mask = [m for m in range(input_len) if m in h]  # vars excluded from the adversarial attacks (i.e., h)
                # Run sparsePGD before trying Entails (which is usually way slower)
                _, _, pop_attacks = adv.optimize_sparseRS(model, numpy_input, abs_input_bounds, y_hat, 
                                                          num_classes=2, k=min(window_size, len(C_setminus_h)), sims=sims, mask=mask, PGDargs=(False, abs_input_bounds, 1), feature_size=window_size)
                if len(pop_attacks) != 0:
                    logger("SparseRS is successful with a total of {} attacks".format(len(pop_attacks)), verbose, '')
                    for pop in pop_attacks:
                        candidate_p = np.argwhere(numpy_input.flatten()!=pop.flatten()).flatten().tolist()
                        if candidate_p not in GAMMA:  # add *all* the attacks that have been found
                            GAMMA += [idx2word(candidate_p, window_size, input_len)]
            # 2.2 Refine the attack found with heuristic or run PickFalseLits
            if len(pop_attacks) != 0:  # SparseRS has found at least an attack
                C_setminus_h = [c for c in range(input_len) if c in pop_attacks[-1]]  # set PickFalseLits arg to the smallest attack found with SparseRS
            logger("Run PickFalseLits with pop_attacks size = {}, adv_attacks = {}".format(len(pop_attacks), adv_attacks), verbose, "DEBUG")            
            #C_prime = PickFalseLits(C_setminus_h, filename, input_, abs_input_bounds, output_constraints, window_size, randomize_pickfalselits, verbose)            
            C_prime = C_setminus_h
            if len(C_prime) > 0 and C_prime not in GAMMA:
                GAMMA += [idx2word(C_prime, window_size, input_len)]
        logger("Iteration n.{}, size of GAMMA = {}".format(timer, len(GAMMA)), verbose, "DEBUG")
        timer += 1
    exec_time = time.time() - start_time
    return h, exec_time, GAMMA
