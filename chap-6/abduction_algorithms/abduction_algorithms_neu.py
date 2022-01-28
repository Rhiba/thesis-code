import itertools
import multiprocessing as mp
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow import keras
from pysat.examples.hitman import Hitman
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import linear as keras_linear
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import sys

import entails_neu
from entails_neu import entails
from hitman_cost import HitmanCost

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

def MinimumHS_mnist(subsets, 
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
        hype:string
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
    if len(subsets) == 0:
        hs = []
        return hs
    if [] in subsets:
        raise Exception("Subsets variable can't contain [] set (empty)")
    h = Hitman(bootstrap_with=subsets, htype='sorted')
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

def MinimumHS(subsets, costs,
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
        hype:string
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
    if len(subsets) == 0:
        hs = []
        return hs
    if [] in subsets:
        raise Exception("Subsets variable can't contain [] set (empty)")
    h = HitmanCost(bootstrap_with=subsets, costs=costs, htype='sorted')
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


def PickFalseLits(C_setminus_h, filename, input_, input_bounds, output_constraints, 
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
        u_bounds = [e[1] for e in input_bounds]
        l_bounds = [e[0] for e in input_bounds]
        res = entails(h, len(h), input_, len(input_), u_bounds, l_bounds, filename)
        if res != 0:  # if there is an attack
            fixed_vars += idx2word([i], window_size, input_len)  # var will be discarded (the attack is still adversarial)
    adv = [i for i in free_vars if i not in fixed_vars]  # keep all free vars that were really adversarial
    for i in adv:
        C_prime += idx2word([i], window_size, input_len)  # expand window
    logger("{}".format(C_prime), verbose, "DEBUG")
    return C_prime

def smallest_explanation_mnist(model, filename, numpy_input, epsilon, y_hat, output_constraints, empties,
                     adv_attacks=False, adv_args=(None, None), sims=10, randomize_pickfalselits=False, HS_maxlen=10000, verbose=True):
    """
    Smallest Explanation API, algorithm 2 paper "Abduction-Based Explanation for ML Models", AAAI-2019.
    Input:
        model:tensorflow.keras.models
            model used to make predictions and extract the Marabou graph
        filename:string
            path to the graph in the format maraboupy.MarabouNetworkTF.MarabouNetworkTF
        numpy_input:numpy.array
            input of the adversarial attack in numpy format
        eps:float
            value used to determine bounds, in terms of [min, max] for each input
        y_hat:int
            integer that specifies the output class
        output_constraints:list
            list of [1st_output_var:int, 2nd_output_var:int, tolerance:float] s.t. the output constraint is expressed in the
             form (1st_output_var - 2nd_output_var <= tolerance) which is approx. (1st_output_var < 2nd_output_var)
        window_size:int
            length of the window used to encode a single variable
        adv_attacks:boolean
            (optional) exploit PickFalseLiterals with adversarial attack routines where number of attacks is minimized
        adv_args:tuple
            (optional) tuple (func, args) used to launch the advresarial attacks routine
        sims:int
            (optional) number of simulations (per number of variables) in the PGD adversarial attacks routine
        randomize_pickfalselits:boolean
            (optional) PickFalseLits function uses a randomized approach to refine the explanations
        HS_maxlen:int
            (optional) max size of GAMMA. Keep it under 150-200 even for machines with lot of memory
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
    start_time = time.time()
    input_ = np.array(numpy_input.flatten().tolist(),dtype=float)
    #print('input_',input_)
    input_len = len(input_)
    #print('input_len',input_len)
    # dont bother with the "background" pixels
    input_bounds = [[0.0 if i in empties else (input_[i]-epsilon if input_[i]-epsilon >= 0 else 0), 0.0 if i in empties else (input_[i]+epsilon if input_[i]+epsilon <= 1 else 1)] for i in range(len(input_))]
    print(input_bounds)
    GAMMA = []
    timer = 0 
    while True:
        # Keep length of Gamma fixed + keep only smallest elements
        if len(GAMMA) > HS_maxlen:
            GAMMA.sort(key=len)
            GAMMA = GAMMA[:HS_maxlen]

        h = MinimumHS_mnist(GAMMA, htype='smallest')
        #h = idx2word(h, window_size, input_len)  # fixed vars
        #if len(h) == 50:
        #    break
        logger("MinimumHS {}".format(h), verbose, "DEBUG")
        u_bounds = [float(e[1]) for e in input_bounds]
        l_bounds = [float(e[0]) for e in input_bounds]
        #print('len bounds from smallest',len(u_bounds),len(l_bounds))
        res = entails(h, len(h), input_, len(input_), u_bounds, l_bounds, filename)
        if res == 0:  # return if there is no attack with h as a smallest explanation
            break
        else:
            C_setminus_h = [c for c in range(input_len) if c not in h and c not in empties]  # free vars used to find an attack, naming consistent with paper notation
            C_prime = C_setminus_h
            
            if len(C_prime) > 0 and C_prime not in GAMMA:
                #GAMMA += [idx2word(C_prime, window_size, input_len)]
                GAMMA += [C_prime]
        timer += 1
    exec_time = time.time() - start_time
    return h, exec_time, GAMMA
def smallest_explanation(model, filename, numpy_input, epsilon, y_hat, output_constraints, window_size, 
                     adv_attacks=False, adv_args=(None, None), sims=10, randomize_pickfalselits=False, HS_maxlen=10000, verbose=True):
    """
    Smallest Explanation API, algorithm 2 paper "Abduction-Based Explanation for ML Models", AAAI-2019.
    Input:
        model:tensorflow.keras.models
            model used to make predictions and extract the Marabou graph
        filename:string
            path to the graph in the format maraboupy.MarabouNetworkTF.MarabouNetworkTF
        numpy_input:numpy.array
            input of the adversarial attack in numpy format
        eps:float
            value used to determine bounds, in terms of [min, max] for each input
        y_hat:int
            integer that specifies the output class
        output_constraints:list
            list of [1st_output_var:int, 2nd_output_var:int, tolerance:float] s.t. the output constraint is expressed in the
             form (1st_output_var - 2nd_output_var <= tolerance) which is approx. (1st_output_var < 2nd_output_var)
        window_size:int
            length of the window used to encode a single variable
        adv_attacks:boolean
            (optional) exploit PickFalseLiterals with adversarial attack routines where number of attacks is minimized
        adv_args:tuple
            (optional) tuple (func, args) used to launch the advresarial attacks routine
        sims:int
            (optional) number of simulations (per number of variables) in the PGD adversarial attacks routine
        randomize_pickfalselits:boolean
            (optional) PickFalseLits function uses a randomized approach to refine the explanations
        HS_maxlen:int
            (optional) max size of GAMMA. Keep it under 150-200 even for machines with lot of memory
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
    start_time = time.time()
    input_ = np.array(numpy_input.flatten().tolist(),dtype=float)
    #print('input_',input_)
    input_len = len(input_)
    #print('input_len',input_len)
    input_bounds = [[input_[i]-epsilon if input_[i]-epsilon >= 0 else 0, input_[i]+epsilon if input_[i]+epsilon <= 1 else 1] for i in range(len(input_))]
    GAMMA = []
    timer = 0 
    while True:
        # Keep length of Gamma fixed + keep only smallest elements
        if len(GAMMA) > HS_maxlen:
            GAMMA.sort(key=len)
            GAMMA = GAMMA[:HS_maxlen]

        h = MinimumHS(GAMMA, [], htype='smallest')
        h = idx2word(h, window_size, input_len)  # fixed vars
        if len(h) == 50:
            break
        logger("MinimumHS {}".format(h), verbose, "DEBUG")
        u_bounds = [float(e[1]) for e in input_bounds]
        l_bounds = [float(e[0]) for e in input_bounds]
        #print('len bounds from smallest',len(u_bounds),len(l_bounds))
        res = entails(h, len(h), input_, len(input_), u_bounds, l_bounds, filename)
        if res == 0:  # return if there is no attack with h as a smallest explanation
            break
        else:
            C_setminus_h = [c for c in range(input_len) if c not in h]  # free vars used to find an attack, naming consistent with paper notation
            C_prime = C_setminus_h
            
            if len(C_prime) > 0 and C_prime not in GAMMA:
                GAMMA += [idx2word(C_prime, window_size, input_len)]
        timer += 1
    exec_time = time.time() - start_time
    return h, exec_time, GAMMA

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

def smallest_explanation_knn(model, filename, orig_input, numpy_input, word_cost_func, input_bounds, y_hat, output_constraints, window_size, 
                     adv_attacks=False, adv_args=(None, None), sims=10, randomize_pickfalselits=False, HS_maxlen=10000, verbose=True):
    start_time = time.time()
    input_ = np.array(numpy_input.flatten().tolist(),dtype=float)
    costs = get_costs(orig_input,window_size,word_cost_func)
    #print(orig_input,word_cost_func)
    #print(costs)
    #print('input_',input_)
    input_len = len(input_)
    #print('input_len',input_len)
    #input_bounds = [[input_[i]-epsilon if input_[i]-epsilon >= 0 else 0, input_[i]+epsilon if input_[i]+epsilon <= 1 else 1] for i in range(len(input_))]
    GAMMA = []
    timer = 0 
    while True:
        # Keep length of Gamma fixed + keep only smallest elements
        if len(GAMMA) > HS_maxlen:
            GAMMA.sort(key=len)
            GAMMA = GAMMA[:HS_maxlen]

        h = MinimumHS(GAMMA, costs, htype='smallest')
        h = idx2word(h, window_size, input_len)  # fixed vars
        if len(h) == 50:
            break
        logger("MinimumHS {}".format(h), verbose, "DEBUG")
        u_bounds = [float(e[1]) for e in input_bounds]
        l_bounds = [float(e[0]) for e in input_bounds]
        #print('len bounds from smallest',len(u_bounds),len(l_bounds))
        res = entails(h, len(h), input_, len(input_), u_bounds, l_bounds, filename)
        if res == 0:  # return if there is no attack with h as a smallest explanation
            break
        else:
            C_setminus_h = [c for c in range(input_len) if c not in h]  # free vars used to find an attack, naming consistent with paper notation
            C_prime = C_setminus_h
            
            if len(C_prime) > 0 and C_prime not in GAMMA:
                GAMMA += [idx2word(C_prime, window_size, input_len)]
        timer += 1
    exec_time = time.time() - start_time
    return h, exec_time, GAMMA
