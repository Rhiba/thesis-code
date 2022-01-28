import ctypes
import numpy as np

libname = "/home/rhiba/packages/lib/libentails.so"
c_lib = ctypes.CDLL(libname)

intp = ctypes.POINTER(ctypes.c_int)
floatp = ctypes.POINTER(ctypes.c_float)

c_lib.entails.restype = ctypes.c_int
c_lib.entails.argtypes = [intp,ctypes.c_int,floatp,ctypes.c_int,floatp,floatp,ctypes.c_char_p]

#int entails( int *h, int h_size, float *input, int input_size, float *u_bounds, float *l_bounds, char* network_path);
def entails(h,h_size,inp,inp_size,u_bounds,l_bounds,network_path):
    # needed for c_char_p type
    network_path = network_path.encode('utf-8')

    h = np.array(h,dtype=ctypes.c_int)
    inp = np.array(inp,dtype=ctypes.c_float)
    u_bounds = np.array(u_bounds,dtype=ctypes.c_float)
    l_bounds = np.array(l_bounds,dtype=ctypes.c_float)

    #print('entails.py')
    #print(u_bounds)
    #print(l_bounds,flush=True)
    #print(len(u_bounds),len(l_bounds),flush=True)
    #print('passed in len',inp_size)
    #print('---')


    h_c = h.ctypes.data_as(intp)
    inp_c = inp.ctypes.data_as(floatp)
    u_bounds = u_bounds.ctypes.data_as(floatp)
    l_bounds = l_bounds.ctypes.data_as(floatp)
    
    res = c_lib.entails(h_c,h_size,inp_c,inp_size,u_bounds,l_bounds,network_path)
    return res


if __name__ == "__main__":
    # 5d 10len input (for fc)
    #h = [i for i in range(5*10)]
    # 5d 16len input (for cnn)
    h = [i for i in range(16*5)]
    h_size = len(h)
    # 5d 10len input (for fc)
    #inp = [0.5,0.5,0.5,0.5,0.5,0.6158970519900322,0.6500097215175629,0.5419477485120296,0.5759362429380417,0.6085240393877029,0.6028607115149498,0.5812925100326538,0.5288058891892433,0.5238163638859987,0.5813853070139885,0.5316681489348412,0.5259015411138535,0.5261953305453062,0.5069217910058796,0.5252024382352829,0.6301251798868179,0.6642986238002777,0.5501794219017029,0.6265065670013428,0.6569506227970123,0.4825815800577402,0.43560905009508133,0.4821596648544073,0.42361055314540863,0.45672742277383804,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    # 5d 16len input (for cnn)
    inp = [0.4614686667919159, 0.514161229133606, 0.49704471230506897, 0.5023226141929626, 0.4676406681537628, 0.5242740511894226, 0.5189854502677917, 0.5091223120689392, 0.4824753701686859, 0.5210372805595398, 0.4937596321105957, 0.5011305809020996, 0.5189799666404724, 0.4832644462585449, 0.5166200399398804, 0.479758620262146, 0.5013266205787659, 0.45646095275878906, 0.5186120271682739, 0.5324658155441284, 0.5091118812561035, 0.5377885103225708, 0.4667411744594574, 0.5166879892349243, 0.5141173601150513, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    inp_size = len(inp)
    eps = 0.003
    ub = [min(x+eps,1) for x in inp]
    lb = [max(x-eps,0) for x in inp]
    # net_path = 'models/neurify_fc_tmp.nnet' <- 5d 10, norm, fc
    # net_path = 'models/neurify_cnn_tmp.nnet' <- 5d 16 (4,4,5), norm, cnn
    net_path = 'models/neurify_cnn_tmp.nnet'
    is_adv = entails(h,h_size,inp,inp_size,ub,lb,net_path)
    print("is adv:",is_adv)

    h = [0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,35,36,37,38,39]
    h_size = len(h)
    is_adv = entails(h,h_size,inp,inp_size,ub,lb,net_path)
    print("is adv:",is_adv)

    h = [i for i in range(10*5)]
    h_size = len(h)
    is_adv = entails(h,h_size,inp,inp_size,ub,lb,net_path)
    print("is adv:",is_adv)
