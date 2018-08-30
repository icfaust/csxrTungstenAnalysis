import scipy
import matplotlib.pyplot as plt
import pickle


def loadBoot(name='bootstrap_lamFitsflatDiff.p'):
    temp = pickle.load(open(name,'rb'))
    output = scipy.zeros((len(temp),len(temp[0][1].x)))
    for i in xrange(len(temp)):
        output[i] = temp[i][1].x

    return output

def confidenceInterval(inp,interval=.025):
    
    length = inp.shape[1]
    interval = scipy.around(length*interval).astype(int)

    output = scipy.zeros((2,inp.shape[1]))
    for i in xrange(length):
        temp = scipy.sort(inp[:,i])
        output[0][i] = temp[interval]#lower bound
        output[1][i] = temp[::-1][interval]#upper bound
    
        #this proceedure assumes that the distribution is not multi-modal
    return output
    
