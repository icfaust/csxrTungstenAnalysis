import scipy

# This is a set of functions to model the CSXR sightline analytically 


def normM2(vals, x1, x2, ymin, yatx):
    output = scipy.zeros(vals.shape)
    #temp = (x1+x2)/(2*x1*x2)
    F = 4*(ymin-yatx)/(x2-x1)/(x1-x2)
    output = F*(vals-x1)*(vals-x2) + yatx
    output[scipy.logical_or(vals < x1, vals > x2)] = 0. 
    return output

    
def normM4(vals, x1, x2, ymin, yatx):  
    output = scipy.zeros(vals.shape)
    avg = (x1+x2)/2
    diff = abs(x2-x1)
    output = 16*(yatx-ymin)*pow((vals-avg)/diff,4) + ymin
    output[scipy.logical_or(vals < x1, vals > x2)] = 0. 
    return output

def normMn(n, vals, x1, x2, ymin, yatx):  
    output = scipy.zeros(vals.shape)
    avg = (x1+x2)/2
    diff = abs(x2-x1)
    output = pow(2,n)*(yatx-ymin)*pow((vals-avg)/diff,n) + ymin
    output[scipy.logical_or(vals < x1, vals > x2)] = 0. 
    return output

def normMn2(n, vals, x1, x2, ymin):  
    output = scipy.zeros(vals.shape)
    avg = (x1+x2)/2
    diff = abs(x2-x1)
    output = pow(2,n)*(1-ymin)*pow((vals-avg)/diff,n) + ymin
    output[scipy.logical_or(vals < x1, vals > x2)] = 0. 
    output /= scipy.sum(output)
    return output

def normMexp(vals, x1, x2, ymin, decay=1e0):  
    output = scipy.zeros(vals.shape)
    idx = scipy.logical_and(vals > x1, vals < x2)
    output[idx] = ymin + scipy.exp(-1*(vals[idx]-x1)/decay) + scipy.exp((vals[idx]-x2)/decay) 
    output /= scipy.sum(output)
    return output
