import scipy
import scipy.signal as sig
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import dd
import multiprocessing as mp
import data as SIFdata
import _gauss


def test(idx=807,shot=33349,num=4):
    data = SIFdata.SIFData(shot)
    
    temp = data.data[idx]

    axis = scipy.mgrid[0:pow(2,16)+1:num]
    output = scipy.histogram(temp,bins=axis)[0]
    plt.semilogy(axis[:-1]+num/2,output)
