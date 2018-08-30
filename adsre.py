#regex the ads data for 

import re
import scipy
import scipy.interpolate

class lineInfo(object):

    def __init__(self, name='data_to_Ian.txt'):
        txtobj = open(name,'r')
        
        self.z = []
        self.charge = []
        self.transition = []
        self.te = []
        self.pec = []
        self.parse(txtobj)
        self.te = scipy.vstack(self.te)
        #if (len(scipy.unique(self.te)) == self.te.shape[1]):
        #    self.te = scipy.unique(self.te) #same te base, drop it like its hot

        self.pec = scipy.vstack(self.pec)
        self._splines = {}


        
    def parse(self,txtobj):
        data = []
        go = 2
        for text in txtobj:
            temp = re.findall('\d{2} ',text)

            if temp and go == 2:
                go -= 1
                self.z += [int(temp[0])]
                self.charge += [int(temp[1])]
                self.transition += [re.findall('\(.*?\)',text)]

            elif go == 1:
                go -= 1
                self.te += [scipy.array(re.findall('\d\.\d*[eE][-+]\d{2}',text)).astype(float)]

            elif go == 0:
                self.pec += [scipy.array(re.findall('\d\.\d*[eE][-+]\d{2}',text)).astype(float)]
                go = 2
                              



    def __call__(self, te, i):
        try:
            return scipy.exp(self._splines[i](scipy.log(te)))
        except KeyError:
            self._splines[i] = scipy.interpolate.interp1d(scipy.log(self.te[i]),
                                                          scipy.log(self.pec[i]),
                                                          bounds_error=False,
                                                          kind='cubic',
                                                          fill_value=(scipy.NINF,scipy.log(self.pec[i,-1])))

            return scipy.exp(self._splines[i](scipy.log(te)))                                                       
