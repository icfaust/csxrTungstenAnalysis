import sys
import matplotlib.pyplot as plt
#import sqlite3
sys.path.append('/afs/ipp-garching.mpg.de/home/i/ianf/codes/python/TRIPPy-master')
import TRIPPy.beam
import TRIPPy.geometry
import TRIPPy.invert
import scipy
import scipy.interpolate
import scipy.io
import eqtools
import GIWprofiles
import dd
import sqlite3 
import multiprocessing


_dbname = 'SIFweight2015.db'
_dbname2 = 'SIFreduced2015.db'

def genSIF(tokamak):
    """ Use the TRIPPy system to yield the spectrometer sightline """

    #set up two points for extracting
    vec1 = TRIPPy.geometry.Vecr([1.785,0.,1.08])
    pt1 = TRIPPy.geometry.Point(vec1,tokamak)

    vec2 = TRIPPy.geometry.Vecr([1.785,0.,1.06])
    pt2 = TRIPPy.geometry.Point(vec2,tokamak)
    
    beam = TRIPPy.beam.Ray(pt1,pt2)
    tokamak.trace(beam) #modify beam to yield the proper in and exit points in the tokamak

    return beam #this contains the functionality to do the analysis.

def genInp(num=1e-3,shot=34228):
    tok = TRIPPy.Tokamak(eqtools.AUGDDData(34228))
    ray = genSIF(tok)
    print(ray.norm.s)
    #inp = scipy.linspace(ray.norm.s[-2],ray.norm.s[-1],num)
    inp = scipy.mgrid[ray.norm.s[-2]:ray.norm.s[-1]:num]
    r = ray(inp).r()[0]
    z = ray(inp).r()[2]
    l = inp - inp[0]
    return [r,z,l]


def goodGIW(time, shot, name="c_W"):
    """extract values which are only valid, otherwise place in zeros"""
    temp = GIWData(shot, data=name)
    interp = scipy.interpolate.interp1d(temp.time,
                                        temp.data,
                                        bounds_error=False)
    output = interp(time)
    output[ scipy.logical_not(scipy.isfinite(output))] = 0. #force all bad values to zero
    return output


def GIWData(shot, filename="GIW", data="c_W"):
    shotfile = dd.shotfile(filename,shot)
    return shotfile(data)


def calcArealDens(l, te, halfte, rho, tped, tcore, nped, ncore):
        
    minrho = scipy.argmin(rho)
    spline1 = scipy.interpolate.interp1d(rho[:minrho+1],
                                         l[:minrho+1],
                                         bounds_error=False,
                                         kind='linear')
        
    spline2 = scipy.interpolate.interp1d(rho[minrho:],
                                         l[minrho:],
                                         bounds_error=False,
                                         kind='linear')
    
    # step 3, find te rho locations
    ne = GIWprofiles.ne(GIWprofiles.te2rho2(te, tcore, tped), ncore, nped)

    rhohalfte = GIWprofiles.te2rho2(halfte, tcore, tped)
    
    bounds =scipy.array([rho[minrho],1.])
    boundte = GIWprofiles.te(bounds, tcore, tped)
    bndidx = scipy.searchsorted(halfte, boundte) #add proper endpoints to the temperature array
    rhohalfte = scipy.insert(rhohalfte, bndidx, bounds) #assumes that Te is positively increasing
    
    #step 4, find l location for those 1/2 te locations AND rho=1 for endpoints  
    l1 = spline1(rhohalfte)
    deltal1 = abs(l1[:-1] - l1[1:])
    deltal1[scipy.logical_not(scipy.isfinite(deltal1))] = 0.

    
    l2 = spline2(rhohalfte)
    deltal2 = abs(l2[:-1] - l2[1:])
    deltal2[scipy.logical_not(scipy.isfinite(deltal2))] = 0.
    
    #plt.semilogx(te,ne*(deltal1*deltal2)/1e19, '.')
    #plt.xlabel('deltal2')
    #plt.show()

    return pow(ne,2)*(deltal1+deltal2)


def weights(inp, te, shot, time):
    """ pull out data vs time necessary to calculate the weights"""
    #condition initialized values

    output = scipy.zeros((len(time), len(te)))
    r = inp[0]
    z = inp[1]
    l = inp[2]
    
    #load GIW ne, Te data
    
    tped = goodGIW(time, shot, name="t_e_ped")
    tcore = goodGIW(time, shot, name="t_e_core")

    nped = goodGIW(time, shot, name="n_e_ped")
    ncore = goodGIW(time, shot, name="n_e_core")

    good = scipy.arange(len(time))[scipy.logical_and(ncore !=0, tcore != 0)] #take only good data

    #use spline of the GIW data to solve for the proper Te, otherwise dont evaluate

    logte = scipy.log(te)
    halfte = scipy.exp(logte[1:]/2. + logte[:-1]/2.) #not going to worry about endpoints
    # because I trust te is big enough to be larger than the range of the profile
    
    #step 1, use array of r,z values to solve for rho
    
    eq = eqtools.AUGDDData(shot)
    rho = eq.rz2rho('psinorm', r, z, time[good], each_t=True, sqrt=True) #solve at each time
    idx = 0

    #step 2, construct 2 splines of l(rho)
    for i in good:
        output[i] = calcArealDens(l, te, halfte, rho[idx], tped[i], tcore[i], nped[i], ncore[i])
        output[i][scipy.logical_not(scipy.isfinite(output[i]))] = 0.
        idx+=1
        #print(idx),

    #step 8, return values for storage
    return output


def weights2(inp, te, shot, time):
    """ pull out data vs time necessary to calculate the weights"""
    #condition initialized values

    output = scipy.zeros((len(time),2))
    r = inp[0]
    z = inp[1]
    l = inp[2]
    
    #load GIW ne, Te data
    
    tped = goodGIW(time, shot, name="t_e_ped")
    tcore = goodGIW(time, shot, name="t_e_core")

    nped = goodGIW(time, shot, name="n_e_ped")
    ncore = goodGIW(time, shot, name="n_e_core")

    good = scipy.arange(len(time))[scipy.logical_and(ncore !=0, tcore != 0)] #take only good data

    #use spline of the GIW data to solve for the proper Te, otherwise dont evaluate

    logte = scipy.log(te)
    halfte = scipy.exp(logte[1:]/2. + logte[:-1]/2.) #not going to worry about endpoints
    # because I trust te is big enough to be larger than the range of the profile
    
    #step 1, use array of r,z values to solve for rho
    
    eq = eqtools.AUGDDData(shot)
    rho = eq.rz2rho('psinorm', r, z, time[good], each_t=True, sqrt=True) #solve at each time
    rhomin = scipy.nanmin(rho,axis=1)
    temax = GIWprofiles.te(rhomin, tcore[good], tped[good])

    idx = 0

    #step 2, construct 2 splines of l(rho)
    for i in good:
        temp = calcArealDens(l, te, halfte, rho[idx], tped[i], tcore[i], nped[i], ncore[i])
        temp[scipy.logical_not(scipy.isfinite(temp))] = 0.
        temp = scipy.sum(temp)
        output[i,0] = temp
        output[i,1] = temax[idx]
        idx+=1
        #print(idx),

    #step 8, return values for storage
    return output


def loadTe(name='W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave'):
    data = scipy.io.readsav(name)
    return data['en']


def run(shotlist, timelist, idx, name='shots2016'):
    conn = sqlite3.connect(_dbname)

    unishots = scipy.unique(shotlist)
    te = loadTe()
    rzl = genInp()


    for i in unishots:
        print(i)
        inp = shotlist == i
        results = weights(rzl, te, i, timelist[inp])

        writeData(idx[inp], results, conn, name)

    conn.close()


def run2(shotlist, timelist, idx, name='shots2016', serial=True):
    conn = sqlite3.connect(_dbname)

    unishots = scipy.unique(shotlist)
    te = loadTe()
    rzl = genInp()

    if serial:
        for i in unishots:
            print(i)
            inp = shotlist == i
            results = weights(rzl, te, i, timelist[inp])

            writeData(idx[inp], results, conn, name)

        conn.close()
    else:        

        index=0       
        lim = len(unishots)
        while index < lim:
            num =35
            if lim-index < num:
                num = lim-index
                print(num)

            pool = multiprocessing.Pool(num)
            output = {}
            indexout = {}
            for i in xrange(num):
                inp = shotlist == unishots[index]
                if index < lim:
                    indexout[i] = idx[inp]
                    output[i] = pool.apply_async(weights,(rzl, te, unishots[index], timelist[inp]))
                index += 1


            pool.close()
            pool.join()
            results = scipy.array([output[i].get() for i in output]) #breakdown the 100 shot chunks and write the data to the sql database

            #return indexout, results, output
            print('   ')
            print('writing to shot: '+str(inp))
            for i in xrange(len(results)):
                writeData(indexout[i], results[i], conn, name)

        conn.close()


def run3(shotlist, timelist, idx, name='shots2016', serial=True):
    conn = sqlite3.connect(_dbname2)

    unishots = scipy.unique(shotlist)
    te = loadTe()
    rzl = genInp()

    if serial:
        for i in unishots:
            print(i)
            inp = shotlist == i
            results = weights2(rzl, te, i, timelist[inp])

            writeData(idx[inp], results, conn, name)

        conn.close()
    else:        

        index=0       
        lim = len(unishots)
        while index < lim:
            num =35
            if lim-index < num:
                num = lim-index
                print(num)

            pool = multiprocessing.Pool(num)
            output = {}
            indexout = {}
            for i in xrange(num):
                inp = shotlist == unishots[index]
                if index < lim:
                    indexout[i] = idx[inp]
                    output[i] = pool.apply_async(weights2,(rzl, te, unishots[index], timelist[inp]))
                index += 1


            pool.close()
            pool.join()
            results = scipy.array([output[i].get() for i in output]) #breakdown the 100 shot chunks and write the data to the sql database

            #return indexout, results, output
            print('   ')
            print('writing to shot: '+str(inp))
            for i in xrange(len(results)):
                writeData(indexout[i], results[i], conn, name)

        conn.close()




def genTable(name='shots2015'):

    te = loadTe()
    #print(len(temp))

    conn = sqlite3.connect(_dbname)
    c = conn.cursor()
    temp = 'CREATE TABLE '+name+' (id INTEGER PRIMARY KEY ASC '
    for i in xrange(len(te)):
        temp += ', val'+str(i)+' REAL'

    temp += ')'

    #print(temp)
    c.execute(temp)
    conn.commit()
    conn.close()

def genTable2(name='shots2016'):

    conn = sqlite3.connect(_dbname2)
    c = conn.cursor()
    temp = 'CREATE TABLE '+name+' (id INTEGER PRIMARY KEY ASC , sumM REAL, maxT REAL)'

    #print(temp)
    c.execute(temp)
    conn.commit()
    conn.close()


def writeData(idx, results, conn, name):
    """ this function writes to the SQL database the new data derived from fitData
    the format is as follow:

    RMSE of fit, shotnumber, time, params (with peak val, offset, width)"""
    c = conn.cursor()
    text = 'INSERT INTO '+name+' VALUES('

    for i in xrange(len(results)):
        newstr = text+str(idx[i])
        for j in results[i]:
            newstr +=', '+str(j)
            
        newstr += ')'
        #print(newstr)
        c.execute(newstr)

    conn.commit() # write data to database
