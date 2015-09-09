import numpy
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import preprocessing

def getdata():
    """
    load data from file "Neutrino.txt"
    """
    datax = numpy.genfromtxt("Neutrino.txt", usecols = (0,1,2))
    datay = numpy.genfromtxt("Neutrino.txt", usecols = (3), dtype = 'str')
    datacolor = []
    for i in range(0, datay.size):
        if(datay[i] == 'Shower'):
            datacolor.append('r')
        else:
            datacolor.append('b')
    datax = preprocessing.scale(datax)
    return datax, datay, datacolor

def Plot3D(datax, datacolor):
    """
    given a set of 3D (x,y,z) ordinates and color, make a scatter plot.
    
    datax: numpy.ndarray
    datacolor: list
    """
    fig = pylab.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(datax[:,0], datax[:,1], datax[:,2], c = datacolor, s = 100)
    pylab.show()

def PCARotation(datax, datay, rotatewith):
    """
    rotate track and shower coordinates with respect to the major directions of track particles.
    """
    pca = PCA(n_components=3)
    feature1 = datax[datay == rotatewith,:]
    feature2 = datax[datay != rotatewith,:]
    pca.fit(feature1)
    return pca.transform(feature1), pca.transform(feature2)

def Preprocessdata(d1, d2):
    """
    discard some data when z is smaller than -0.8. then remove the z coordinates.
    """
    d2 = d2[d2[:,2] > -0.8,:]
    d = numpy.vstack((d1,d2))
    t = numpy.empty(d.shape[0])
    t[:] = 0
    t[0:d1.shape[0]] = 1
    return d[:,:2], t

def Plot2DwSVM(datax, datay):
    """
    plot data and corresponding SVM results.
    """
    clf = svm.SVC()
    clf.fit(datax, datay)
    step = 0.02
    x_min, x_max = datax[:,0].min(), datax[:,0].max()
    y_min, y_max = datax[:,1].min(), datax[:,1].max()
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, step), numpy.arange(y_min, y_max, step))
    Z = clf.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pylab.contourf(xx, yy, Z, 10, cmap=pylab.cm.Oranges)
    pylab.scatter(datax[datay == 1,0], datax[datay == 1,1], c='b', s=50)
    pylab.scatter(datax[datay != 1,0], datax[datay != 1,1], c='r', s=50)
    pylab.show()
    

def transformData(data):
    """
    this function will add 
    3   sin(dec), 
    4   cos(dec), 
    5   tan(dec),
    6   sin(RA), 
    7   cos(RA),
    8   tan(RA),
    9   dec*RA, 
    10  sin(dec*RA), 
    11  cos(dec*RA), 
    12  tan(dec*RA),
    13  dec/RA, 
    14  sin(dec/RA), 
    15  cos(dec/RA),
    16  tan(dec/RA),
    17  RA/dec, 
    18  sin(RA/dec), 
    19  cos(RA/dec),
    20  tan(RA/dec) as the appends to the input array
    """
    #sin(dec)
    a = numpy.sin(data[:,1])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #cos(dec)
    a = numpy.cos(data[:,1])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #tan(dec)
    a = numpy.tan(data[:,1])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #sin(RA)
    a = numpy.sin(data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #cos(RA)
    a = numpy.cos(data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #tan(RA)
    a = numpy.tan(data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #dec*RA
    a = data[:,1]*data[:,2]
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #sin(dec*RA)
    a = numpy.sin(data[:,1]*data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #cos(dec*RA)
    a = numpy.cos(data[:,1]*data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #tan(dec*RA)
    a = numpy.tan(data[:,1]*data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #dec/RA
    a = data[:,1]/data[:,2]
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #sin(dec/RA)
    a = numpy.sin(data[:,1]/data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #cos(dec/RA)
    a = numpy.cos(data[:,1]/data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #tan(dec/RA)
    a = numpy.tan(data[:,1]/data[:,2])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #RA/dec
    a = data[:,2]/data[:,1]
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #sin(RA/dec)
    a = numpy.sin(data[:,2]/data[:,1])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #cos(RA/dec)
    a = numpy.cos(data[:,2]/data[:,1])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    #tan(RA/dec)
    a = numpy.tan(data[:,2]/data[:,1])
    a.shape = (a.shape[0], 1)
    data = numpy.concatenate((data, a), axis = 1)
    return data

