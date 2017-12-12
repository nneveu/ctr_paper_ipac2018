import numpy as np
import scipy.io
from plotting import *
import matplotlib.pyplot as plt

file1 = 'Sep21short.mat'


mat1 = scipy.io.loadmat(file1)

print mat1

charge = np.ndarray.flatten(mat1['q1'])
signal = np.ndarray.flatten(mat1['q'])
scope = np.ndarray.flatten(mat1['lfd'])

print len(charge)

#np.savetxt('scope',np.around(scope,decimals=2),fmt='%.2f')
#scope = np.loadtxt('scope', dtype = np.float)
#scope = np.around(scope,decimals=2)

meanq = np.mean(charge)
stdq = np.std(charge)
print meanq,stdq

#pre-filter
#zeros = np.where(charge<meanq-3.0*stdq)
#charge[zeros] = meanq

#band width in sigma
band = 1.8

## pre filter
points = np.where((charge>meanq-band/2.0*stdq) & (charge<meanq+band/2.0*stdq))
charge = charge[points]
signal = signal[points]
scope = scope[points]
norm = signal / charge


print len(signal), " points selected"

def smooth(x,y):
    from scipy.interpolate import spline
    xnew = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between T.min and T.max
    ynew = spline(x,y,xnew)
    return xnew,ynew


def smartmean(charge,signal,scope,dt,num):

#minimal and maximal position of the interferometer mirror
    minpos = np.min(scope)
    maxpos = np.max(scope)

#filtered measurements with respect to charge fluctuations
    mirrorpos =  np.zeros(num)
    charge_f = np.zeros(num)
    signal_f = np.zeros(num)
    std_f = np.zeros(num)
    
    norm_f = np.zeros(num)

    for i in range (0,num):
        pos = minpos + i*dt
        print pos
        mask = np.where((scope>pos-dt/2.0) & (scope<pos+dt/2.0))
        print len(scope[mask])
        mirrorpos[i] = np.mean(scope[mask])
        norm_f[i] = np.mean(signal[mask]/charge[mask]/np.max(norm))
        std_f[i] = np.std(signal[mask]/charge[mask]/np.max(norm))
            #norm_f = norm_f / np.max(norm)
            
    return mirrorpos,norm_f,std_f

plt.figure()

plt.plot(2*scope,norm/np.max(norm),'o',alpha=0.1)

xf,yf, yerr = smartmean(charge, signal,scope,0.1,36) #96 is number of mirror positions
xs,ys = smooth(2*xf,yf)

plt.errorbar(2*xf,yf,yerr=yerr,fmt='.')
plt.plot(xs,ys,'-',color='blue')
plt.xlabel('Path difference (mm)')
plt.ylabel('Interference (arb. units)')
#plt.plot(scope,norm0/np.max(y0),'o',alpha=0.15)

plt.show()


#norm = np.reshape(norm0,(len(norm0)/10,10))
#norm = np.mean(norm,axis=1)

plt.figure()
plt.plot(scope, signal, 'o')
#x1,y1 = mat1['lfds'],mat1['qs']
#plt.plot(x1[0,:],y1[0,:]/np.max(y1[0,:]),'-')
plt.show()


x0,y0 = mat1['lfds'],norm0

x1,y1 = mat1['lfds'],mat1['qs']



plt.figure()
plt.plot(x0[0,:],y0/np.max(y0),'-',label=r'masked beam')
plt.plot(scope,norm0/np.max(y0),'o',alpha=0.15)
plt.ylim(0.2,1.3)

#plt.plot(x1[0,:],y1[0,:]/np.max(y1[0,:]),'-',label='round beam (new plate)')
#plt.plot(x2[0,:],y2[0,:]/np.max(y2[0,:]),'-',label='slit')
#plt.plot(x3[0,:],y3[0,:]/np.max(y3[0,:]),'-',label='b2')
#plt.plot(x4[0,:],y4[0,:]/np.max(y4[0,:]),'-',label='slit2')
#plt.plot(x5[0,:],y5[0,:]/np.max(y5[0,:]),'-',label='dark')

plt.xlim(0,-12)
plt.legend(loc='lower right')
plt.xlabel('Position x (mm)')
plt.ylabel('Interference (norm.)')

plt.figure()
#f1 = np.fft.fft(y1/np.max(y1))

f1 = np.fft.fft(y1[0,:]/np.max(y1[0,:]))




freq = np.fft.fftfreq(len(f1),d = ((0.2/1.0e3/3.0e8)))/1.0e9
#freq = freq[0:len(freq)/2]

print np.shape(f1), np.shape(freq)
print np.abs(f1)

plt.plot(freq[0:len(freq)/2],np.abs(f1[0:len(f1)/2]),'o-',label='round beam (new plate)')
plt.plot(freq[0:len(freq)/2],np.abs(f2[0:len(f2)/2]),'o-',label='slit')
#plt.plot(freq[0:len(freq)/2],np.abs(f3[0:len(f3)/2]),'o-',label='b2')
plt.plot(freq[0:len(freq)/2],np.abs(f4[0:len(f4)/2]),'o-',label='slit2')
#plt.plot(freq[0:len(freq)/2],np.abs(f5[0:len(f5)/2]),'o-',label='dark')


plt.legend(loc='upper right')

plt.ylim(0,10)
plt.xlim(75,np.max(freq)/2)
plt.xlabel(r'$\omega$ (GHz)')

plt.show()
