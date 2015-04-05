import numpy as np 
import scipy.signal as ss

def qea(im):
    """
    Quasi-eigen approximation function
    Input
    - im : 1d vector that contains a time series
    Ouput
    - ia : instantaneous amplitude
    - ip : instantaneous phase
    - ifeq: instantaneous frequency
    """
    im = im.ravel()
    # computes analytic signal
    H = ss.hilbert(im)
    H = im+1j*H
    # obtain IA and IP from analytic signal
    ia = np.abs(H)
    ip = np.angle(H)
    # obtain IF using QEA function
    h1 = H[1:-1]
    h0 = H[:-2]
    h2 = H[2:]
    ifeq = np.real(np.arccos((h2+h0)/(2*h1))/np.pi/2)
    # pad extremes copying
    ifeq= np.hstack((ifeq[:1],ifeq,ifeq[-1:]))
    return(ia,ip,ifeq)

def amfm_CCA(im):
    """
    Channel component analysis for AM-FM
    Input
    - im : 1d vector that contains a time series
    Output
    - ia : Instantaneous amplitude computed for 3 channels
    - ip : Instantaneous phase computed for 3 channels
    - ifeq: Instantaneous frequency computed for 3 channels
    """
    # Filter bank
    bp = [ ]
    # Low pass 0 0.02
    bp.append(ss.remez(50, [0, 0.02, 
                            0.05, 0.5], 
                       [1,0]))
    # Pass band 0.02 0.25
    bp.append(ss.remez(50, [0, 0.02, 0.05,
                            0.20, 0.25, 0.5],
                       [0, 1, 0]))
    # High pass 0.25 0.5
    bp.append(ss.remez(50, [0, 0.20,
                            0.25, 0.5],
                       [0, 1],
                       type = "hilbert"))
    # apply filterbank
    filt = lambda x: ss.convolve(im,x,'same')
    in_channels = map(filt,bp)
    # compute IA, IP and IF from filterbank output
    out_channels = map(qea,in_channels)
    # Organize results into a matrix of channels by time points
    ia = []
    ip = []
    ifeq = []
    for chan in out_channels:
        ia.append(chan[0])
        ip.append(chan[1])
        ifeq.append(chan[2])
    ia = np.array(ia)
    ip = np.array(ip)
    ifeq = np.array(ifeq)
    return(ia,ip,ifeq)

def amfm_DCA(im):
    """
    Dominant component analysis for AM-FM
    Input 
    - im : 1d vector that contains a time series
    Ouput
    - ia : instantaneous amplitude
    - ip : instantaneous phase
    - ifeq: instantaneous frequency
    """
    ia,ip,ifeq = amfm_CCA(im)
    t = np.argmax(ia, axis = 0)
    ifeq = [f[t[n]] for n,f in enumerate(ifeq.T)]
    ip = [f[t[n]] for n,f in enumerate(ip.T)]
    ia = ia.max(axis=0)
    return(ia,ip,ifeq)
        
if __name__=="__main__":
    # Demo for 1d Data
    import matplotlib.pyplot as plt
    # number of time points
    N = 1000
    # Generate a chirp that sweeps all frequencies
    c = ss.chirp(range(N),0,N-1,0.49)
    # Compute fft for plot
    C = np.fft.fft(c, N*10)
    # This is the groundtruth IF
    f = 0.5*np.arange(N)/(N-1)
    # Compute AM-FM DCA 
    ia,ip,ifeq = amfm_DCA(c)
    # plot results
    plt.subplot(311)
    plt.plot(c)
    plt.title("Time series")
    plt.subplot(312)
    plt.plot(np.fft.fftfreq(N*10), np.abs(C),'.')
    plt.title("Frequency spectrum")
    plt.subplot(313)
    plt.plot(f)
    plt.plot(ifeq)
    plt.legend(["Ideal","Estimated"],loc = "best")
    plt.title("Frequency vs time")
    plt.show()


