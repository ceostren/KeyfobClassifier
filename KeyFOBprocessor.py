
import numpy as np
from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay


fileslst = ['civic_samp1.fc32','civic_samp2.fc32','civic_samp3.fc32', 'marc_sub_samp1.fc32','marc_sub_samp2.fc32','accord_samp1.fc32','accord_samp2.fc32','accord_samp3.fc32','noise_samp1.fc32','noise_samp2.fc32','noise_samp3.fc32','subout_samp1.fc32','subout_samp2.fc32','subout_samp3.fc32']

noise_added_samples = []

for f in fileslst:
    label = 'Empty'
    if "civic" in f:
        label = 'Civic'
    elif "marc_sub" in f:
        label = 'Forester'
    elif "accord" in f:
        label = 'Accord'
    elif "subout" in f:
        label = 'Outback'
    
    # Read in signal
    x = np.fromfile(f, dtype=np.complex64)
    sample_rate = 1e6
    #center_freq = 434e6
    offset = 5e5



    # Noise generation
    N = len(x)
    nlevel = [.000001,.00001,.00005,.00009,.0001, .0005, .001,.005,.009, .01, .05,.1,.3,.5,.7,.9]
    
    for l in nlevel:
        noise = (np.random.randn(N)+1j*np.random.randn(N))/np.sqrt(2)
        x_noise = x + (l*noise)
        SNR = np.var(x)/np.var(l*noise)
        #print(10*np.log10(np.var(x)))
        #print(10*np.log10(SNR))
        noise_added_samples.append([x_noise,label,l])

data = []
labels = []
for ns in noise_added_samples:
    x = ns[0]
    fft_size = 128
    num_rows = int(np.floor(len(x)/fft_size))
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)

    # plt.imshow(spectrogram, aspect='auto', extent = [-sample_rate/2/1e6, sample_rate/2/1e6, 0, len(x)/sample_rate])
    # plt.xlabel("Frequency [MHz]")
    # plt.ylabel("Time [s]")
    # plt.title(ns[1] + " Sample with noise x"+ str(ns[2]))
    #plt.show()

    spectrogram = np.mean(spectrogram,axis=0)
    #plt.plot(spectrogram)
    #plt.show()
    data.append(spectrogram)
    labels.append(ns[1])
print(len(data))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)
print(y_test)
lin_clf = svm.LinearSVC(C=1e-5)
lin_clf.fit(X_train,y_train)
print(lin_clf.score(X_test,y_test))
y_pred = lin_clf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


realtest = np.fromfile('spect_out_sub_hond_5_each.c64', dtype=np.complex64)
start = 0
end = len(realtest)
step = 256000
for j in range(start, end-step, step//2):
    s = j
    cur_sample = realtest[s:s+step]
    x = cur_sample
    fft_size = 128
    num_rows = int(np.floor(len(x)/fft_size))
    spectrogram = np.zeros((num_rows, fft_size))
    try:
        for i in range(num_rows):
            spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)

        plt.imshow(spectrogram, aspect='auto', extent = [-sample_rate/2/1e6, sample_rate/2/1e6, 0, len(x)/sample_rate])
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
        #plt.show()

        spectrogram = np.mean(spectrogram,axis=0)
        #plt.plot(spectrogram)
        #plt.show()
        print(lin_clf.predict(spectrogram.reshape(1,-1)))
    except:
            print("Error")

