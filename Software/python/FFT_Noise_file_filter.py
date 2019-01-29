from    matplotlib          import  pyplot    as  plt
from    scipy.fftpack       import  fftfreq
import  scipy.io.wavfile                      as  wavfile
import  numpy                                 as  np
import  scipy
import  soundfile                             as  sf

#----1-----
Overlay_FFT = []                                   # empty list to store FFT values for overlay plot 
Overlay_freqs = []                                 # empty list to store FFT values for overlay plot


def Run_FFT( wav_file ):
    OG_fs_rate, OG_Signal = wavfile.read(r"C:\Users\flobo\Documents\Gits\PD3D\3dstet\Software\Audio Files\talking.wav")
    n = OG_Signal.shape[0]
    fs_rate, signal = wavfile.read( wav_file )                  # Reads .wav file
    l_audio = len(signal.shape)                                 # Get number of channels

    if( l_audio == 2 ):                                         
        signal = signal.T[0]

    N = signal.shape[0]                             # Number of samples
    secs = N / float(fs_rate)                       # Time duration
    Ts = 1.0/fs_rate                                # Sampling interval
    t = scipy.arange(0, secs, Ts)                   # time vector

    
    print( "Opening {}".format(wav_file) )
    print( "Sampling Frequency  , f : {} ".format(fs_rate)  )
    print( "Length of audio file, t : {}s".format(secs)     )
    print( "Timestep            , Ts: {} ".format(Ts)       )
    print( "# of Samples        , N : {} ".format(N)        )
    print( "# of Channels           : {} ".format(l_audio)  )
    print( '\n' )
    
    FFT = np.fft.rfft(signal, n)                     # Full FFT
    FFT_side = FFT[range(N//2)]                     # One side FFT range

    freqs = np.fft.rfftfreq(n, d=Ts)                 # Full FFT frequency
    freqs_side = freqs[range(N//2)]                 # One side FFT frequency range
    Mag_db = 20*np.log10(FFT_side/max(FFT_side))    # Converts scale to relative db



    return( signal, FFT, FFT_side, freqs, freqs_side, t, Mag_db )

#----2-----

'''audio file'''

filename = r"C:\Users\flobo\Documents\Gits\PD3D\3dstet\Software\Audio Files\talking.wav"
signal, FFT, FFT_side, freqs, freqs_side, t, Mag_db = Run_FFT( filename )

plt.figure(1)
plt.suptitle('Audio Signal', fontsize=16)
plt.subplot(411)
p1 = plt.plot(t, signal, "g")                       # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(412)
p2 = plt.plot(freqs, abs(FFT), "r")                      # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.grid(True)

plt.subplot(413)
p3 = plt.semilogx(freqs_side, Mag_db, "b")          # plotting the fft spectrum on db scale, and Log(Hz)
plt.xlabel('Frequency Log(Hz)')
plt.ylabel('relative db single-sided')
plt.grid(True)

FFT_side = abs(FFT[ range(signal.shape[0]//64) ] )       # Zooming in y-axis
freqs_side = abs(freqs[range(signal.shape[0]//64)] )      # Zooming in x-axis           

plt.subplot(414)
p3 = plt.plot(freqs_side, abs(FFT_side), "r-.")     # plotting the positive fft spectrum (ZOOMED-IN)
plt.xlabel('Frequency (Hz)')
plt.ylabel('ZOOMED IN')
plt.grid(True)

Overlay_FFT.append( abs(FFT) )                           # stores fft values for overlay graph
Overlay_freqs.append( freqs )                       # stores frequency values for overlay graph

fft_1_x = freqs                                     # Store frequency elements of FFT2
fft_1_y = FFT                                       # Store amplitude elements of FFT2

t_signal = t
signal_signal = signal

#==========

'''Noise file'''

filename = r"C:\Users\flobo\Documents\Gits\PD3D\3dstet\Software\Audio Files\roomtone.wav"
signal, FFT, FFT_side, freqs, freqs_side, t, Mag_db = Run_FFT( filename )

plt.figure(2)
plt.suptitle('Noise Signal', fontsize=16)
plt.subplot(411)
p1 = plt.plot(t, signal, "g")                       # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(412)
p2 = plt.plot(freqs, abs(FFT), "r")                      # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.grid(True)

plt.subplot(413)
p3 = plt.semilogx(freqs_side, Mag_db, "b")          # plotting the fft spectrum on db scale, and Log(Hz)
plt.xlabel('Frequency (Log(Hz)')
plt.ylabel('db single-sided')
plt.grid(True)

FFT_side = FFT[ range(signal.shape[0]//64) ]        # Zooming in y-axis
freqs_side = freqs[range(signal.shape[0]//64)]      # Zooming in x-axis            

plt.subplot(414)
p3 = plt.plot(freqs_side, abs(FFT_side), "k:")      # plotting the positive fft spectrum (ZOOMED-IN)
plt.xlabel('Frequency (Hz)')
plt.ylabel('ZOOMED IN')
plt.grid(True)

Overlay_FFT.append( abs(FFT) )                           # stores fft values for overlay graph
Overlay_freqs.append( freqs )                       # stores frequency values for overlay graph

fft_2_x = freqs                                     # Store frequency elements of FFT2
fft_2_y = FFT                                       # Store amplitude elements of FFT2
 
#----3-----

Signal_FFT, Noise_FFT = Overlay_FFT[0], Overlay_FFT[1]
Signal_freqs, Noise_freqs = Overlay_freqs[0], Overlay_freqs[1]

plt.figure(3)
plt.suptitle('FFT Overlay', fontsize=16)

plt.subplot(111)
plt.plot(Signal_freqs, Signal_FFT, "r-.")           # plotting the Signal fft spectrum
plt.plot( Noise_freqs, Noise_FFT , "k:" )           # plotting the Noise fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

#----4-----

pad_num_x = abs( len(fft_1_x) - len(fft_2_x) )                # Calculate how many zeros we need to pad with        
pad_num_y = abs( len(fft_1_y) - len(fft_2_y) )                # Calculate how many zeros we need to pad with

print( "Before padding..." )                                  # [INFO]
print( "Length of FFT1_x: {}".format(len(fft_1_x)) )          # [INFO]
print( "Length of FFT1_y: {}".format(len(fft_1_y)) )          # [INFO]
print( "Length of FFT2_x: {}".format(len(fft_2_x)) )          # [INFO]
print( "Length of FFT2_y: {}\n".format(len(fft_2_y)) )        # [INFO]

fft_2_x = np.pad( fft_2_x, (0, pad_num_x), 'constant' )       # Pad array with zeros to the right side
fft_2_y = np.pad( fft_2_y, (0, pad_num_y), 'constant' )       # Pad array with zeros to the right side

print( "Padding with {} zeros\n".format(pad_num_x) )          # [INFO]

print( "After padding..." )                                   # [INFO]
print( "Length of FFT1_x: {}".format(len(fft_1_x)) )          # [INFO]
print( "Length of FFT1_y: {}".format(len(fft_1_y)) )          # [INFO]
print( "Length of FFT2_x: {}".format(len(fft_2_x)) )          # [INFO]
print( "Length of FFT2_y: {}".format(len(fft_2_y)) )          # [INFO]

plt.figure(6)                                                 # Create figure
x = fft_1_x                                                   # fft frequency axis
y = fft_1_y - fft_2_y                               # Subtract one fft from the other
plt.suptitle('Subtracting ROOMTONE.wav from TALKING.wav', fontsize=16)
plt.plot( fft_1_x, fft_1_y , "r-." )
plt.plot(x, abs(y), 'k:')                                     # Plot FFT
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

#plt.show()

#----5-----

OG_fs_rate, OG_Signal = wavfile.read(r"C:\Users\flobo\Documents\Gits\PD3D\3dstet\Software\Audio Files\talking.wav")
n = OG_Signal.shape[0]

#inverted_fft = OG_Signal
inverted_fft = np.fft.irfft(y, n)
#inverted_fft = np.fft.irfft(np.fft.fft(OG_Signal))
inverted_fft = inverted_fft.astype(np.int16)


##positive_ifft = inverted_fft[range(len(inverted_fft)//2)]

#sf.write(r"C:\Users\pd3dlab\Desktop\Samer FFT\reformated_fft.wav", inverted_fft, 44100)
wavfile.write(r"C:\Users\flobo\Documents\Gits\PD3D\3dstet\Software\Audio Files\reformated_fft.wav", 44100, inverted_fft)

##
##
##dual_channel_ifft = np.array( [ [inverted_fft], [inverted_fft] ] )
##
##reformed_wav = wavfile.write(r"C:\Users\pd3dlab\Desktop\Samer FFT\reformated_fft.wav", 44100, inverted_fft.real)

##plt.figure(7)
##plt.suptitle('original vs. reformated', fontsize=16)
##plt.subplot(211)
##p1 = plt.plot(t_signal, signal_signal, "g")                       # plotting the signal
##plt.xlabel('Time')
##plt.ylabel('Amplitude')
##plt.grid(True)
##
##plt.subplot(212)
##p2 = plt.plot(t_signal, inverted_fft, "r")                      # plotting the complete fft spectrum
##plt.xlabel('Time)')
##plt.ylabel('Amplitude')
##plt.grid(True)





##Signal_FFT_Array = np.array([[fft_1_x], [fft_1_y]])
##Noise_FFT_Array = np.array([[fft_2_x], [fft_2_y]])
##
##Filtered_FFT = np.subtract(Signal_FFT_Array, Noise_FFT_Array)
##fft_filtered_y = np.zeros(len(fft_1_y))
##
##for n in range(len(fft_1_x)):
##    if fft_2_x[n] == fft_1_x[n]:
##        fft_filtered_y[n] = fft_1_y[n] - fft_2_y[n]
        





##''' Save modified wav file to file'''
##
##def to_integer(signal):
##    # Take samples in [-1, 1] and scale to 16-bit integers,
##    # values between -2^15 and 2^15 - 1.
##    signal /= max(signal)
##    return np.int16(signal*(2**15 - 1))
##
##fft_data = np.array( y )
##filename_filtered = r"C:\Users\Samer Armaly\Desktop\modified_fft.wav"
##inverse_fft = np.fft.ifft( fft_data )
##freq_sampling = 44100
###inverse_fft = to_integer( inverse_fft )
##wavfile.write(filename_filtered, freq_sampling, np.real(inverse_fft))
##
##filename = r"C:\Users\Samer Armaly\Desktop\talking.wav"
##signal, FFT, FFT_side, freqs, freqs_side, t = run_FFT( filename )
##print(inverse_fft)
##plt.figure(6)                                                                                  # Same ^
##plt.suptitle('filtered.wav', fontsize=16)
##p1 = plt.plot(t, inverse_fft, "k:")                                   
##plt.xlabel('Time')
##plt.ylabel('Amplitude')
##plt.grid(True)
