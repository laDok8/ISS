from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import cmath
import os


# ISS projekt
# xdokou14

def dft(inputf, padding=0):
    mmax = len(inputf[0])
    if mmax < padding:
        mmax = padding
    f = np.empty((0, mmax), np.complex)
    for sample in inputf:
        f0 = []
        if len(sample) < padding:
            sample = np.append(sample, np.zeros(padding - len(sample)))
        N = len(sample)
        for p in range(N):
            a = 0j
            for k in range(N):
                a += sample[k] * cmath.exp(-2j * cmath.pi * p * k * (1 / N))
            f0 = np.append(f0, a)
        f = np.append(f, f0.reshape((1, mmax)), axis=0)
    return f


def idft(inputf, padding=0):
    inputf = inputf.reshape((1, len(inputf)))
    mmax = len(inputf[0])
    if mmax < padding:
        mmax = padding
    for sample in inputf:
        f0 = []
        if len(sample) < padding:
            sample = np.append(sample, np.zeros(padding - len(sample)))
        N = len(sample)
        for p in range(N):
            a = 0j
            for k in range(N):
                a += sample[k] * cmath.exp(2j * cmath.pi * p * k * (1 / N))
            a /= N
            f0 = np.append(f0, a)
        return np.array(f0)

maskoff_tone, fs = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_tone.wav")
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_sentence.wav", maskoff_tone, fs)
maskon_tone, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskon_tone.wav")
max_corel = 0
max1 = 0
max2 = 0
for i in np.arange(maskoff_tone.size - fs, step=1000):
    if i + fs > maskoff_tone.size:
        break
    sub_sound = maskoff_tone[i:i + fs]
    list_corel = np.correlate(maskon_tone, sub_sound)
    for i2 in range(list_corel.size):
        if list_corel[i2] > max_corel:
            max_corel = list_corel[i2]
            max1 = i
            max2 = i2
# 1 sekunda vzorku s nejvysi korelaci
maskoff_tone_phase = maskoff_tone[max1:]
maskon_tone_phase = maskon_tone[max2:]
maskoff_tone = maskoff_tone[max1:max1 + fs]
maskon_tone = maskon_tone[max2:max2 + fs]
# ustrednit
maskoff_tone -= np.mean(maskoff_tone)
maskon_tone -= np.mean(maskon_tone)
# normalizace
maskoff_tone /= np.abs(maskoff_tone).max()
maskon_tone /= np.abs(maskon_tone).max()
# ramce
frame_length = int(fs * 0.02)
ramecoff = []
ramecon = []
for i in np.arange(fs, step=int(fs * 0.01)):
    if i + frame_length > fs:
        break
    end = (i + frame_length)
    ramecoff.append(maskoff_tone[i:end])
    ramecon.append(maskon_tone[i:end])
# plot frames
fig, ax = plt.subplots(2)
time = np.arange(ramecoff[0].size) / fs
ax[0].plot(time, ramecoff[10])
ax[0].set_ylabel("y")
ax[0].set_xlabel("čas")
ax[0].set_title("Rámec bez roušky")
ax[1].plot(time, ramecon[10])
ax[1].set_ylabel("y")
ax[1].set_xlabel("čas")
ax[1].set_title("Rámec s roušou")
plt.tight_layout()
plt.show()

# plot frame
time = np.arange(ramecoff[0].size) / fs
fig, ax = plt.subplots(4)
ax[0].plot(time, ramecoff[10])
ax[0].set_ylabel("y")
ax[0].set_xlabel("čas")
ax[0].set_title("Rámec")
ramecoff2 = ramecoff
ramecon2 = ramecon
ramecoff = np.array(ramecoff)
ramecon = np.array(ramecon)
# central cliping
for i in np.arange(len(ramecoff)):
    max_clip = np.abs(ramecoff[i]).max()
    for p in np.arange(len(ramecoff[0])):
        if ramecoff[i][p] > 0.7 * max_clip:
            ramecoff[i][p] = 1
        elif ramecoff[i][p] < -0.7 * max_clip:
            ramecoff[i][p] = -1
        else:
            ramecoff[i][p] = 0
for i in np.arange(len(ramecon)):
    max_clip = np.abs(ramecon[i]).max()
    for p in np.arange(len(ramecon[0])):
        if ramecon[i][p] > 0.7 * max_clip:
            ramecon[i][p] = 1
        elif ramecon[i][p] < -0.7 * max_clip:
            ramecon[i][p] = -1
        else:
            ramecon[i][p] = 0

ax[1].plot(time, ramecoff[10])
ax[1].set_ylabel("y")
ax[1].set_xlabel("čas")
ax[1].set_title("Centrální klipovaní 70%")

# autocorealtion
fundamental_freq = []
fundamental_freq2 = []
autocorel = []
autocorel2 = []
freq_threshold = 50
l_ram = len(ramecoff[0])
for i in range(len(ramecoff)):
    pomoc_ramec = np.concatenate([ramecoff[i], np.zeros(l_ram)])
    for p in range(l_ram):
        autocorel.append(sum(pomoc_ramec[:l_ram] * pomoc_ramec[p:l_ram + p]))

    pomoc_ramec2 = np.concatenate([ramecon[i], np.zeros(l_ram)])
    for p in range(l_ram):
        autocorel2.append(sum(pomoc_ramec2[:l_ram] * pomoc_ramec2[p:l_ram + p]))
    if i == 10:
        ax[2].plot(range(len(autocorel)), autocorel)
        ax[2].axvline(freq_threshold, color='black', label='prah')
        ax[2].scatter(np.argmax(autocorel[freq_threshold:]) + freq_threshold,
                      max(autocorel[freq_threshold:]), color='r', label='lag')
        ax[2].set_ylabel("y")
        ax[2].set_xlabel("vzorky")
        ax[2].set_title("Autokorelace")
        ax[2].legend(loc="upper right")

    fundamental_freq.append(16000 / (np.argmax(autocorel[freq_threshold:]) + freq_threshold))
    fundamental_freq2.append(16000 / (np.argmax(autocorel2[freq_threshold:]) + freq_threshold))
    autocorel = []
    autocorel2 = []

ax[3].plot(range(len(fundamental_freq)), fundamental_freq, label='bez rousky')
ax[3].plot(range(len(fundamental_freq2)), fundamental_freq2, label='s rouskou')
ax[3].legend(loc="upper right")
ax[3].set_ylabel("f0")
ax[3].set_xlabel("rámce")
ax[3].set_title("Základní frekvence rámců")
# vzorec rozptylu np.sum(np.abs(fundamental_freq-np.mean(fundamental_freq))**2)/len(fundamental_freq) -- vychazi stejne
fig.text(.5, .025, "(bez roušky) střední hodnota: " + str(np.mean(fundamental_freq).round(5)) + " rozptyl: " + str(
    (np.std(fundamental_freq) ** 2).round(5)), ha='center')
fig.text(.5, .005, "(s rouškou)  střední hodnota: " + str(np.mean(fundamental_freq2).round(5)) + " rozptyl: " + str(
    (np.std(fundamental_freq2) ** 2).round(5)), ha='center')
fig.tight_layout()
plt.show()

# DFT
plt.figure(figsize=(7, 3))
# spectr = dft(ramecoff2, 1024)
spectr = np.fft.fft(ramecoff2, 1024)
spectr_off = spectr
spectr = np.transpose(spectr[:, :512])
spectr = 10 * np.log10(np.abs(spectr) ** 2 + 1e-20)
plt.pcolormesh(np.linspace(0, 1, len(spectr[0])), np.linspace(0, fs / 2, len(spectr)), spectr.real)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.gca().set_title('Spectrogram bez roušky bez okénkové funkce')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 3))
# spectr = dft(ramecon2, 1024)
spectr = np.fft.fft(ramecon2, 1024)
spectr_on = spectr
spectr = np.transpose(spectr[:, :512])
spectr = 10 * np.log10(np.abs(spectr) ** 2 + 1e-20)
plt.pcolormesh(np.linspace(0, 1, len(spectr[0])), np.linspace(0, fs / 2, len(spectr)), spectr.real)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.gca().set_title('Spectrogram s rouškou bez okénkové funkce')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

# mask spectrum
fig, ax = plt.subplots()
mask_spectr = np.mean(np.abs(spectr_on / spectr_off), axis=0)
mask_spectr2 = 10 * np.log10(np.abs(mask_spectr) ** 2 + 1e-20)
ax.plot(np.linspace(0, fs / 2, len(mask_spectr2[:512])), mask_spectr2[:512])
ax.set_ylabel("Spektralní hustota výkonu [dB]")
ax.set_xlabel("Frekvence [Hz]")
ax.set_title("Frekvenční charakteristika roušky")
plt.tight_layout()
plt.show()

# freq response - mask
fig, ax = plt.subplots()
mask_response = np.fft.ifft(mask_spectr[:512], 1024)
# mask_response = idft(mask_spectr[:512], 1024)
ax.plot(range(len(mask_response.real)), mask_response.real)
ax.set_ylabel("H[n]")
ax.set_xlabel("n")
ax.set_title("Impulzní odezva roušky")
plt.tight_layout()
plt.show()

# apply filter
mask_response = mask_response[:512]  # echo
sentece_of, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_sentence.wav")
sentece_on, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskon_sentence.wav")
sentece_filter = signal.lfilter(mask_response.real, 1, sentece_of)
maskoff_tone2, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_tone.wav")
tone_filter = signal.lfilter(mask_response.real, 1, maskoff_tone2)
fig, ax = plt.subplots(3)
ax[0].set_title("Věty")
ax[0].plot(np.arange(sentece_of.size) / fs, sentece_of, label='bez roušky')
ax[0].plot(np.arange(sentece_filter.size) / fs, sentece_filter, label='s rouškovým filtrem')
ax[0].set_ylabel("y")
ax[0].set_xlabel("čas [s]")
ax[0].legend()
ax[1].set_title("Věta s rouškou")
ax[1].plot(np.arange(sentece_on.size) / fs, sentece_on)
ax[1].set_ylabel("y")
ax[1].set_xlabel("čas [s]")
ax[2].set_title("Tóny")
ax[2].plot(np.arange(maskoff_tone2.size) / fs, maskoff_tone2, label='bez roušky')
ax[2].plot(np.arange(tone_filter.size) / fs, tone_filter, label='s rouškovým filtrem')
ax[2].set_ylabel("y")
ax[2].set_xlabel("čas [s]")
ax[2].legend()
plt.tight_layout()
plt.show()
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_sentence.wav", sentece_filter, fs)
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_tone.wav", tone_filter, fs)

###  11 - window
fig, ax = plt.subplots(2)
window = np.blackman(len(ramecon2[0]))
ax[0].plot(window)
ax[0].set_title("Blackman")
ax[0].set_ylabel("amplituda")
ax[0].set_xlabel("vzorky")
window = np.fft.fft(window, 1024)[:512]
window = 10 * np.log10(np.abs(window) ** 2 + 1e-20)
ax[1].plot(np.linspace(0, fs / 2, len(window)), window.real)
ax[1].set_title("Frekvenční odezva Blackman")
ax[1].set_ylabel("Spektralní hustota výkonu [dB]")
ax[1].set_xlabel("Frekvence [Hz]")
plt.tight_layout()
plt.show()

# DFT
plt.figure(figsize=(7, 3))
ramecoff2_window = np.blackman(len(ramecoff2[0])) * ramecoff2
spectr_window = np.fft.fft(ramecoff2_window, 1024)
spectr_off_window = spectr_window
spectr_window = np.transpose(spectr_window[:, :512])
spectr_window = 10 * np.log10(np.abs(spectr_window) ** 2 + 1e-20)
plt.pcolormesh(np.linspace(0, 1, len(spectr_window[0])), np.linspace(0, fs / 2, len(spectr_window)), spectr_window.real)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.gca().set_title('Spectrogram bez roušky s okénkovou funkcí')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 3))
ramecon2_window = np.blackman(len(ramecon2[0])) * ramecon2
spectr_window = np.fft.fft(ramecon2_window, 1024)
spectr_on_window = spectr_window
spectr_window = np.transpose(spectr_window[:, :512])
spectr_window = 10 * np.log10(np.abs(spectr_window) ** 2 + 1e-20)
plt.pcolormesh(np.linspace(0, 1, len(spectr_window[0])), np.linspace(0, fs / 2, len(spectr_window)), spectr_window.real)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.gca().set_title('Spectrogram s rouškou s okénkovou funkcí')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

# mask spectrum
mask_spectr_window = np.mean(np.abs(spectr_on_window / spectr_off_window), axis=0)

# freq response - mask
mask_response_window = np.fft.ifft(mask_spectr_window[:512], 1024)

# apply filter
mask_response_window = mask_response_window[:512]  # echo
sentece_of, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_sentence.wav")
sentece_on, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskon_sentence.wav")
sentece_filter = signal.lfilter(mask_response_window.real, 1, sentece_of)
maskoff_tone2, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_tone.wav")
tone_filter = signal.lfilter(mask_response_window.real, 1, maskoff_tone2)
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_sentence_window.wav", sentece_filter, fs)
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_tone_window.wav", tone_filter, fs)

###  13 - same
ramecoff_same = []
ramecon_same = []

for i in range(len(ramecoff2)):
    if fundamental_freq[i] == fundamental_freq2[i]:
        ramecoff_same.append(ramecoff2[i][:])
        ramecon_same.append(ramecon2[i][:])

# DFT
spectr_same = np.fft.fft(ramecoff2, 1024)
spectr_off_same = spectr_same

spectr_same = np.fft.fft(ramecon2, 1024)
spectr_on_same = spectr_same

# mask spectrum
fig, ax = plt.subplots()
mask_spectr_same = np.mean(np.abs(spectr_on_same / spectr_off_same), axis=0)
mask_spectr_same = 10 * np.log10(mask_spectr_same ** 2 + 1e-20)
ax.plot(np.linspace(0, fs / 2, len(mask_spectr_same[:512])), mask_spectr_same[:512])
ax.set_ylabel("Spektralní hustota výkonu [dB]")
ax.set_xlabel("Frekvence [Hz]")
ax.set_title("Frekvenční charakteristika roušky only-match")
plt.tight_layout()
plt.show()

# freq response - mask
mask_response_same = np.fft.ifft(mask_spectr_same[:512], 1024)

# apply filter
mask_response_same = mask_response_same[:512]  # echo
sentece_of, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_sentence.wav")
sentece_on, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskon_sentence.wav")
sentece_filter = signal.lfilter(mask_response_same.real, 1, sentece_of)
maskoff_tone2, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_tone.wav")
tone_filter = signal.lfilter(mask_response_same.real, 1, maskoff_tone2)
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_sentence_only_match.wav", sentece_filter, fs)
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_tone_only_match.wav", tone_filter, fs)
##  10 - overlap add

M = len(mask_response)
N = 8 * M
step_size = N - M
H = np.fft.fft(mask_response, N)
position = 0
sentece_filter = np.zeros(len(sentece_of) + M)
while position + step_size <= len(sentece_of):
    sentece_filter[position:position + N] += (
        np.fft.ifft(np.fft.fft(sentece_of[position:step_size + position], N) * H)).real
    position += step_size
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_sentence_overlap_add.wav", sentece_filter, fs)
position = 0
sentece_filter = np.zeros(len(sentece_of) + M)
while position + step_size <= len(sentece_of):
    sentece_filter[position:position + N] += (
        np.fft.ifft(np.fft.fft(maskoff_tone2[position:step_size + position], N) * H)).real
    position += step_size
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_tone_overlap_add.wav", sentece_filter, fs)
# ze studijnich zdroju filterbank.mat
A = np.array([
    [1, -5.99314978194146, 14.9663698425288, -19.9339787828265, 14.9352152366527, -5.96822458496144, 0.993768070555755],
    [1, -5.99117344135964, 14.9568537416994, -19.9156748933291, 14.9176369558589, -5.95979699084515, 0.992154628007440],
    [1, -5.98856858322409, 14.9444107246558, -19.8919454340968, 14.8950585745789, -5.94908082648596, 0.990125544699216],
    [1, -5.98510460261366, 14.9280154461799, -19.8609917071884, 14.8659304582186, -5.93542455706556, 0.987574962976399],
    [1, -5.98045167579716, 14.9062225761943, -19.8203234636664, 14.8281567069564, -5.91797491747640, 0.984370775814811],
    [1, -5.97413212852788, 14.8769696332719, -19.7664520987354, 14.7788724420020, -5.89560637431399, 0.980348534387371],
    [1, -5.96544599640858, 14.8372793418806, -19.6944345728732, 14.7141196108520, -5.86682253956173, 0.975304188354799],
    [1, -5.95335661501475, 14.7828094165514, -19.5971943497827, 14.6283735525777, -5.82961746336100, 0.968985587529103],
    [1, -5.93631434312268, 14.7071725030427, -19.4645130978740, 14.5138523836273, -5.78127971335492, 0.961082779217451],
    [1, -5.91198478930943, 14.6009150861992, -19.2815467967546, 14.3595168494284, -5.71811564631919, 0.951217330080896],
    [1, -5.87683046413374, 14.4500039547408, -19.0266816314074, 14.1496457009456, -5.63506072712623, 0.938931233161793],
    [1, -5.82546969036031, 14.2336359582362, -18.6685401841630, 13.8618704031171, -5.52514107206799, 0.923676496139393],
    [1, -5.74970264930923, 13.9212050380389, -18.1620617392564, 13.4646247990916, -5.37874701303822, 0.904807329832556],
    [1, -5.63705390875799, 13.4684474638276, -17.4440131274997, 12.9142339871712, -5.18269945149946, 0.881578059262132],
    [1, -5.46864697969933, 12.8134275064186, -16.4294612362706, 12.1526040044079, -4.91915671825194, 0.853151537098770],
    [1, -5.21624236295590, 11.8747533276931, -15.0133902931702, 11.1082181352470, -4.56458265800619, 0.818624942306534],
    [1, -4.83845216591003, 10.5584527058472, -13.0865297327598, 9.70678157825266, -4.08938352345573, 0.777082179284583],
    [1, -4.27677416348210, 8.78809020141143, -10.5802617899477, 7.90421365675864, -3.45959578352370, 0.727684018908161],
    [1, -3.45379284096798, 6.58541086409681, -7.55137007483893, 5.76205898548693, -2.64337104933381, 0.669807276754337],
    [1, -2.27989674029712, 4.23575256060841, -4.26736262444801, 3.58081800906784, -1.62693084924136, 0.603240261306894],
    [1, -0.682980909101583, 2.52370564862010, -1.10301215983892, 2.04585573848484, -0.445996329380557,
     0.528429750073459],
    [1, 1.31158141936795, 2.78286198422165, 2.05149313197317, 2.13647831640717, 0.765132168116848, 0.446750365139959],
    [1, 3.41567866392510, 5.94086921145382, 6.23668938235665, 4.22909780993418, 1.72524148624049, 0.360726733850011],
])

B = np.array([
    [3.80987153612954e-09, 0, -1.14296146083886e-08, 0, 1.14296146083886e-08, 0, -3.80987153612954e-09],
    [7.59789474204466e-09, 0, -2.27936842261340e-08, 0, 2.27936842261340e-08, 0, -7.59789474204466e-09],
    [1.51938845856758e-08, 0, -4.55816537570273e-08, 0, 4.55816537570273e-08, 0, -1.51938845856758e-08],
    [3.03496585141781e-08, 0, -9.10489755425343e-08, 0, 9.10489755425343e-08, 0, -3.03496585141781e-08],
    [6.05987970997354e-08, 0, -1.81796391299206e-07, 0, 1.81796391299206e-07, 0, -6.05987970997354e-08],
    [1.20951723298649e-07, 0, -3.62855169895947e-07, 0, 3.62855169895947e-07, 0, -1.20951723298649e-07],
    [2.41281620804269e-07, 0, -7.23844862412808e-07, 0, 7.23844862412808e-07, 0, -2.41281620804269e-07],
    [4.81009044445994e-07, 0, -1.44302713333798e-06, 0, 1.44302713333798e-06, 0, -4.81009044445994e-07],
    [9.58121316390535e-07, 0, -2.87436394917161e-06, 0, 2.87436394917161e-06, 0, -9.58121316390535e-07],
    [1.90649255758428e-06, 0, -5.71947767275285e-06, 0, 5.71947767275285e-06, 0, -1.90649255758428e-06],
    [3.78862849216772e-06, 0, -1.13658854765032e-05, 0, 1.13658854765032e-05, 0, -3.78862849216772e-06],
    [7.51654707701598e-06, 0, -2.25496412310480e-05, 0, 2.25496412310480e-05, 0, -7.51654707701598e-06],
    [1.48821952944896e-05, 0, -4.46465858834688e-05, 0, 4.46465858834688e-05, 0, -1.48821952944896e-05],
    [2.93906303746625e-05, 0, -8.81718911239875e-05, 0, 8.81718911239875e-05, 0, -2.93906303746625e-05],
    [5.78595381657480e-05, 0, -0.000173578614497244, 0, 0.000173578614497244, 0, -5.78595381657480e-05],
    [0.000113458409195232, 0, -0.000340375227585695, 0, 0.000340375227585695, 0, -0.000113458409195232],
    [0.000221409730259076, 0, -0.000664229190777228, 0, 0.000664229190777228, 0, -0.000221409730259076],
    [0.000429516610778815, 0, -0.00128854983233645, 0, 0.00128854983233645, 0, -0.000429516610778815],
    [0.000827232469246805, 0, -0.00248169740774042, 0, 0.00248169740774042, 0, -0.000827232469246805],
    [0.00157941475011366, 0, -0.00473824425034097, 0, 0.00473824425034097, 0, -0.00157941475011366],
    [0.00298447535020483, 0, -0.00895342605061449, 0, 0.00895342605061449, 0, -0.00298447535020483],
    [0.00557158418772481, 0, -0.0167147525631744, 0, 0.0167147525631744, 0, -0.00557158418772481],
    [0.0102582849159760, 0, -0.0307748547479281, 0, 0.0307748547479281, 0, -0.0102582849159760],
])
maskoff_second = maskoff_tone_phase[:fs]
maskon_second = maskon_tone_phase[:fs]

mask_spectr_energy = []
for i in range(len(A)):
    # print("stable" if np.all(np.abs(np.roots(A[i]))<1) else "not stable")
    off_filtered = np.sum(np.abs(signal.lfilter(B[i, :], A[i, :], maskoff_second)) ** 2)
    on_filtered = np.sum(np.abs(signal.lfilter(B[i, :], A[i, :], maskon_second)) ** 2)
    mask_spectr_energy.append(on_filtered / off_filtered)
mask_spectr_energy = np.array(mask_spectr_energy)
mask_spectr_energy = 10 * np.log10(mask_spectr_energy ** 2 + 1e-20)
fig, ax = plt.subplots()
ax.plot(np.linspace(0, fs / 2, len(mask_spectr_energy)), mask_spectr_energy)
ax.set_ylabel("Spektralní hustota výkonu [dB]")
ax.set_xlabel("Frekvence [Hz]")
ax.set_title("Frekvenční charakteristika roušky porovnáním energie")
plt.tight_layout()
plt.show()

# double lag
tone_double_lag = sentece_of[20000:20000 + fs]
frame_pick = 35
# ustrednit
tone_double_lag -= np.mean(tone_double_lag)
# normalizace
tone_double_lag /= np.abs(tone_double_lag).max()
# ramce
frame_length = int(fs * 0.02)
ramecoff = []
ramecon = []
for i in np.arange(fs, step=int(fs * 0.01)):
    if i + frame_length > fs:
        break
    end = (i + frame_length)
    ramecoff.append(tone_double_lag[i:end])

time = np.arange(ramecoff[0].size) / fs

ramecoff = np.array(ramecoff)
ramecon = np.array(ramecon)
# central cliping
for i in np.arange(len(ramecoff)):
    max_clip = np.abs(ramecoff[i]).max()
    for p in np.arange(len(ramecoff[0])):
        if ramecoff[i][p] > 0.6 * max_clip:
            ramecoff[i][p] = 1
        elif ramecoff[i][p] < -0.6 * max_clip:
            ramecoff[i][p] = -1
        else:
            ramecoff[i][p] = 0
# autocorealtion
fundamental_freq = []
fig, ax = plt.subplots()
autocorel = []
lags = []
freq_threshold = 50
fundamental_freq2 = []
l_ram = len(ramecoff[0])
for i in range(len(ramecoff)):
    pomoc_ramec = np.concatenate([ramecoff[i], np.zeros(l_ram)])
    for p in range(l_ram):
        autocorel.append(sum(pomoc_ramec[:l_ram] * pomoc_ramec[p:l_ram + p]))
    if i == frame_pick:
        ax.plot(range(len(autocorel)), autocorel)
        ax.axvline(freq_threshold, color='black', label='prah')
        ax.scatter(np.argmax(autocorel[freq_threshold:]) + freq_threshold,
                   max(autocorel[freq_threshold:]), color='r', label='lag')
        ax.set_ylabel("y")
        ax.set_xlabel("vzorky")
        ax.set_title("Autokorelace")
    fundamental_freq.append(16000 / (np.argmax(autocorel[freq_threshold:]) + freq_threshold))
    lags.append((np.argmax(autocorel[freq_threshold:]) + freq_threshold))
    autocorel = []
# double lag fix
med = np.median(lags[:])
for i in range(len(lags)):
    if np.abs(lags[i] - med) > med / 4:
        lags[i] = med
    fundamental_freq2.append(16000 / lags[i])
    if i == frame_pick:
        ax.axvline(lags[i], color='r', label='oprava lagu')
        ax.legend(loc="upper right")
fig.tight_layout()
plt.show()
# phase shift

cherypick = 3000
maskoff_tone = maskoff_tone2[cherypick:cherypick + fs]
maskon_tone2, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskon_tone.wav")
maskon_tone = maskon_tone2[cherypick:cherypick + fs]
# ramce
frame_length = int(fs * 0.025)
ramecoff = []
ramecon = []
i = 0
while i + frame_length <= len(maskoff_tone):
    end = (i + frame_length)
    ramecoff.append(maskoff_tone[i:end])
    ramecon.append(maskon_tone[i:end])
    i += int(fs * 0.01)
# plot frame
time = np.arange(ramecoff[0].size) / fs
fig, ax = plt.subplots(3)
ax[0].plot(time, ramecoff[10], label='bez roušky')
ax[0].plot(time, ramecon[10], label='s rouškou')
ax[0].set_ylabel("y")
ax[0].set_xlabel("čas")
ax[0].legend()
ax[0].set_title("Rámce před zarovnáním")
ramecoff = np.array(ramecoff)
ramecon = np.array(ramecon)
ramecoff_clip = np.array(ramecoff, copy=True)
ramecon_clip = np.array(ramecon, copy=True)
# central cliping
for i in np.arange(len(ramecoff_clip)):
    max_clip = np.abs(ramecoff_clip[i]).max()
    for p in np.arange(len(ramecoff_clip[0])):
        if ramecoff_clip[i][p] > 0.7 * max_clip:
            ramecoff_clip[i][p] = 1
        elif ramecoff_clip[i][p] < -0.7 * max_clip:
            ramecoff_clip[i][p] = -1
        else:
            ramecoff_clip[i][p] = 0
for i in np.arange(len(ramecon_clip)):
    max_clip = np.abs(ramecon_clip[i]).max()
    for p in np.arange(len(ramecon_clip[0])):
        if ramecon_clip[i][p] > 0.7 * max_clip:
            ramecon_clip[i][p] = 1
        elif ramecon_clip[i][p] < -0.7 * max_clip:
            ramecon_clip[i][p] = -1
        else:
            ramecon_clip[i][p] = 0

# zarovnani
corel_array = []
ramecoff = np.array(ramecoff)
ramecon = np.array(ramecon)
ramecoff2 = np.empty((0, 320))
ramecon2 = np.empty((0, 320))
for i in np.arange(len(ramecoff)):
    corel1 = signal.correlate(ramecoff_clip[i], ramecon_clip[i])
    maxCor = np.argmax(corel1)
    corel2 = signal.correlate(ramecon_clip[i], ramecoff_clip[i])
    if maxCor < 320:
        maxCor = 320
    if np.argmax(corel2) < maxCor:
        maxCor = np.argmax(corel2)
        if maxCor < 320:
            maxCor = 320
        corel_array.append(400 - maxCor)
        h1 = np.array((ramecoff[i, :maxCor])[:320])
        ramecoff2 = np.append(ramecoff2, h1.reshape((1, 320)), axis=0)
        h2 = np.array((ramecon[i, 400 - maxCor:])[:320])
        ramecon2 = np.append(ramecon2, h2.reshape((1, 320)), axis=0)
    else:
        corel_array.append(-(400 - maxCor))
        h1 = np.array((ramecoff[i, 400 - maxCor:]))[:320]
        ramecoff2 = np.append(ramecoff2, h1.reshape((1, 320)), axis=0)
        h2 = np.array((ramecon[i, :maxCor])[:320])
        ramecon2 = np.append(ramecon2, h2.reshape((1, 320)), axis=0)
ramecon = ramecon2
ramecoff = ramecoff2

ax[1].plot(time[:320], ramecoff[10], label='bez roušky')
ax[1].plot(time[:320], ramecon[10], label='s rouškou')
ax[1].set_ylabel("y")
ax[1].set_xlabel("čas [s]")
ax[1].legend()
ax[1].set_title("Rámce po zarovnání")

ax[2].set_title("Fázový posun rámců")
ax[2].plot(np.linspace(0, 2, len(corel_array)), corel_array)
ax[2].set_ylabel("fázový posun [rámce]")
ax[2].set_xlabel("čas [s]")

plt.tight_layout()
plt.show()

ramecoff = np.array(ramecoff)
ramecon = np.array(ramecon)

# DFT
spectr_off = np.fft.fft(ramecoff, 1024)
spectr_on = np.fft.fft(ramecon, 1024)

# mask spectrum
mask_spectr_phase = np.mean(np.abs(spectr_on / spectr_off), axis=0)

# freq response - mask
mask_response_phase = np.fft.ifft(mask_spectr_phase[:512], 1024)

# apply filter
mask_response_phase = mask_response_phase[:512]  # echo
sentece_of, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_sentence.wav")
sentece_on, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskon_sentence.wav")
sentece_filter = signal.lfilter(mask_response_phase.real, 1, sentece_of)
maskoff_tone2, _ = sf.read(os.path.dirname(os.getcwd())+"\\audio\\maskoff_tone.wav")
tone_filter = signal.lfilter(mask_response_phase.real, 1, maskoff_tone2)
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_sentence_phase.wav", sentece_filter, fs)
sf.write(os.path.dirname(os.getcwd())+"\\audio\\sim_maskon_tone_phase.wav", tone_filter, fs)
