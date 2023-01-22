from pyo import *

print (pa_get_input_devices())
print (pa_get_output_devices())


ps = Server(audio="pa", sr=44100,nchnls=1,ichnls=1,duplex=1)
ps.setInputDevice(3)
ps.setOutputDevice(3)
s=ps.boot()

mic = Input().play().out()
mic.ctrl()

"""
env = WinTable(8)
wsize = .1
trans = -7

ratio = pow(2., trans/12.)
rate = -(ratio-1) / wsize

ind = Phasor(freq=rate, phase=[0,0.5])
win = Pointer(table=env, index=ind, mul=.7)
snd = Delay(mic, delay=ind*wsize, mul=win).mix(1).out(1)
"""
Spectrum(mic)
s.gui(locals())
