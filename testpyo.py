#!/bin/bash

import pyo
import time

indevs,indexes = pyo.pa_get_input_devices()
for i in range(len(indevs)):
    print(f" {indexes[i]:02d}: {indevs[i]}")

indevdex = input("enter input device:")
indevdex = int(indevdex)
print(f"input selected: {indevdex}")

outdevs,outdexes = pyo.pa_get_output_devices()
for i in range(len(outdevs)):
    print(f" {outdexes[i]:02d}: {outdevs[i]}")
outdevdex = input("enter output device:")
outdevdex = int(outdevdex)
print(f"output selected: {outdevdex}")



s = pyo.Server(audio="portaudio") 


# audio="pa",nchnls=2,ichnls=1)


t = s.boot()

t.setInputDevice(indevdex)
t.setOutputDevice(outdevdex)

i = pyo.Input(indevdex)
j= i.out()


t.start()

y = i.ctrl()
z=j.ctrl()

s.gui(locals())

