#!/usr/bin/env python3

# Author: aqeelanwar
# Created: 13 March,2020, 9:19 PM
# Email: aqeel.anwar@gatech.edu

from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
graph_colors = [ col['color'] for col in mpl.rcParams['axes.prop_cycle'] ]

game_instance = None
gui_done=False


import io,sys
import sounddevice as sd
import argparse
import logging
import numpy as np
from scipy.io import wavfile
import queue
import matplotlib.pyplot as plt
import threading
from time import time as utime, strftime
import time
bufferslock = threading.Lock()
processq = queue.Queue()
cmdq = queue.Queue()
# switchmodes
SW_NULL='START' # JUST started nothing is happening
SW_METRO='Ticks' # play metronome
SW_PLAY='play' # PLAYING existing loops
SW_PLAY2REC='play(rec)' # hit rec, waiting for top of next loop
SW_REC='REC!' #recording
SW_REC2PLAY='rec(play)'
SW_PAUSE='[pause]' #paused
switch=SW_NULL
idev=None
odev=None

blip = wavfile.read("sounds/Woodblock.wav")
blip=blip[1][::2]
blip=np.float32(blip/5768.0/20.)*5
print(f"blip len={blip.shape} min/max={blip.min()}/{blip.max()}")
bli2 = wavfile.read("sounds/Woodblock1.wav")
bli2=bli2[1][::2]
bli2=np.float32(bli2/40000.0)*5
print(f"bli2 len={bli2.shape} min/max={bli2.min()}/{bli2.max()}")

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

print(sd.query_devices())

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-d', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-i', '--input-device', type=int_or_str,
                    help='input device ID or substring', default="none")
parser.add_argument('-o', '--output-device', type=int_or_str,
                    help='output device ID or substring')
parser.add_argument('-c', '--channels', type=int, default=1,
                    help='number of channels')
parser.add_argument('-t', '--dtype', help='audio data type',default="float32")
parser.add_argument('-s', '--samplerate', type=int, help='sampling rate',default=44100)
parser.add_argument('-b', '--blocksize', type=int, help='block size',default=512)
parser.add_argument('-m', '--tempo', type=int, help='tempo in bpm',default=85)
parser.add_argument('-a', '--beats', type=int, help='beats per measure',default=4)
parser.add_argument('-u', '--measures', type=int, help='beats per measure',default=2)
parser.add_argument('-n', '--nblocks', type=int, help='number of blocks in buffer',default=256)
parser.add_argument('-l', '--latency', type=int, help='latency in blocks',default=0)
parser.add_argument('-N', '--numbuff', type=int, help='number of buffers',default=9)
parser.add_argument('-D', '--graphdecimate', type=int, help='stride to apply to graphs',default=2)
args = parser.parse_args()
blocksize = int(args.blocksize)
metro_interval = (int(args.samplerate*60/args.tempo) // int(args.blocksize)) * int(args.blocksize)
metro_pickup=((int(args.samplerate)//8) //int(args.blocksize)) * int(args.blocksize)

reclen = metro_interval * int(args.beats) * int(args.measures)

buffers =[] #np.zeros(args.blocksize * args.nblocks)

lasttick=0
attack = 0
noisefloor=0
kvolume = 0
prevblock=None
sync_samples2go = None
metro_elapsed = 0
recbuffer = None
mixbuffer = None
metbuffer = None
prerecordbuffer = None

MBP = 0

# callbacks are stripped down to very minimal form.
# algorithm for each call is very simple:
# output callback is called when a block of sound
# data needs to be supplied ASACP (as soon as 
# computationally (computerly?) possible).
# in this program, the mixbuffer will always be loaded up
# in advance with the audio to be played back at the current
# Master Buffer Pointer (MBP) location (i.e. written to the outdata aray),
# and the recbuffer will always be ready to receive the indata.

# The overall approach is to design the workflow of the
# system so that all audio data processing will occur during
# the playback and recording of the audio, and preventing
# buffer underruns.


def output_callback( MBP, outdata, frames, cbtime, status ):
    #print(f"{time.time()}:output_callback; outdata.shape={outdata.shape}, frames={frames}")
    """
    The output callback is now trivial; it just copies the
    contents of the mixbuffer block at the MBP location
    to the outdata buffer.
    """
    global mixbuffer
    assert args.blocksize == frames
    #print("oc",end='',flush=True)
    outdata[:] = mixbuffer[MBP:MBP+args.blocksize,...].reshape(outdata[:].shape)


def input_callback( MBP, indata, frames, cbtime, status ):
    #print(f"{time.time()}:input_callback")
    """
    The input callback is equally trivial; it just copies
    the just-recorded audio from the indata buffer to the
    recbuffer. Done.
    """
    global recbuffer
    assert args.blocksize == frames
    PrevBP = (MBP-args.blocksize) % reclen
    recbuffer[PrevBP:PrevBP+args.blocksize] = indata.reshape((args.blocksize,)).reshape(recbuffer[PrevBP:PrevBP+args.blocksize].shape)



def process_block(MBP):
    #print(f"{time.time()}:process_block")
    set_button()
    # send dummy value into queue to kick off the data processing.
    processq.put_nowait(0)
    # the callback now is done and the
    # actual data processing takes place in another thread.
    #print(f"{time.time()}:process_block return")
    return

def callback(indata,outdata,frames,cbtime,status):
    #print(f"{time.time()}:callback")
    #outdata = np.zeros_like(outdata)
    #crap=indata
    #return
    """
    The callback vastly simplified: quickly and trivially
    supply the requested playback audio from 'mixbuffer' and copy the
    just-recorded incoming audio to 'recbuffer'.
    Then, once those two operations are completed and the audio for the next block
    is being played/recorded, then undertake the audio processing
    necessary to prepare for the next callback.
    """
    # supply sound device with output to be played from 'mixbuffer'
    output_callback(MBP,outdata,frames,cbtime,status)
    # copy incoming recorded data into 'recbuffer'
    input_callback(MBP, indata,frames,cbtime,status)
    # now kick off audio processing
    process_block(MBP)




def process_thread():
    #print(f"{time.time()}:process_thread")
    global buffers, switch, attack,noisefloor, kvolume,sync_samples2go, metbuffer,metbuffer,recbuffer,mixbuffer, prerecordbuffer, metro_elapsed, metro_interval,MBP,outputq, inputq, graph_colors, processq
    while not gui_done:
        try:
            dummy = processq.get_nowait()
        except queue.Empty as e:
            sd.sleep(1)
            continue

        #== process data in preparation for next play/record block 
        #print("ic",end='',flush=True)
        NextBP = (MBP+args.blocksize) % reclen
        #if NextBP==0:
        #    print('\n****\n',end='',flush=True)
        #==== input code ====
        # recbuffer[MBP:MBP+args.blocksize] contains just-recorded data.
        # put the new audio into 'indata'
        indata = recbuffer[MBP:MBP+args.blocksize]
        # calculate volume of indata
        volume = np.abs(indata).sum()/len(indata)
        #if volume > kvolume: kvolume = volume
        kvolume = kvolume*0.93 + volume * 0.07

        # act based on current switch mode
        if switch==SW_NULL:
            # collecting noise floor data only; no prep needed
            noisefloor = noisefloor * 0.99 + (np.abs(indata).sum()/len(indata))*0.01
        elif switch==SW_METRO:
            # we're playing back the mixbuffer which contains the metronome ticks
            if NextBP ==0 :
                # we're done playing the metronome track!
                # we must setup recording
                #print('\n==done metro==\n',flush=True)
                with bufferslock:
                    # init the first mixbuffer with the metronome track
                    mixbuffer = metbuffer * 1
                    buffers = [ ]
                    # have a fresh blank recbuffer waiting
                    recbuffer = recbuffer * 0
                    # switch to recording-a-track mode
                    switch = SW_REC
                    set_button()
        elif switch==SW_REC:
            # recorded audio already in recbuffer,
            # may need to adjust in place
            #fact = 1.0
            #if kvolume < noisefloor*1.5:
            #    fact = 0 #((kvolume/1.5)**1.5)*1.5
            with bufferslock:
                #recbuffer[MBP:MBP+args.blocksize] *= fact
                if NextBP == 0:
                    # we just filled up the current track
                    if len(buffers)==args.numbuff:
                        # if we have reached arg.numbuff buffers, we need to mix the buffers[2] into buffers[1]
                        buffers[1] = np.array(buffers[1]) + np.array(buffers[2])
                        # now move all buffers down
                        buffers = buffers[0:2]+buffers[3:args.numbuff]
                        graph_colors = graph_colors[0:2] + graph_colors[3:] + [graph_colors[2]]
                    # append newly recorded recbuffer into the array of buffer history
                    buffers = buffers + [recbuffer*0.96]
                    mixbuffer=np.sum(buffers[:],axis=0)
                    recbuffer = recbuffer * 0

        elif switch==SW_PLAY:
            # don't need to do anything, it's already been done.
            pass
        elif switch==SW_PAUSE:
            # don't play anything, don't record anything, and don't increment the MBP
            return
        MBP = NextBP
        NextBP = (MBP+args.blocksize) % reclen

def audio_handler():
    #print("starting audio_handler...")
    global switch,buffers,mixbuffer,recbuffer
    if True:
        # we are just starting, set up everythig
        recbuffer = np.zeros((reclen,args.channels),dtype=np.float32)
        mixbuffer = np.zeros_like(recbuffer)
        MBP=0
        buffers=[]
        switch = SW_NULL
        print({'device':(idev,odev),
            'samplerate':args.samplerate, 'blocksize':args.blocksize,
            'dtype':args.dtype, 'latency':0*args.latency,
            'channels':args.channels, 'callback':callback})
        with sd.Stream(device=(idev,odev),
                       samplerate=args.samplerate, blocksize=args.blocksize,
                       dtype=args.dtype, latency=0*args.latency,
                    channels=args.channels, callback=callback) as strm:
            print ("stream open!")
            while not gui_done:
                sd.sleep(1)

        print("bombed out!!!")
        exit_button()
    return

def set_button():
    cmdq.put_nowait(("L",switch))

def main_button():
    global switch,buffers,metbuffer,metbuffer,recbuffer,mixbuffer,attack, game_instance,\
            sync_samples2go, metro_elapsed, metro_interval,MBP
    #print("--main_button--")
    #print(f"metro_interval={metro_interval}")
    if switch == SW_NULL:
        switch = SW_METRO
        set_button()
        MBP = 0
        # prepare the metronome ticks
        recbuffer = np.zeros((reclen,))
        mixbuffer = np.zeros((reclen,))
        metbuffer = np.zeros((reclen,))
        for i in range(args.beats*args.measures-1):
            if i % args.beats == 0:
                mixbuffer[i*metro_interval+metro_pickup:i*metro_interval+len(bli2)+metro_pickup] = bli2[:,0]
            else:
                mixbuffer[i*metro_interval+metro_pickup:i*metro_interval+len(blip)+metro_pickup] = blip[:,0]

    elif switch == SW_PLAY:
        switch = SW_PLAY2REC
        set_button()
    elif switch == SW_PLAY2REC:
        switch = SW_PLAY
    elif switch==SW_REC:
        # rec2play mode, stops recording but plays through
        # to end of loop and then does exactly what REC does
        # (which is to (potentially) mix 2->1 and bump down the
        # recordings, append the recbuffer, and prepare a new
        # mixbuffer, and then goes into play mode.
        switch = SW_REC2PLAY
    elif switch == SW_REC2PLAY:
        # ok now we will resume recording until end of loop.
        # the effect is that hitting the main button will toggle
        # between SW_REC and SW_REC2PLAY until the current loop
        # finishes.  If switch is SW_REC2PLAY then recording
        # ends.  If it is SW_REC then it just keeps on recording.
        switch = SW_REC
    set_button()

def save_button(junk=0):
    if len(buffers)==0: return
    tstamp=strftime("%Y%m%d_%H%M%S")
    for i in range(len(buffers)):
        finame=f"loopy_{tstamp}-{i:03d}.wav"
        wavfile.write(finame, args.samplerate, buffers[i])
    print(f"loopy_{tstamp}-000 to {len(buffers)-1:03d}.wav saved.")
    pass

def exit_button(junk=0):
    global gui_done
    gui_done = True
    cmdq.put_nowait(("X","destroy"))
    pass

def undo_button(junk=0):
    global switch,buffers,mixbuffer,recbuffer
    with bufferslock:
        if len(buffers) > 1:
            buffers.pop()
        if len(buffers) == 1:
            mixbuffer=np.zeros(reclen)
            recbuffer=np.zeros(reclen)
            buffers=[]
            switch=SW_METRO
        else:
            mixbuffer=np.sum(buffers,axis=0)
        if switch==SW_METRO:
            switch=SW_NULL
            buffers=[]
            mixbuffer=None
            recbuffer=None
    set_button()
    game_instance.window.update()

class Visualizations():
    # ------------------------------------------------------------------
    # Initialization functions
    # ------------------------------------------------------------------
    def __init__(self):
        self.window = Tk()
        self.window.title('Da Looper')
        self.window.geometry("+400+0")
        self.window.config(bg='black')
        #self.canvas = Canvas(self.window, width=size_of_board, height=size_of_board)
        #self.canvas.pack()
        #self.label = Label(self.window,height=40,width=120,font=('Courier',8))
        #self.label.pack()

        self.fig = plt.Figure(figsize=(12,8),facecolor='black')
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.axis('off')
        self.ax1.set_facecolor('black')
        self.lines=[]
        for i in range(20):
            line, = self.ax1.plot([],[], lw=.8)
            self.ax1.set_facecolor('black')
            self.lines.append(line)

        self.canvas = FigureCanvasTkAgg(self.fig,master=self.window)
        self.ax1.set_facecolor('black')
        self.canvas.draw()
        self.canvas.get_tk_widget().config(bg='#000000')
        self.canvas.get_tk_widget().pack()
        self.canvas.get_tk_widget().config(bg='#000000')

        #self.ax1.set_ylim(0,255)
        #self.ax1.set_xlim(0,500)

        self.button1 = Button(self.window, text="START LOOP", command=main_button) #, default=tk.DISABLED,state=tk.DISABLED)
        self.button1.pack(side=RIGHT,padx=5, pady=5)
        self.button2 = Button(self.window, text="Exit", command=exit_button)
        self.button2.pack(side=RIGHT,padx=5, pady=5)
        self.button3 = Button(self.window, text="Save", command=save_button)
        self.button3.pack(side=RIGHT,padx=5, pady=5)
        self.button4 = Button(self.window, text="Undo", command=undo_button)
        self.button4.pack(side=RIGHT,padx=5, pady=5)
        self.button1.focus_set()
        self.window.bind("q",exit_button)
        self.window.bind("<Escape>",undo_button)
        self.window.bind("Q",exit_button)
        self.window.bind("s",save_button)
        self.interval = 1000//60
        self.start()


    def start(self):
        mybuffers=None
        with bufferslock:
            if len(buffers) > 0:
                if switch==SW_REC:
                    stuff = [metbuffer] + buffers + [recbuffer]
                else:
                    stuff = buffers
                mybuffers = np.stack(stuff,axis=1)[::args.blocksize,...]
        #~ self.arduinoData = serial.Serial('com5', 115200)
        #~ self.arduinoData.flushInput()
        if mybuffers is not None:
            self.ani = animation.FuncAnimation(
                self.fig,
                self.update_graph,
                interval=self.interval,
                repeat=False)
        else: # FOR AN ENDLESS LOOP:
            self.ani = animation.FuncAnimation(
                self.fig,
                self.update_graph,
                interval=self.interval,
                repeat=False)
        self.running = True
        self.ani._start()
        self.start_time = time.time()
        print('started animation')

    def update_graph(self, i):
        if gui_done: self.ani.event_source.stop()
        mybuffers=None
        with bufferslock:
            if len(buffers) > 0:
                EMBP = (MBP + args.blocksize*2) % len(buffers[0])
                dataindex = np.int32(range(0,len(buffers[0]),args.graphdecimate)) #blocksize//256))
                dataindex = np.concatenate((dataindex[dataindex >= EMBP],dataindex[dataindex < EMBP]))
                if switch == SW_NULL:
                    stuff = None
                    mybuffers = None
                elif switch == SW_METRO:
                    stuff = buffers
                    mybuffers = np.stack(stuff,axis=1)[dataindex,...]
                    mybuffers = stuff[dataindex,...]
                elif switch == SW_REC or switch== SW_REC2PLAY:
                    stuff = buffers + [recbuffer]
                    mybuffers = np.stack(stuff,axis=1)[dataindex,...]
                elif switch == SW_PLAY or switch == SW_PLAY2REC:
                    stuff = buffers
                    mybuffers = np.stack(stuff,axis=1)[dataindex,...]
        if mybuffers is not None:
            for i in range(mybuffers.shape[1]):
                data_add = 0
                if switch == SW_REC and MBP >= reclen-metro_interval*args.beats and len(buffers) == 9 and i > 1:
                    if i > 0:
                        data_add = -(MBP - reclen-metro_interval*args.beats)/metro_interval/args.beats * 2 - 4
                if switch == SW_REC and i == mybuffers.shape[1] - 1:
                    self.lines[i].set_data(np.arange(len(mybuffers[:,i]))[(reclen-MBP)//args.graphdecimate:],mybuffers[:,i][(reclen-MBP)//args.graphdecimate:]+i*2+data_add)
                    self.lines[i].set_color(graph_colors[i])
                    self.lines[i].set_linewidth(3)
                else:
                    self.lines[i].set_data(np.arange(len(mybuffers[:,i])),mybuffers[:,i]+i*2+data_add)
                    self.lines[i].set_color(graph_colors[i])
                    self.lines[i].set_linewidth(1)
            for i in range(mybuffers.shape[1],len(self.lines)):
                self.lines[i].set_data([],[])

        else:
            self.lines[0].set_data([],[])
        self.ax1.set_ylim(-1,20)
        if mybuffers is not None:
            self.ax1.set_xlim(0,len(mybuffers))
        """
        if mybuffers is not None:  #self.points and i >= self.points - 1:
            # animation finished; clean up
            self.running = False
            self.ani = None
            ms = int(((time.time() - self.start_time) * 1000) / (int(self.points_ent.get())))
            print('animation finished. Took {} ms / frame'.format(int(ms)))
        """
        #self.ax1.set_facecolor('black')
        self.canvas.get_tk_widget().config(bg='#000000')
        return self.lines

    def mainloop(self):
        self.window.mainloop()

    # ------------------------------------------------------------------
    # Logical Functions:
    # The modules required to carry out game logic
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Drawing Functions:
    # The modules required to draw required game based object on canvas
    # ------------------------------------------------------------------


def check_messages():
    global gui_done
    #print("K",end='',flush=True)
    while True:
        if gui_done:
            exit_button()
        cmd=""
        try:
            cmd,value=cmdq.get_nowait()
        except queue.Empty as e:
            break
        if len(cmd)==0:
            pass
        else:
            #print(cmd,end='',flush=True)
            if cmd=='L':
                game_instance.button1.config(text=value)
            if cmd=='E':
                game_instance.button1.config(state=tk.NORMAL)
            if cmd=='X':
                gui_done = True
                game_instance.window.destroy()

def guithread():
    global gui_done, game_instance
    game_instance = Visualizations()
    while not gui_done:
        check_messages()
        game_instance.window.update()
        sd.sleep(1)
    #game_instance.mainloop()
    print("GUI exiting!!!")
    gui_done = True




t2 = threading.Thread(target=process_thread)
t2.start()

#t3 = threading.Thread(target=audio_handler)
#t3.start()

t1 = threading.Thread(target=guithread)
t1.start()
#guithread()

audio_handler()

t1.join()
t2.join()
#t3.join()
