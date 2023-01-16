#!/bin/bash

import pyo
import time
s = pyo.Server(audio="jack",nchnls=1,ichnls=1).boot()
s.start()

time.sleep(1)
i = pyo.Input(1)
i.out()


s.gui(locals())

