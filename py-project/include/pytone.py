#!/usr/bin/env python
"""Play a fixed frequency sound."""
from __future__ import division
import math

from pyaudio import PyAudio # sudo apt-get install python{,3}-pyaudio
from threading import Thread
from time import sleep

try:
    from itertools import izip
except ImportError: # Python 3
    izip = zip
    xrange = range

def sine_tone(frequency, duration, volume=1, sample_rate=22050):
    n_samples = int(sample_rate * duration)
    restframes = n_samples % sample_rate

    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(1), # 8bit
                    channels=1, # mono
                    rate=sample_rate,
                    output=True)
    s = lambda t: volume * math.sin(2 * math.pi * frequency * t / sample_rate)
    samples = (int(s(t) * 0x7f + 0x80) for t in xrange(n_samples))
    for buf in izip(*[samples]*sample_rate): # write several samples at a time
        stream.write(bytes(bytearray(buf)))

    # fill remainder of frameset with silence
    stream.write(b'\x80' * restframes)

    stream.stop_stream()
    stream.close()
    p.terminate()


def play_tone(y=0):
    # play low-tone
    frequency=(y+1)*100.00  # Hz, waves per second A4
    duration=1              # seconds to play sound
    volume=1              # 0..1 how loud it is
    sample_rate=22000       # number of samples per second
    # send audio to thread
    thread1 = Thread(target = sine_tone, args = (
        frequency, duration, volume, sample_rate,
    ))
    thread1.start()