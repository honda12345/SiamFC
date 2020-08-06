from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    root_dir = os.path.expanduser('/home/honda/data/GOT-10k')
#    root_dir = os.path.expanduser('/home/honda/data/ILSVRC')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
#    seqs = ImageNetVID(root_dir, subset=('train', 'val'))

    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
