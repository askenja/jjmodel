"""
Created on Tue Feb 23 15:44:21 2016
Init module 
@author: Skevja
"""

import os
import inspect
from .version import __version__

localpath = '/'.join(os.path.abspath(inspect.getfile(inspect.currentframe())).split('/')[:-1])
localpath += '/'
