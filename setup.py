"""
Created on Wed Jan 13 23:47:44 2021
@author: skevja
"""

from setuptools import setup, find_packages
import os

supplementary_data = [os.path.join(dp,f) for dp,dn,filenames in os.walk('jjmodel/input/gas') 
                      for f in filenames if (os.path.isfile(os.path.join(dp, f)) & ('~' not in f))]
supplementary_data1 = [os.path.join(dp,f) for dp,dn,filenames in os.walk('jjmodel/input/mass_loss') 
                       for f in filenames if (os.path.isfile(os.path.join(dp, f)) & ('~' not in f))]
supplementary_data2 = [os.path.join(dp,f) for dp,dn,filenames in os.walk('jjmodel/input/isochrones') 
                       for f in filenames if (os.path.isfile(os.path.join(dp, f)) & ('~' not in f))]
supplementary_data3 = [os.path.join(dp,f) for dp,dn,filenames in os.walk('jjmodel/tutorials') 
                       for f in filenames if (os.path.isfile(os.path.join(dp, f)) & ('~' not in f))]
supplementary_data4 = [os.path.join(dp,f) for dp,dn,filenames in os.walk('jjmodel/docs') 
                       for f in filenames if (os.path.isfile(os.path.join(dp, f)) & ('~' not in f))]

supplementary_data += supplementary_data1
supplementary_data += supplementary_data2
supplementary_data += supplementary_data3
supplementary_data += supplementary_data4

for i,item in enumerate(supplementary_data):
	supplementary_data[i] = supplementary_data[i][8:]


def readme():
    with open('README.md') as f:
        return f.read()

setup(name = "jjmodel",
      version = 0.3,
      description = "Dynamical model of the MW disk (Just-Jahreiss model)",
      long_description = readme(),
      author = "Kseniia Sysoliatina",
      author_email = "k.sysoliatina@gmail.com",
      url = "https://github.com/",
      packages = find_packages(),
      package_dir = {'jjmodel' : 'jjmodel'},
      package_data = {'jjmodel' : supplementary_data},
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Operating System :: Linux',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering'
          ],
      zip_safe=False
     )
