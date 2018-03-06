#!/usr/bin/env python

from distutils.core import setup

try:
    from catkin_pkg.python_setup import generate_distutils_setup
    setup_args = generate_distutils_setup(
        packages=['phasestatemachine'],
        package_dir={'': 'src'})
    setup(**setup_args)

except ImportError:
    setup(name='Phase-state machine module',
          version='0.1.0',
          description='Library to create and execute phase-state machines',
          author='Raphael Deimel',
          author_email='raphael.deimel@tu-berlin.de',
          url='http://www.mti-engage.tu-berlin.de/',
          packages=['phasestatemachine'],
          package_dir={'': 'src'},
         )

