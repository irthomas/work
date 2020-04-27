# Python scripts for working with NOMAD HDF5 data

## Installation

* Clone repo to your working directory
* Edit `tools.file.paths` to reflect your environment

## Issues

### Paths

On Linux, if the modules cannot be found, check the `PYTHONPATH`:

`echo $PYTHONPATH`

If the path is not included, add it as follows:

`export PYTHONPATH="${PYTHONPATH}:/path/to/work"`


Add this line to `.bashrc` in your home directory if you want the change to be permanent.


### Updates

Note that old scripts are not always adapted to the current directory structure and may not work.

The directories `instrument` and `tools` are under active development and should work correctly.
