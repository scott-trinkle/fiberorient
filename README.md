# fiberorient

![](imgs/demo.png)

fiberorient is a package for performing structure tensor analysis and
calculating orientation distribution functions from 3D imaging data.

Code, data, and examples stem from work published as: 

Trinkle, S., Foxley, S., Kasthuri, N., La Rivi`ere, P., “[Synchrotron x-ray
micro-ct as a validation dataset for diffusion mri in whole mouse
brain](https://doi.org/https://doi.org/10.1002/mrm.28776),” Magnetic Resonance
in Medicine, vol. 86, no. 2, pp. 1067–1076, 2021.

For more information, see [my blog
post](https://www.scotttrinkle.com/news/microct-paper/).

## Installation

For local installation, first clone the repo:

`git clone https://github.com/scott-trinkle/fiberorient.git`

`cd` into the directory:

`cd fiberorient`

and run (preferably in a virtual environment): 

`pip install -e .`

## Usage

Coming soon!

### Axis convention

Assume the shape of the input data is `(n0, n1, n2)`. Each orientation output from
the structure tensor analysis pipeline will be a 3D vector `vec` with `vec[0]`
along the n0 direction, `vec[1]` along the n1 direction, and `vec[2]` along the
n2 direction. 

For expansion onto spherical harmonics and visualization with the fury package,
it is convenient to define these dimensions as a right-handed coordinate system
with `(n0,n1,n2)` corresponding to `(x,y,z)`, where the polar angle $(0,\pi)$ is
formed by cos$^{-1}$(z) and the azimuth $(0,2\pi)$ is formed by tan$^{-1}$(y/x).

## Data

Coming soon!
