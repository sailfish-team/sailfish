
Visualization and control architecture
======================================

[[toc]]

**Assumptions**:

* Computation (simulation, basic data post-processing) is distributed in a cluster.
* Direct access to individual nodes cannot be guaranteed (this is the case with
  some clusters, where only the head node is accessible from the outside world).
* 3D data is too large to be sent and processed in the browser, and therefore
  as much processing as possible should be delegated to the backend.

**Notation**:
Whenever this document refers to 'tensor', an N-dimensional array is meant. In particular,
the tensor can be a scalar or a vector.

# Visualization 

Basic visualization module requirements:

* Should have no dependencies, not use any plugins (e.g. Flash, ActiveX, etc), and 
  should work in a modern browser on any platform (Windows, **Linux**, OS X).
* Should work both as:
  * standalone UI, which can be hosted anywhere (HTML + JS) (e.g. github, us.edu.pl)
    and connected to a visualization backend
  * full UI embedded in an IPython notebook (should be easy if the previous point works)
  * individual components callable/embeddable in an IPython notebook
    (e.g. if we have support for visualizing streamlines, it should be possible
    to just call a function in IPython)

## Existing solutions

### ParaView

* scriptable with Python
* rich set of visualization and data processing functions
* can be used to render scenes off-screen (but setup might be non-trivial, involving starting a headless X server; this could likely be automated)

### ParaView Web

* provides components to access a remote ParaView rendering server
* can be embedded into an existing document (http://paraviewweb.kitware.com/apps/LiveArticles/)


### ThreeJS

* client-side rendering of OpenGL primitives
* can load and render STL files
* no complex mesh processing like VTK
* no complex visualization routines (e.g. streamlines)
* examples exist for embedding within an IPython notebook

### VTK

* can be used to create rendered scenes that are then sent to the UI as an image
* provides utilities for mesh manipulation (e.g. decimation)

# Control

The following control commands need to be supported:

* adjust ''every'', ''from'' for data saving
* create a checkpoint
* create an output data dump
* add a probe point [future, needs support for probe points in Sailfish]

# Proposed architecture

The proposed architecture is presented visually below and extends the current
Sailfish architecture with two new components:

* the visualizer proxy
* visualization handlers

```
frontend
   ^
   |
   |  JSON, WebSockets
   |                                    ZMQ
   V              <------------------------------------------ vis. handler 2
visualizer proxy  <----------------------> vis. handler 1           ^
   ^                                         ^                      |
   |  ZMQ                                    |                      |
   v         ZMQ, execnet            ZMQ     V                      |
controller  <------------->  master <----> runner 1                 V
                                    <---------------------------> runner 2
```
**TODO**: This proposed architecture actually breaks a current assumption that
no communication exists between the subdomain runners and the controller. This
is unlikely to be real limitation though. If it turns out to be, the direct
connections between visualization handlers and the proxy could be tunneled
to the controller via the master. Note that current ``visualizer.py`` suffers from
the same problem.

## Visualizer proxy

The visualizer proxy could run in a separate process/thread in the controller.
Its role is to expose an externally accessible interface that can be reached
from an in-browser frontend. 

## Visualization handlers

The visualization handlers are threads started by the subdomain runners. They
have access to the full resources of the subdomain runner. Their roles are:

* to control the behavior of the runner, executing remote commands as needed
* to summarize data from the GPU and pass it back to the visualizer proxy

## Backend visualization functionality

Basic data requests to be sent to Sailfish:

* axis-aligned 2D slice of one or more fields, passed through a processing function (e.g. mag(ux, uy, uz))
* non-axis aligned 2D slice of one or more fields (data needs to be interpolated on the GPU), postprocessed as above
* axis-aligned 3D subvolume, postprocessed as above

### Processing functions

Ideally, the processing function could be an arbitrary sympy expression defined 
over the symbols representing the available macroscopic fields. This expression
can then be transformed into a CUDA function on the fly (we already do that e.g.
for time-dependent boundary conditions), compiled and loaded onto the device as
a CUBIN file, and called from the visualization thread / SubdomainRunner.

The following types of processing functions should be supported:

* pointwise (output dense tensor data)
* reducers (can be implemented efficiently on the GPU; useful for computing averages)
* [future] subsamplers; these could be used to provide subsampling of the data (every N-th point), or
  e.g. compute N streamlines

#### Extensions

We could allow field references to take relative coordinates, e.g. ``(ux[-1,0,0] + ux[1,0,0]) * 0.5``
could be used to compute a central difference. This is used in computing quantities
that require derivatives of the basic fields, such as vorticity.

We could also allow the called to provide a full CUDA kernel to be executed on
the device. This would provide the greatest flexibility, though it's not clear
if there are any good use cases for this.


## Embedding in IPython

TBD
