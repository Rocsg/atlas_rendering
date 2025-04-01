# 🌍 Atlas Rendering

**Atlas Rendering** is a project allowing for the 3D render of plants, originally rice plants, with the use of X-Ray acquisitions of said plants.

## 🚀 Features
- Structures of interest have to be generated with the programs from the surface_generation module. These programs use csv representing the position of the centerlines of said structures. A csv radius can be used to sample the radius along the center. If not used use a csv with only 2 points with both having the same radius. An information radius is used to indicate radius changes on structures with no potent data for radius.
-A volume Render is made from a tiff file. If said tiff file comes from a bigger tiff file the offset(x,y,z) must be indicated in the config file for it to be well placed.
