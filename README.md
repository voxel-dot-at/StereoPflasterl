# StereoPflasterl

An interactive tool to patch stereo camera misalignment for [OpenCV](https://opencv.org/)


StereoPflasterl  reads two configuration files: One with intrinsic parameters of the first and second camera,  and a file with the extrinsic parameters (R, T matrices) defining the geometric relation between both. The information is typically created by single and stereo camera calibration.

start stereoPflasterl by passing a link to the config file and an image pair:

```
stereoPflasterl -cf <config.yaml> -l <left.jpg> -r <right.jpg>
```

The main config file contains links to the intrinsic and extrinsic camera parameters, adjustment values for various optical aspects and parameters for the stereo matching itself.

The rationale behind the adjustment parameters is to keep the original geometry information unchanged so there is always a way back to a known state. with 'p' you can print the current parameter set including the final matrices for inclusion in updated yaml files.


* Example config.yaml

```
%YAML:1.0

calib_intrinsic: "intrinsics.yml"
calib_extrinsic: "extrinsics.yml"

algorithm: sgbm
offset: 0

bounding_box:
    max_x: 8.0
    min_x: -8.0
    max_y: 19.0
    min_y: -19.0
    max_z: 11.0
    min_z: -2.0

#
# debugging: display rectified image, disparity map, 3d point cloud (no)
#
display_options:
    rectified: 1
    disparity: 1
    cloud: 0

rgb: 1

#
# Intrinsics: adjust FoV (multiplicative factor) and image center (in pixels):
#             left refers to camera 1 (M1, D1), right to camera 2 (M2, D2)
mul_left: 1.0001
pos_left_x: 0.01
pos_left_y: -0.002

mul_right: 1.0001
pos_right_x: 0.01
pos_right_y: -0.002


#
# Extrinsics: translation parameters - added to the T matrix values 
#

#
# Extrinsics: rotation parameters - added to the R matrix values 
#
rotate_180: true
rot_x: 0.002882276329806893
rot_y: 0.0006651406914938984
rot_z: -0.0001108567819156497


#
# stereo matching parameters for sgbm; this is an RGB sensor so we scale the image down to eliminate RGGB bayer patterns
#
minDisparity:            0    
numDisparities:        140  
blockSize:               7 
speckleWindowSize:      20  
speckleRange:            2   
disp12MaxDiff:           1   
preFilterCap:            0   
uniquenessRatio:         5   
scale:                 0.5 

```

* Example intrinsics.yaml

The data has been derived from a quick calibration of an Elphel.com stereo head. If you look at the intrinsic matrices, you'll notice small deviations in the field of view values, for example. If you have the chance to cross-check with real-world measurements you want to adjust these (hypothesis: the lenses resp. the camera sensor plane could have moved out of focus a little, or the initial calibration was not super-perfect).

```
%YAML:1.0
M1: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 2.1102845422039572e+03, 0., 1.3371777817527804e+03, 0.,
       2.1099967106802037e+03, 9.6770516528651433e+02, 0., 0., 1. ]
D1: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [ -9.8237302363664070e-02, 8.5404257071666295e-02,
       2.2157292082804601e-04, -6.7555133450274797e-05,
       1.4949231159613949e-01 ]
M2: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 2.1080454863154951e+03, 0., 1.2944130935034059e+03, 0.,
       2.1083919572481072e+03, 1.0112601425228062e+03, 0., 0., 1. ]
D2: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [ -9.8666718175414794e-02, 9.7779763850499885e-02,
       -2.8453253031277844e-04, -4.4661929761932175e-04,
       1.3075339671327391e-01 ]
```

* Example extrinsics.yaml

The extrinsics specify the translation of the second camera with respect to the first (n.b. that values are in cm in this calibration!) and the rotation matrix of the optical center - print it as a 3x3 matrix or use rotationMatrixToEulerAngles() to convert the values into angles.
```
%YAML:1.0
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ -2.50e+01, -1.1853301202394763e-01,
       -1.3956485320394347e-01 ]
R: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 9.9999266681168397e-01, 1.1576710377343971e-03,
       3.6504959423130801e-03, -1.1110750174500405e-03,
       9.9991821802791436e-01, -1.2740595283811542e-02,
       -3.6649468157193893e-03, 1.2736445879784260e-02,
       9.9991217169868940e-01 ]
```

## Compiling

StereoPflasterl has been compiled with

* OpenCV v 4.5
* gcc 3.20
* gcc v12

but should work with a broad range of variants.