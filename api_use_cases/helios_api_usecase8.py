"""
We want to add rigid motions (translation, rotation) to our loaded sceneparts. This may also be useful for the Blender add-on.

The Blender add-on `dyn_b2h` iterates over the keyframes, determines the number of frames in between the keyframes and the difference in translation and rotation
and adds this to the XML with the `<dmotion>` syntax, see https://github.com/3dgeo-heidelberg/dyn_b2h/blob/main/dyn_b2h/operators.py

Instead of writing XML code, we could have a Python function to add a single motion (composed of different motions) or
to add a sequence of motions in a vectorized way.
Below, there are two suggestions.
"""

import helios

# info coming, e.g., from blender animation
fps = 30  # frame rate in frames per second
frame_diff = 1  # frame difference between each keyframe = between each motion


# read scene part and apply a single motion
cube = helios.Scenepart.read("sceneparts/toyblocks/cube.obj")  # ideally, recognizes file format from extension
cube.add_motion(translation=[0.5, 0, 0],
                rotation_axis=[0, 0, 1],  # rotate around z
                rotation_angle=5,  # in degrees; eventually, decide if we want to support other rotation notation
                radians=False,  # as default?
                rotation_centre=[0, 0., 0.],  # coordinate origin as default, but can be set by the user
                loop=1  # how often to execute; 0 means the motion will be applied as infinite loop
                )
# further supported arguments:
# rotate_around_self: boolean, if true, rotation axis is centered in the object
# auto_crs: if set to 1, automatically translates the rotation center to the internal reference system of the sim; 0 or 1; default equal to 1

# discuss, if to expose all the arguments related to reflection, glide plane, helical motion, and rotation symmetry, see https://github.com/3dgeo-heidelberg/helios/wiki/Dynamic-scenes
# probably, rotation and translation are sufficient

# add the next motion in the sequence
cube.add_motion(translation=[0.2, 0, 0],
                loop=50
                )  # move 10 m in x direction, in 50 frames/steps (50 * 0.2)
cube.add_motion(translation=[10, 0, 0],
                duration=50
                )  # move 10 m in x direction, in 50 frames/steps (same result as above)

cube.dyn_time_step = frame_diff / fps  # duration of one time step in s

# Instead of adding each motion one by one to the sequence, have a function to directly specify a motion sequence using arrays
# more likely to be used by the plugin, i.e., in the loop, store the transformation info in arrays and then add the full motion sequence with helios:
cube.add_motion_sequence(  # of N motions
    translations=arr,  # N x 4 arraym each row has [x, y, z]  # or None
    rotations=arr2,  # N x 4 array, each row has [x, y, z, angle]  # or None
    rotation_centres=arr3,  # N x 3 array, each row has [x, y, z]  # or None  (then scene origin will be used if "rotations" is defined)
    # loop = 1 by default for each motion in the sequence
)

# list all sceneparts that shall be included in the scene
scene = helios.Scene(sceneparts=[cube])

# then simulate
# ...

"""
Lukas:
- Can we visualize the motion somehow in a way similar to the other visualization options?
- Would it make sense to have a "MovingScenepart" object that contains the "BaseScenepart"
(which can be an obj, vox, …) and a list of motions, rather than appending the motions to
the scenepart directly? This is just brainstorming, I'm not sure about it.
"""
