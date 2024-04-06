"""
We have a couple of scene parts, from which we want to assemble the scene, then run a survey
"""

import helios
import copy


# OPTION 1
# read each scene part separately from file (and apply some transformations)
# could also be called "read_scenepart"
cube = helios.Scenepart.from_file("sceneparts/toyblocks/cube.obj")  # recognizes file format from extension
cube = helios.Scenepart.from_file("sceneparts/toyblocks/cube.obj", file_format="obj")  # support additional parameter to set the file format explicitly
cube2 = copy.deepcopy(cube)
'''
Alberto:
The main benefit of one function for each is to have fluent programming.
Alternativel one could also do
.transform(translation=x).transform(scale=y)
We could easily write .translate(x) to be an alias for
.transform(translation=x), then users can decide what they want to use.

Lukas:
if translate, rotate are given in the same function call, which order are they applied in?
What happens with subsequent transformations, do they replace or do they “stack”?
'''
cube2.transform(translation=[10, 0, 0], scale=[0.5], on_ground=1)  # all trafos in one function or better separate .translate(), .scale(), .rotate()?
apple_tree = helios.Scenepart.from_file("sceneparts/trees/apple.obj")
apple_tree.transform(rotation=[0, 0, 45])
beech_tree = helios.Scenepart.from_file("sceneparts/trees/beech.obj")
house = helios.Scenepart.from_file("sceneparts/urban/house.obj")
cloud = helios.Scenepart.from_file("sceneparts/xyz/cloud.xyz", voxel_size=0.25)  # custom parameter for the xyzloader (-> voxel model)
terrain = helios.Scenepart.from_file("sceneparts/terrain/terrain.tiff")

# instead of reading from file, also enable interface with e.g., open3D (or other popular software), at least for 3D meshes
# i.e., create OBJ in open3D and provide this as a scenepart to helios
import open3d as o3d
armadillo_mesh = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
armadillo = helios.Scenepart.from_o3d(mesh)  # or sth. like that

# list all sceneparts that shall be included in the scene
scene = helios.Scene(sceneparts=[terrain, cube, cube2, apple_tree, beech_tree, house, cloud], scene_id="my_scene")
scene_plot = scene.show()  # of course, we want some visual feedback on how the scene looks like
# with options to further modify the rendering using whatever framework (open3D, PyVista, matplotlib, etc.)

scan_pattern = helios.ScanSettings(pulse_freq=300_000, scan_angle=60, scan_freq=100)
platform_settings = helios.PlatformSettings(speed=5, altitude=50)

survey = helios.Survey(scene=scene,
                       scanner=helios.Scanner.get_scanner_by_id("riegl_vux-1uav"),
                       platform=helios.Platform.get_platform_by_id("quadcopter_linearpath"),
                       name="use_case_4_uls",
                       trajectory_interval=0.05)

# define waypoints
wp = [[50, -100], [50, 100], [-50, 100], [-50, -100]]
active = [1, 0, 1, 0]

survey.add_legs(pos=wp, scan_pattern=scan_pattern, platform_settings=platform_settings)
survey.run()  # no output folder given, so do not write file
points, traj, _ = survey.output()
