"""
We load an existing scene (from XML) and then configure the survey (platform, scanner, settings)
with Python and start the simulation

Examples for both TLS and ALS below; waypoint-based and (interpolated) trajectory-based
"""

import helios
import numpy as np

scene_xml = "some_file.xml"

# a) TLS

'''
Alberto:
I don't have an opinion on this but it is important to decide whether
we want to follow a many modules structure (e.g., Scanner, Platform,
Scene, ScenePart, Primitive) or a simpler alternative with almost
everything accessible from the helios module). This will condition the
API design in a fundamental way and thus we must make it clear as soon
as possible.
'''
scene = helios.Scene.from_file(scene_file=scene_xml)  # or: helios.load_scene(scene_file=scene_xml)
scene.show()  # visualize scene prior to simulation (e.g. with open3D/PyVista)

# predefined template to make the life of users easier
scan_pattern = helios.get_scanner_template(preset="riegl_vz400_highres")  # optional feature to choose from built-in presets
# might be a helios.util function?
# or be called: helios.ScanSettings.get_preset() / .get_template()
# user should also be able to save their own presets and load them, see below

# e.g., this one works well for a RIEGL VZ400; might look like this
scan_pattern = helios.ScanSettings(pulse_freq=300_000,
                                   horizontal_res=0.04,
                                   vertical_res=0.04,
                                   horizontal_fov=360,
                                   vertical_angle_min=-40,
                                   vertical_angle_max=60,
                                   save_as="riegl_vz400_hannahs_pattern")
# save in some local helios user folder? in suitable file format (XML/JSON/.py)
# or:
scan_pattern.to_file()  # or scan_pattern.save(...) ?

'''
Alberto:
We could implement further validation methods (even configurable through
YAML files) on the Python side. Maybe a Python package for validation
strategies is worth the effort. It is easier to write these things in
Python and other researches could contribute with less effort.
Just an idea.
Hannah: Yes, something like Pydantic?
'''
survey = helios.Survey(scene=scene,
                       scan_pattern=scan_pattern,  # question: how to validate that settings are allowed for the scanner?
                       scanner=helios.Scanner.get_scanner_by_id("riegl_vz400"),  # maybe with the optional parameter "source_xml", taking a file path
                       platform=helios.Platform.get_platform_by_id("tripod"),
                       name="use_case_2_tls",  # determines name of the folder in which output will be stored
                       trajectory_interval=0.05)  # see `trajectory_time_interval`, controls at which GPSTime Interval the trajectory is written

# look_at option for TLS only, should derive the start and stop head rotation so the centre of the field of view points toward the given point or in the given direction
focus_point = (5, 1)
survey.add_leg(look_at=focus_point, scan_pattern=scan_pattern,
               pos=(0, 0, 0), horizontal_fov=90, horizontal_res=0.017, vertical_res=0.017)  # settings from pattern (fov, resolution) are overwritten
"""
Lukas:
'look_at' for TLS (platform='tripod') is interesting but I'm not sure about it defining the _center_ of the fov.
Maybe this could also be replaced by a helper function
helios.utils.define_leg_from_focuspoint(…) that returns a leg object that can then be passed to survey.add_leg?
"""
survey.add_leg(look_at=90, scan_pattern=scan_pattern,  # instead of focus point, provide direction as number in degrees (0° for north)
               pos=(5, 5, 0), horizontal_fov=90, horizontal_res=0.017, vertical_res=0.017)

# instead of leg by leg, define multiple legs at one using an array of positions
survey.add_legs(horizontal_fov=360, scan_pattern=scan_pattern,
                pos=[[-5, -5, 0], [-5, 0, 0], [5, 5, 0], [0, -5, 0], [0, 0, 0], [0, 5, 0]])  # should also be possible to use focus_point here
# sensible broadcasting should be supported, i.e., the following also works:
survey.add_legs(horizontal_fov=360, scan_pattern=[pattern1, pattern2, pattern1, pattern2, pattern3, pattern3],
                pos=[[-5, -5, 0], [-5, 0, 0], [5, 5, 0], [0, -5, 0], [0, 0, 0], [0, 5, 0]])  # scan pattern will be applied to respective legs

survey.run(format="laz",
           output="output/",
           threads=0)


points, traj, _ = survey.output()


# b) ALS

scan_pattern = helios.get_scanner_template(preset="riegl_lms-q780")
# e.g., this one works well for a RIEGL LMS Q780; might look like this:
# pulse_freq=300_000, scan_angle=45, scan_freq=100

platform_settings = helios.PlatformSettings(speed="30", altitude="600")  # speed in m/s, altitude in m
# idea: also allow altitude above ground -> need to define "ground" object; use `helios_isGround` material definition?

survey = helios.Survey(scene=scene,
                       scanner=helios.Scanner.get_scanner_by_id("riegl_lms-q780"),
                       platform=helios.Platform.get_platform_by_id("sr22"),
                       scan_pattern=scan_pattern,
                       platform_settings=platform_settings,
                       name="use_case_2_als")

'''
Alberto:
I like the flight planner module both for a full python specification
and also to read directly from file.
'''
# different options to define waypoints
# 1) provide array
wp = [[50, -100], [50, 100], [-50, 100], [-50, -100]]  # if no z value provided, use value from platform template?
active = [1, 0, 1, 0]  # first and third leg are active, others not; boolean mask
# 2) use flight planner module
wp = helios.util.flight_planner.compute_flight_lines(bounding_box=[-50, -100, 50, 100], spacing=50, rotate_deg=0.0, flight_pattern="parallel")
# 3) Load from file
wp = helios.util.flight_planner.flight_lines_from_file("some_file.shp")  # also allow JSON files etc.

survey.add_legs(pos=wp, active=active, scan_pattern=scan_pattern, platform_settings=platform_settings)
# question: would it be better to split function into "add_waypoints() for ALS and "add_scanpos()" for TLS?

survey.preview()  # some kind of visualization? e.g., 2D map preview, 3D preview with scene and markers for SPs/waypoints, etc.
# survey.preview("2D")
# survey.preview("3D")  # similar to scene.show(), but with marker (e.g. diamonds) for leg positions;
# for static platforms also show movement direction and orientation of platform

survey.run(format="laz",
           output="output/",
           threads=0)

points, traj, _ = survey.output()
'''
Alberto:
Also as this is a research software, I think writing some tables
and/or CSV files with the characteristics of the survey would be nice.
I mean things such as the number of primitives, vertices, the simulation
time (not the execution time), the volume of the bounding box, the
depth for each KDTree in the KDGrove, etc.
Hannah: Yes, so something like a "report" or "summary", see below
'''
report_folder = "/path/to/folder"
survey.save_report("/path/to/folder")  # tbd how this looks like exactly, probably several files, like
# summary_scene.csv (with the number of primitives, vertices, scene parts, volumes, etc.)
# summary_KDGrove (with the depth of the KDTrees, etc.?)
# summary_simulation.csv (with the simulation time, the number of points, the number of legs, the length of the trajectory (if it is a moving platform))

# c) ULS (trajectory instead of waypoints)

scan_pattern = helios.get_scanner_template(preset="riegl_vux-1uav-22")
# e.g., this one works well for a RIEGL LMS Q780; might look like this:
# pulse_freq=600_000, scan_angle=90, scan_freq=100

platform_settings = helios.PlatformSettings(speed="5", altitude="60")  # speed in m/s, altitude in m

survey = helios.Survey(scene=scene,
                       scanner=helios.Scanner.get_scanner_by_id("riegl_vux-1uav-22"),
                       platform=helios.Platform.get_platform_by_id("copter_linearpath"),
                       scan_pattern=scan_pattern,
                       platform_settings=platform_settings,
                       name="use_case_3_uls")

input_traj = np.loadtxt("data/trajectories/trajectory.txt")  # e.g., t x y z roll pitch yaw
survey.trajectory(traj_array=input_traj)

# or: if the order is not like assumed by default (t x y z roll pitch yaw)
survey.trajectory(traj_array=input_traj, columns="x y z roll pitch yaw t")  # or `x_index`, `y_index`, etc., tbd.

"""
Lukas: What happens if I use `survey.trajectory()` and `survey.add_leg()` on the same survey?

Hannah: Mhm, good question. The way they currently work, those options are exclusive, so helios we would probably
a) raise an error if you were to use one after using the other or
b) raise a watning and overwrite the other.

Lukas: Would raise a warning, except legs = []

Hannah: But would they "erase" a previously defined trajectory?

Lukas: Rather not.
maybe one could 'free' the trajectory with `survey.trajectory=None`
and other wise, you would get:
ValueError: Overwriting a trajectory in a survey is not supported. Set it to _None_ first explicitly.
"""

survey.run(format="laz",
           output="output/",
           threads=0)

points, traj, _ = survey.output()
