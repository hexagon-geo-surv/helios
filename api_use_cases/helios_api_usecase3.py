"""
We consecutively want to run several simulations with an existing scene with different acquisition settings

Examples for different scenarios
3.1: ULS - Change of platform settings (flight altitude and speed)
3.2: TLS - Change of scanner settings (horizontal and vertical resolution)
3.3: TLS - Change of legs (scan positions)
"""

import helios

scene_xml = "some_file.xml"

# 3.1 Change of altitude and speed (ULS example)

scene = helios.load_scene(scene_file=scene_xml)
altitudes = [50, 75, 100]
speeds = [3, 4.5, 6]
scan_angle = 60   # off-nadir
pulse_freq = 600_000
scan_freq = 60_000

scan_pattern = helios.ScanSettings(scan_angle=scan_angle, pulse_freq=pulse_freq, scan_freq=scan_freq)
platform_setting_list = helios.utils.create_platform_setting_list(speed=speeds,
                                                                  altitude=altitudes,
                                                                  combination_mode="zip")  # returns list of 3 template objects  # umbenennen
# other `combination_mode` could be "cartesian_product" (i.e., all possible combinations); would return list of 9 template objects

wp = [[50, -100], [50, 100], [-50, 100], [-50, -100]]  # if no z value provided, use value from platform template?

base_survey = helios.Survey(scene=scene,
                            scanner=helios.Scanner.get_scanner_by_id("riegl_vux-1uav22"),
                            platform=helios.Platform.get_platform_by_id("quadcopter_linearpath"),
                            name="use_case_3_1")
base_survey.add_legs(pos=wp, scanner_template=scan_pattern)

# store outputs in dictionary
outputs = {}
for i, platform_settings in enumerate(platform_setting_list):
    survey = base_survey.clone()  # or .copy()
    survey.platform_settings = platform_settings
    # could also be done on the leg level:
    # survey.legs[2] = platform_settings2
    survey.name += f"_{i}"
    # or encode the survey settings in name
    survey.name += f"_{altitudes[i]}m_{speeds[i]}m_per_s".replace(".", "dot")
    survey.run(format="laz",
               output="output/")
    outputs[survey.name] = survey.output()


# 3.2 Change of horizontal and vertical resolution (TLS example)

scene = helios.load_scene(scene_file=scene_xml)
scan_pos = [[-5, 0], [0, 5], [5, 0], [0, -5]]
horizontal_res = vertical_res = [0.017, 0.04, 0.1]

templates_list = helios.util.create_scanner_setting_list(pulse_freq=300_000,
                                                         vertical_fov=120,
                                                         horizontal_res=horizontal_res, vertical_res=vertical_res,
                                                         combination_mode="zip")  # returns list of three template objects

base_survey = helios.Survey(scene=scene,
                            scanner=helios.Scanner.get_scanner_by_id("riegl_vz400"),
                            platform=helios.Platform.get_platform_by_id("tripod"),
                            name="use_case_3_2")
base_survey.add_legs(pos=scan_pos)  # add positions but do not set settings yet (i.e., use the defaults currently)

for i, template in enumerate(templates_list):
    survey = base_survey.legs.set_scanner_template(template)
    # alternatively, get legs separately and assign template to them (esp. if using different templates for different legs):
    for leg in base_survey.legs:
        leg.set_scanner_template(template)
    survey.name += f"_{horizontal_res*100}_mm_res"
    survey.run(format="laz",
               output="output/")
    pc, _, _ = survey.output()


# or without scanner setting template
base_survey = helios.Survey(scene=scene,
                            scanner=helios.Scanner.get_scanner_by_id("riegl_vz400"),
                            platform=helios.Platform.get_platform_by_id("tripod"),
                            name="use_case_3_2")
base_survey.add_legs(pos=scan_pos, pulse_freq=300_000, vertical_fov=120, horizontal_res=0.017, vertical_res=0.017)
# each iteration, add 0.0017 ° to the resolution to get scans of the same scenes and different point densities
for i in range(5):
    survey = base_survey.clone()
    for leg in survey.legs:
        leg.vertical_res += 0.017
        leg.horizontal_res += 0.017
    survey.run(format="laz",  # question is, if the survey object is "reusable" after completing the simulation?
               output="output/")
    pc = survey.output().to_numpy()
'''
Alberto:
In principle, once the simulation has finished there is a shutdown
and the corresponding resources (e.g., memory) are released.
That's why we need to copy/clone a Survey before running it and then use that other survey.
Alternatively, now we have a SimulationPlayer to support many runs
of the same simulation (with potential updates of some scene parts),
thus it can be used to run the same simulation many times. It is
automatically engaged when specifying swaps for any scene part.
'''

# 3.3 Change of legs (here: TLS scan positions)

scene = helios.Scene.from_file(scene_file=scene_xml)  # or helios.load_scene()
horizontal_res = vertical_res = 0.04

template = helios.ScanSettings(pulse_freq=300_000,
                               vertical_fov=120, horizontal_fov=100,
                               horizontal_res=horizontal_res, vertical_res=vertical_res)

base_survey = helios.Survey(scene=scene,
                            scanner=helios.Scanner.get_scanner_by_id("riegl_vz400"),
                            platform=helios.Platform.get_platform_by_id("tripod"),
                            name="use_case_3_3")

# scanning grid with three different densities
grid_spacings = [10, 15, 20]

# for each grid spacing, create a new survey
for grid_spacing in grid_spacings:
    scan_pos = helios.util.sampling_grid(30, 30, grid_spacing=grid_spacing)  # some survey planning utility function, cf. helios_util.py; similar to what we have in "flight_planner"
    survey = base_survey.clone()
    survey.legs = []  # remove all existing legs (if any)
    survey.add_legs(scan_pos, scan_pattern=template)
    survey.run(format="laz",
               output="output/")
