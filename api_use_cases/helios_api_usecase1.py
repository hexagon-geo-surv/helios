"""
All input specifications (scene, platform, scanner, survey) exist (encoded in XML files).
We load them, start a simulation and retrieve the outputs (points, traj) as numpy arrays for further processing/visualisation
"""

import helios
import numpy as np

survey_xml = "some_file.xml"
survey = helios.Survey.from_file(survey_file=survey_xml)  # or: helios.load_survey(survey_file=survey_xml) or sth. like that
# when loading from XML, validate using xmlschema (or pydantic in case of other format/JSON)

survey.fullwave(beam_sample_quality=3,  # extra entry point for full wave settings (probably not used sooo often, users will rather stick with the defaults)
                bin_size=0.2,  # [ns]
                win_size=1,  # [ns]
                max_fullwave_range=150,  # [m]
                min_echo_width=2.5,  # [ns]
                pulse_length=4)  # [ns]
# and/or
survey.fullwave.beam_sample_quality = 3
survey.fullwave.bin_size = 0.2
survey.fullwave.window_size = survey.fullwave.pulse_length / 4

# this should then also work:
# survey.fullwave = survey2.fullwave  # copy settings from other survey object

survey.run(threads=0,  # or survey.run()
           save_config=True)  # or `save_settings`/`write_xml`; write survey file (currently: survey XML file); can also be JSON
'''
Alberto:
Yes, using None is a valid alternative.
The other approach would be to define a class to represent the output
with methods to check it, also to get/read those values and maybe to
apply some post-processing function in a controlled/safe way.
'''
points, traj, _ = survey.output()  # one object for pc, traj (and fullwave?); structured numpy array; how to deal with fullwave here? Return "None" object, if the user did not want full waveform output?
# e.g., num_points x num_columns (x, y, z, GPSTime, intensity, etc.)
"""
Lukas:
The points from survey.output() should be ideally comparible with the point object from laspy
https://laspy.readthedocs.io/en/latest/api/laspy.lasdata.html
https://laspy.readthedocs.io/en/latest/api/laspy.point.record.html
"""

# three different options to get just the coordinates
coords = points[:, :3]
coords = points.xyz
coords = np.array([points.x, points.y, points.z]).T

# or get just a single point cloud attribute
intensity = points.intensity


# alternatively, to generate output to files
survey_xml = "some_file.xml"
survey = helios.Survey(survey_file=survey_xml)
survey.run(format="laz",  # alternatives: "las", "xyz" ("ascii", "txt" will also result in "xyz"); default should be LAS or LAZ
           output="output/",  # if "output" not set, do not write to file
           threads=0,
           write_waveform=True,  # default is False
           calc_echo_width=True,
           gps_start_time="2024-03-07 10:00:00"  # shoudl also accept a Python datetime object, e.g. datetime.now
           # additionally, kD-tree stuff and other CLI arguments?
           )
points, traj, fullwave = survey.output()
"""
Lukas:
survey.run() should start the run but Python code continues until survey.output() is called, so that we can do stuff in the meantime
i.e., survey.output() blocks the thread

Ideally, we would also provide a callback, i.e., survey.run(on_finished=...) and/or survey.run(on_progress=...) where a function is called
when progess is made or the survey is finished, similar to how it works with pyQT (--> helios_api_usecase7.py)
"""


# how best to set verbosity level? As argument of survey.run() or at some higher level (e.g., helios.set_verbosity())
