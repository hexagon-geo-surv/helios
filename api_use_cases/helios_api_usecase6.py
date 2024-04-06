"""
We want to define a custom scanner, then start a survey using this scanner (over an existing scene)
"""

import helios

scanner_xml = "data/myscanners.xml"
# new TLS scanner
new_scanner = helios.Scanner(
    id="riegl_vz600i",
    name="RIEGL VZ-600i",
    optics="rotate",
    accuracy_m=0.005,
    beam_divergence_rad=0.00035,  # 0.35 mrad at the 1/e2 points
    pulse_freqs_hz=[140_000, 600_000, 1_200_000, 2_200_000],
    max_nor=[5, 10, 15, 15],  # typically, we can only set one value, maybe we can extend HELIOS++ to take different values depending on chosen pulse frequency
    scan_angle_max_deg=120,
    scan_angle_effective_max_deg=52.5,  # -40° to +65°
    scan_freq_min_hz=4,
    scan_freq_max_hz=420,
    head_rotate_per_sec_max=360,
    beam_sample_quality=3,
    beam_origin={
        "loc": [0, 0, 0.2],
        "rot": [0, 0, 0]},  # angle for x, y, and z?
    head_rotate_axis=[0, 0, 1]
)
new_scanner.to_xml(scanner_xml)  # write to XML file (if file exists, append at the end within the <document> check; might include a check if the same scanner ID already exists; if so: raise warning or error)
# instead of a file, could also just provide a name and save in some local helios folder (same location as "scanner_templates"/"scan_patterns", etc.)
"""
Lukas:
`.to_xml()` could create the XML string, and `.to_file(…)` could write this to either a file object handle or a path if a string is given.
"""

scene_xml = "some_file.xml"
scene = helios.Scene(scene_xml)

template = helios.ScannerSettings(pulse_freq=1_200_000, horizontal_res=0.017, vertical_res=0.017,
                                  vertical_angle_min=-40, vertical_angle_max=65,
                                  horizontal_fov=100)

survey = helios.Survey(scene=scene,
                       scanner=new_scanner,
                       platform=helios.Platform.get_platform_by_id("tripod"),
                       name="use_case_6_tls")
focus_point = (0, 0, 0)
survey.add_leg(look_at=focus_point, scan_pattern=template,
               pos=(3, 0, 0), horizontal_fov=90, horizontal_res=0.017, vertical_res=0.017)
survey.add_leg(look_at=focus_point, scan_pattern=template,
               pos=(0, 3, 0), horizontal_fov=90, horizontal_res=0.017, vertical_res=0.017)

'''
Alberto:
I like .to_xml, then we could also support other formats in the future
in a similar way to numpy with diferent .to_x methods.
'''
survey.to_xml(survey_xml="data/surveys/survey_file.xml",  # write survey to XML file for archiving/documentation purpose; could also be called .archive()
              )
# could also be done via an argument to the "start" method
# by default, will write the survey only (if it uses an existing scene XML and existing scanners)

# to save more configurations (not only the survey.xml)
'''
Alberto:
Yes, you can export the scene parts (it is a good solution).
However, there are alternatives, like serializing all of them to a
single binary file that can be read later on. Then, future simulations
should have a smaller scene loading time because the binary scene parts
are directly loaded in memory without further processing.

Hannah: So we could add sth. like `scenepart_bin="data/sceneparts/sceneparts.bin` to the function below?

Lukas:
I would use `.to_xml()` and `.to_file()` on survey, scene, and scanner to write the individual XMLs/Files and
`.save()` on the survey to save all of the configurations at once.
This could also write a zip file with all required files incl. input sceneparts, etc.
'''
survey.save(survey_xml="surveys/survey_file.xml",
            scene_xml="scenes/scene_file.xml",
            scanner_xml="scanners.xml")  # discuss if always following the same schema (i.e., subfolders for sceneparts, scene) or how much the user can customize how the outputs are written

# old suggestion
# survey.to_xml(survey_xml="data/surveys/survey_file.xml",
#               scene_xml="data/surveys/scene_file.xml",  # problem: what if sceneparts generated in Python and not loaded from file - how to write the scene XML? Only possible if scenepart is exported?
#               scanner_xml=scanner_xml,
#               )

survey.run(format="las",
           output="output/")
