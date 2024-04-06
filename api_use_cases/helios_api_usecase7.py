"""
We want to start a given survey (load from scene for simplicity).
During simulation, we want to visualize the acquired points in a 3D plot,
i.e., retrieve the simulated points and the trajectory points in regular
intervals and update the plot
"""

import helios
import time
import numpy as np


survey_xml = "some_file.xml"
survey = helios.Survey(survey_file=survey_xml)

scene = survey.get_scene()  # get scene that is associated with the survey
scene_vis = scene.show()  # tbd which backend to use here; open3d?

# Empty list for trajectory values and/or measurement values
tpoints = []
mpoints = []
callback_counter = 0


# TODO: Look this up and implement ;)
def callback(measurements=None, trajectory=None):  # or: output=None
    global mpoints
    global tpoints
    global callback_counter

    callback_counter += 1

    n = 10

    # Executes in every nth iteration of callback function.
    if callback_counter >= n:

        # Extract trajectory points.
        t = output[1]

        if len(t) != 0:
            tpoints.append(trajectory[-1, :3])

            callback_counter = 0

    if len(measurements) == 0:
        return

    # Add current values to list.
    try:
        mpoints.append([np.hstack(measurements[-1, :3],
                        measurements.hitObjectId[-1])])

    except Exception as err:
        print(err)
        pass


survey.run(on_progress=callback())

# similarly, we can have sth. like:
# survey.run(on_finish=...)


while survey.is_running():
    if len(mpoints) > 0:
        # update points in visualization
        scene_vis.add_sim_points(mpoints)
        scene_vis.add_traj_points(tpoints)

        # + apply colour, refresh gui, etc.
    time.sleep(0.1)

output = survey.output()
