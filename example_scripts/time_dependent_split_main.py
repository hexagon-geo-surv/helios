import time_dependent_split_functions as tds
import os
import sys
from pathlib import Path
import numpy as np

### Paths to relevant files and directories.
os.chdir("C:/Users/an274/heliospp_alt")
HELIOS_DIR = Path("C:/Users/an274/heliospp_alt")
sys.path.append(str(HELIOS_DIR))
original_scene = HELIOS_DIR / "data" / "scenes" / "demo" / "tds_trees_scene.xml"
output_dir = HELIOS_DIR / "data" / "scenes" / "demo" / "uls_tds_trees"
Path(output_dir).mkdir(parents=True, exist_ok=True)
original_survey = HELIOS_DIR / "data" / "surveys" / "demo" / "tds_trees.xml"
bboxes_clouds = 'output/tds/bboxes/uls_tds_trees'  ##This is currently hardcoded, survey name should be read from xml.
merged_bboxes_las = 'output/tds/merged_bboxes.laz'
interval_clouds = r"C:/Users/an274/heliospp_alt/output/tds/interval_surveys/uls_tds_trees" ##This is currently hardcoded, survey name should be read from xml.
merged_intervals = r"C:/Users/an274/heliospp_alt/output/tds/merged_intervals/"
merged_filtered_intervals = "output/tds/merged_filtered_intervals/"
final_cloud = "output/tds/final_cloud.laz"
import pyhelios


# Create sub scenes with one part each out of the original scene file.
split_scene_files = tds.split_xml(original_scene, output_dir)


# Initialize arrays that store min and max coordinates for each scene part.
mins = np.zeros(shape=(len(split_scene_files), 3))
maxs = np.zeros(shape=(len(split_scene_files), 3))


# Run simulation for each scene part to get bounding box information.
for i, paths in enumerate(split_scene_files):
    # Overwrite original survey for each scene part scene.
    tds.write_survey(original_survey, paths)

    pyhelios.loggingSilent()
    # Build simulation parameters
    simBuilder = pyhelios.SimulationBuilder(
        str(original_survey),
        'C:/Users/an274/helios/assets/',
        'output/'
    )
    simBuilder.setNumThreads(0)
    simBuilder.setRebuildScene(True)
    simBuilder.setLasOutput(True)
    simBuilder.setZipOutput(True)

    simB = simBuilder.build()
    scene = simB.sim.getScene()
    shift = scene.getShift()
    part = scene.getScenePart(0)
    part.computeBound()
    bbox = part.getBound()

    min_coords = [bbox.getMinVertex().getPosition().x + shift.x,
                  bbox.getMinVertex().getPosition().y + shift.y,
                  bbox.getMinVertex().getPosition().z + shift.z]
    max_coords = [bbox.getMaxVertex().getPosition().x + shift.x,
                  bbox.getMaxVertex().getPosition().y + shift.y,
                  bbox.getMaxVertex().getPosition().z + shift.z]

    mins[i] = min_coords
    maxs[i] = max_coords


# Create .objs using the min, max values of the scene parts. Obj paths are stored in list.
objs_outfiles = []
for i in range(len(mins)):
    obj_outfile = tds.create_obj_box(mins[i], maxs[i], f"BBox_{i+1}.obj", output_dir)
    objs_outfiles.append(obj_outfile)


# Writes scenes and surveys with one bbox .obj each.
Bbox_scene_outfiles = Bbox_scene_outfile = tds.write_scene_string(output_dir, objs_outfiles)
survey_outfiles = tds.write_multiple_surveys(original_survey, Bbox_scene_outfiles, output_dir, f"BBox_survey")


# Run simulation for each bbox survey.
for path in survey_outfiles:
    pyhelios.loggingSilent()
    # Build simulation parameters

    simBuilder = pyhelios.SimulationBuilder(
        str(path),
        'C:/Users/an274/heliospp_alt/assets/',
        'output/tds/bboxes'
    )
    simBuilder.setNumThreads(0)
    simBuilder.setRebuildScene(True)
    simBuilder.setLasOutput(True)
    simBuilder.setZipOutput(True)
    simBuilder.setFixedGpsTimeStart("2024-07-07 00:00:00")
    simB = simBuilder.build()

    sim = simBuilder.build()
    sim.start()
    sim.join()



# Merges all legs of all bbox surveys into one las/laz
sub_dirs = []
paths = []
for fil in os.listdir(bboxes_clouds):
        sub_dirs.append(os.path.join(bboxes_clouds,fil))
for i, sub_dir in enumerate(sub_dirs):

        for fil in os.listdir(sub_dir):
            paths.append(os.path.join(sub_dir, fil))
tds.laz_merge(paths, merged_bboxes_las)


# Checks for scene parts in a user defined interval
interval = 30
obj_ids = tds.objs_in_intervall(merged_bboxes_las, interval)


# Writes scenes and surveys for intervals.
intervall_scene_outfiles = tds.gen_intervall_scene(original_scene, output_dir, obj_ids)
intervall_surveys = tds.write_multiple_surveys(original_survey, intervall_scene_outfiles, output_dir, f"Interval_survey")


# Run simulation for each interval survey.
for path in intervall_surveys:
    pyhelios.loggingSilent()
    pyhelios.setDefaultRandomnessGeneratorSeed("123")

    # Build simulation parameters

    simBuilder = pyhelios.SimulationBuilder(
        str(path),
        'C:/Users/an274/heliospp_alt/assets/',
        'output/tds/interval_surveys'
    )
    simBuilder.setNumThreads(0)
    simBuilder.setRebuildScene(True)
    simBuilder.setLasOutput(True)
    simBuilder.setZipOutput(False)
    simBuilder.setFixedGpsTimeStart("2024-07-07 00:00:00")

    sim = simBuilder.build()
    sim.start()
    sim.join()


# Merge legs of interval and write them to separate interval pcs.
sub_dirs = []
for file in os.listdir(interval_clouds):
        sub_dirs.append(os.path.join(interval_clouds,file))

for i, sub_dir in enumerate(sub_dirs):
        paths = []
        for file in os.listdir(sub_dir):
            paths.append(os.path.join(sub_dir, file))
        tds.laz_merge(paths, f"{merged_intervals}merged_intervall_{i+1}.laz")


# Filter merged interval pcs to points inside the interval time.
tds.filter_and_write(merged_intervals,merged_filtered_intervals, interval)


# Merge filtered interval pcs to final pc.
filtered_clouds_path = []
for file in os.listdir(merged_filtered_intervals):
    filtered_clouds_path.append(os.path.join(merged_filtered_intervals,file))
tds.laz_merge(filtered_clouds_path, final_cloud )



