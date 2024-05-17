import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
HELIOS_DIR = Path(__file__).parent.parent
sys.path.append(str(HELIOS_DIR))
import pyhelios
# import xml_splitter_v2 as xml
# import survey_writer as sw


def write_survey(template_path, paths):
    template = ET.parse(template_path)
    template_root = template.getroot()

    scene_path = template_root.find('.//survey[@scene]')
    scene_path.attrib["scene"] = str(paths) + "#scene"

    output_filename = f'{template_path}'
    template.write(output_filename, xml_declaration= True)


def split_xml(file, output_dir):
    """Split an XML file into multiple files, one for each part."""

    xml_start = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scene id="scene" name="scene">
"""

    xml_end = """
    </scene>
</document>
"""
    # Parse the XML file
    tree = ET.parse(file)
    root = tree.getroot()
    scene = root.find("scene")
    parts = scene.findall("part")

    outfiles = []
    # Split the XML file into multiple files
    for i, part in enumerate(parts):
        part_fp = part.find(".//param[@key='filepath']").get("value")
        part_name = Path(part_fp).stem
        outfile = Path(output_dir)  / f"{part_name}.xml"
        with open(outfile, "w") as f:
            f.write(xml_start)
            f.write("        " + ET.tostring(part).decode("utf-8"))
            f.write(xml_end)
        outfiles.append(outfile)

    return outfiles, i

working_dir = os.getcwd()
print(working_dir)

file = HELIOS_DIR / "data" / "scenes" / "demo" / "block_ext.xml"
output_dir = HELIOS_DIR / "data" / "scenes" / "demo" / "block_split"
Path(output_dir).mkdir(parents=True, exist_ok=True)
template_path = HELIOS_DIR / "data" / "surveys" / "demo" / "block_survey_empty.xml"


split_scene_files, i = split_xml(file, output_dir)
mins = np.zeros(shape=(len(split_scene_files), 3))
maxs = np.zeros(shape=(len(split_scene_files), 3))
centres = np.zeros(shape=(len(split_scene_files), 3))
for i, paths in enumerate(split_scene_files):
    write_survey(template_path, paths)

    pyhelios.loggingQuiet()
    # Build simulation parameters
    simBuilder = pyhelios.SimulationBuilder(
        str(template_path),
        'assets/',
        'output/'
    )
    simBuilder.setNumThreads(0)
    simBuilder.setLasOutput(True)
    simBuilder.setZipOutput(True)
    simB = simBuilder.build()
    scene = simB.sim.getScene()
    shift = scene.getShift()
    part = scene.getScenePart(0)

    bbp = part.computeBound()
    bbox = part.getBound()
    min = [bbox.getMinVertex().getPosition().x + shift.x, 
           bbox.getMinVertex().getPosition().y + shift.y, 
           bbox.getMinVertex().getPosition().z + shift.z]
    max = [bbox.getMaxVertex().getPosition().x + shift.x, 
           bbox.getMaxVertex().getPosition().y + shift.y, 
           bbox.getMaxVertex().getPosition().z + shift.z]
    bbox_centre = np.mean([min, max], axis=0)
    mins[i] = min
    maxs[i] = max
    centres[i] = bbox_centre


scene_bbox_min = np.min(mins, axis=0)
scene_bbox_max = np.max(maxs, axis=0)

# scene shift is derived via bbox centre
bbox_centre = np.mean(np.vstack((scene_bbox_min, scene_bbox_max)), axis=0)
print(bbox_centre)

# apply scene shift to all scene part bboxes
mins_shifted = mins - bbox_centre
maxs_shifted = maxs - bbox_centre

# approximate scene parts with single regular detailed voxel
# get centroid and longest axis of each (shifted) scene part bbox
# set min corner to centroid - 0.5 * longest axis
# set res to longest axis of scene part bbox
# find out how to set the transmittance




