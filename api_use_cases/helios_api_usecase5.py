"""
We want to run several simulations over a scene with the same survey settings,
but changes in the scene (e.g. coordinate transformations of scene parts or even replacing scene parts)
"""

import helios

survey_xml = "some_file.xml"
survey = helios.Survey(survey_file=survey_xml)
scene = survey.get_scene()  # get scene object from survey (as survey is linked to scene, platform, scanner, etc.)
scene.show(annotate_id=True)  # this option allows annotating each scenepart in the 3D view with its ID
sname = survey.name

# 10 epochs
# each epoch, one scenepart moves a certain distance
# in epoch 3, a new scenepart is added
# in epoch 4, another scenepart is rotated
# in epoch 5, another scenepart disappears
# in epoch 7, the material properties of one object are modified
epochs = 10
for e in range(epochs):
    # sceneparts can be retrieved by ID (as specified in the XML or (if not specified) corresponding to the order in the XML)
    sp_car = scene.sceneparts[3]
    sp_car.transform(translation=[0.5, 1.0])
    # alternatively
    # sp_car.x += 0.5
    # sp_car.y += 1.
    if e == 3:
        new_sp = helios.Scenepart.from_file("some.obj")  # or: helios.Scenepart.load(open3d_mesh) / .from_o3d(open3d_mesh), etc.
        scene.add_sp(new_sp)
    if e == 4:
        sp_car2 = scene.sceneparts[4]
        sp_car2.rotate(axis="z", angle="180",  # car now parks the other way around :)
                       origin=sp_car2.origin)  # origin as an array, i.e., here: object origin
        # may also be a custom point or `sp_car2.bbox_center`, `sp_car2.cog`, or `scene.origin`
        # TODO: Feature request: support for other notations for rotations as well (matrix, euler angles)
    if e == 5:
        scene.sceneparts.pop(1)  # or: scene.sceneparts.remove(1)  # where 1 is the index of the scenepart
    if e == 7:
        sp_tree = scene.sceneparts[2]
        mat = sp_tree.material.get_by_name("leaves")  # returns a reference of the material
        print(mat)  # print material definition, either just like in .mat file or in JSON format or so
        mat.helios_reflectance = 0.1
        mat.kd = [0.05, 0.4, 0.05]
        sp_tree.material = mat  # set the material again after making the changes
    survey.name = sname + f"epoch_{e}"
    # maybe some command needed to update the scene/survey accordingly?
    survey.run(format="laz",
               output="output/",
               threads=0)
