"""Render the actual model's home keyframe; display changes only, no rollout."""
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/hexapod/hexapod.xml'))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
mujoco.mj_forward(model, data)
original_qpos = data.qpos.copy()
model.vis.global_.offwidth = 1800
model.vis.global_.offheight = 1350
model.vis.quality.offsamples = 4
model.vis.headlight.ambient[:] = .45
model.vis.headlight.diffuse[:] = .65
model.vis.headlight.specular[:] = .15
model.geom('floor').matid = -1
model.geom('floor').rgba[:] = [.96, .97, .98, 1]
model.geom('prey').rgba[3] = 0
for i in range(model.ngeom):
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[i])
        model.geom_rgba[i] = [.16, .21, .27, 1] if mesh == 'motor' else [.23, .48, .61, 1]
option = mujoco.MjvOption()
option.geomgroup[1] = 0  # coincident visual copies would cause z-fighting
camera = mujoco.MjvCamera()
camera.lookat[:] = [0, 0, .085]
camera.distance = .83
camera.azimuth = 135
camera.elevation = -33
with mujoco.Renderer(model, height=1350, width=1800) as renderer:
    renderer.update_scene(data, camera=camera, scene_option=option)
    pixels = renderer.render()
    Image.fromarray(pixels).save(Path(__file__).with_name('hexapod_model.png'))

view_specs = [
    ('overhead', 135, -55, .90),
    ('side', 45, -33, .83),
    ('oblique', 225, -33, .83),
]
label_font = ImageFont.truetype(
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf', 30
)
panels = []
with mujoco.Renderer(model, height=700, width=900) as renderer:
    for label, azimuth, elevation, distance in view_specs:
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.distance = distance
        renderer.update_scene(data, camera=camera, scene_option=option)
        panel = Image.fromarray(renderer.render()).convert('RGB')
        labeled = Image.new('RGB', (panel.width, panel.height + 58), 'white')
        labeled.paste(panel, (0, 0))
        draw = ImageDraw.Draw(labeled)
        bounds = draw.textbbox((0, 0), label, font=label_font)
        x = (labeled.width - (bounds[2] - bounds[0])) // 2
        draw.text((x, panel.height + 13), label, font=label_font, fill=(31, 43, 55))
        panels.append(labeled)
platform = Image.new('RGB', (sum(panel.width for panel in panels), panels[0].height), 'white')
for x, panel in zip(range(0, platform.width, panels[0].width), panels):
    platform.paste(panel, (x, 0))
platform.save(Path(__file__).with_name('platform_views.png'), optimize=True)
assert model.nu == 18 and np.array_equal(data.qpos, original_qpos)
assert pixels.shape == (1350, 1800, 3) and pixels.std() > 10
assert platform.size == (2700, 758) and platform.getbbox()
