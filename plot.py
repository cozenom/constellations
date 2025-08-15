import numpy as np
from math import radians, tan
from datetime import datetime, timezone

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
from matplotlib.transforms import Bbox

from skyfield.api import load, wgs84, Star
from skyfield.data import hipparcos, stellarium

import json

# --- Stereographic projection centered on zenith ---
def stereographic_from_altaz(alt_deg, az_deg):
    z = np.radians(90.0 - alt_deg)
    r = 2.0 * np.tan(z / 2.0)
    a = np.radians(az_deg)
    x = r * np.cos(a)
    y = r * np.sin(a)
    return x, y

# --- Observer Configuration ---
LAT = 34.0522
LON = -118.2437
ELEV_M = 100

ts = load.timescale()
t = ts.from_datetime(datetime.now(timezone.utc))

eph = load('de421.bsp')
earth = eph['earth']
observer = earth + wgs84.latlon(LAT, LON, elevation_m=ELEV_M)

# --- Planet Positions ---
planets = {
    'Mercury': eph['mercury'],
    'Venus': eph['venus'],
    'Mars': eph['mars'],
    'Jupiter': eph['jupiter barycenter'],
    'Saturn': eph['saturn barycenter'],
    'Uranus': eph['uranus barycenter'],
    'Neptune': eph['neptune barycenter']
}
planet_positions = {}
for name, planet in planets.items():
    astrometric = observer.at(t).observe(planet).apparent()
    alt, az, _ = astrometric.altaz()
    if alt.degrees > 0:
        x, y = stereographic_from_altaz(alt.degrees, az.degrees)
        planet_positions[name] = (x, y)

# --- Star Catalog & Constellations ---
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

with load.open("./constellationship.fab") as f:
    constellations = stellarium.parse_constellations(f)

with load.open("./constellation_abbreviations.json") as f:
    const_names = json.load(f)

star_positions = observer.at(t).observe(Star.from_dataframe(stars)).apparent()
alt, az, _ = star_positions.altaz()
stars['alt_degrees'] = alt.degrees
stars['az_degrees'] = az.degrees
visible = stars['alt_degrees'] > 0.0

# --- Projection & Edge Collection ---
sx, sy = stereographic_from_altaz(stars['alt_degrees'].to_numpy(),
                                  stars['az_degrees'].to_numpy())
stars['x'] = sx
stars['y'] = sy

visible_constellations = []
constellation_edges_xy = []
name_to_xy_for_label = {}

for name, edges in constellations:
    if not edges:
        continue
    hips = set([h1 for (h1, h2) in edges] + [h2 for (h1, h2) in edges])
    vis_any = visible.loc[list(hips)].any()
    if not vis_any:
        continue

    visible_constellations.append(name)

    segs = []
    xs_for_label = []
    ys_for_label = []
    for h1, h2 in edges:
        if (h1 in stars.index) and (h2 in stars.index):
            v1 = visible.loc[h1]
            v2 = visible.loc[h2]
            if v1 and v2:
                x1, y1 = stars.at[h1, 'x'], stars.at[h1, 'y']
                x2, y2 = stars.at[h2, 'x'], stars.at[h2, 'y']
                segs.append([(x1, y1), (x2, y2)])
                xs_for_label += [x1, x2]
                ys_for_label += [y1, y2]
    if segs:
        constellation_edges_xy.extend(segs)
        if xs_for_label:
            name_to_xy_for_label[name] = (np.median(xs_for_label), np.median(ys_for_label))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 12))
fig.patch.set_facecolor('#000000')

# Draw constellation lines
if constellation_edges_xy:
    lc = LineCollection(constellation_edges_xy, colors='#ebcc34', linewidths=0.8)
    ax.add_collection(lc)

# Plot visible stars
limiting_magnitude = 6.0
bright = (stars['alt_degrees'] > 0) & (stars['magnitude'] <= limiting_magnitude)
mag = stars.loc[bright, 'magnitude']
marker_size = (0.3 + limiting_magnitude - mag) ** 2.0
ax.scatter(stars.loc[bright, 'x'], stars.loc[bright, 'y'], s=marker_size, color='#FFFFFF', zorder=3)

# --- Label Overlap Handling ---
used_bboxes = []

def check_overlap(new_bbox):
    for bbox in used_bboxes:
        if bbox.overlaps(new_bbox):
            return True
    return False

# --- Planet Labels ---
planet_marker_sizes = {
    'Mercury': 25, 'Venus': 35, 'Mars': 30,
    'Jupiter': 50, 'Saturn': 45, 'Uranus': 35, 'Neptune': 35
}

for name, (x, y) in planet_positions.items():
    ax.scatter(x, y, color='orange', s=planet_marker_sizes[name], edgecolors='white', zorder=2)
    text = ax.text(x, y + 0.05, name, color='orange', fontsize=5, fontweight='bold', ha='center', va='bottom',
                   path_effects=[path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
    used_bboxes.append(bbox)

# --- Constellation Labels ---
used_bboxes = []

offset_dist = 0.08  # initial offset radius

offsets = [
    (0, 0),
    (0.05, 0),
    (-0.05, 0),
    (0, 0.05),
    (0, -0.05),
    (0.04, 0.04),
    (-0.04, 0.04),
    (0.04, -0.04),
    (-0.04, -0.04)
]

for name, (cx, cy) in name_to_xy_for_label.items():
    vec = np.array([cx, cy])
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec_unit = vec / norm
        cx_base, cy_base = vec + vec_unit * offset_dist
    else:
        cx_base, cy_base = cx, cy  # no offset

    # Try several positions around base point to avoid overlaps
    for dx, dy in offsets:
        cx_off = cx_base + dx
        cy_off = cy_base + dy

        label = ax.text(cx_off, cy_off, const_names[name],
                        fontsize=5, ha='center', va='center',
                        color='#FFFFFF', fontweight='bold',
                        path_effects=[
                            path_effects.Stroke(linewidth=1.5, foreground='black'),
                            path_effects.Normal()
                        ])
        
        # Check for overlap
        renderer = fig.canvas.get_renderer()
        bbox = label.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        overlap = any(bbox.overlaps(prev) for prev in used_bboxes)

        if not overlap:
            used_bboxes.append(bbox)
            break
        else:
            label.remove()



print(used_bboxes)
# --- Circle boundary (horizon) ---
R = 2.0
ax.set_xlim(-R * 1.02, R * 1.02)
ax.set_ylim(-R * 1.02, R * 1.02)
circle = plt.Circle((0, 0), R, transform=ax.transData,
                    facecolor='#010057', alpha=1,
                    edgecolor="#FFFFFF", linewidth=2)
ax.add_artist(circle)

# --- Final touches ---
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
ax.set_title(f'Visible Constellations — {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")}\n'
             f'Lat {LAT:.4f}, Lon {LON:.4f}',
             fontsize=18, color='#FFFFFF', pad=14)

# Cardinal labels
label_offset = 0.04
font_props = dict(color='#ebcc34', fontsize=12, ha='center', va='center', bbox=None)
ax.text(0,  R + label_offset, 'N', **font_props)
ax.text(R + label_offset, 0,  'E', **font_props)
ax.text(0, -R - label_offset, 'S', **font_props)
ax.text(-R - label_offset, 0, 'W', **font_props)

plt.tight_layout()
plt.savefig('visible_constellations.png', dpi=200, bbox_inches='tight')

# --- Console Output ---
visible_constellations = sorted(set(visible_constellations))
print("Visible constellations right now:")
for n in visible_constellations:
    print(" •", n, " ", const_names[n])
