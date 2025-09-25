import json
import os
from datetime import datetime, timezone

import matplotlib.patheffects as path_effects
import numpy as np
from geopy.geocoders import Nominatim
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos, stellarium


# --- Stereographic projection (zenith-centered) ---
def stereographic_from_altaz(alt_deg, az_deg):
    z = np.radians(90.0 - alt_deg)
    r = 2.0 * np.tan(z / 2.0)
    a = np.radians(az_deg)
    x = r * np.sin(a)
    y = r * np.cos(a)
    return x, y


# --- Get Place Coordinates ---
def getloc(name: str):
    geolocator = Nominatim(user_agent="locfinder")
    location = geolocator.geocode(name)

    if location:
        return (
            location.point.longitude,
            location.point.latitude,
            location.point.altitude,
        )
    else:
        return -1


# --- Get Antipodal (opposite side) Coordinates ---
def get_antipodal(lat, long):
    return -lat, (long + 180) if long < 0 else (long - 180)


def get_astronomical_data(place, time, plot_visible=True):
    # --- Location setup ---
    LAT, LON, ELEV_M = getloc(place)
    original_coords = (LAT, LON)

    if not plot_visible:
        LAT, LON = get_antipodal(LAT, LON)
    ELEV_M = 0  # Set elevation to 0 for consistent behavior

    # --- Time and observer setup ---
    ts = load.timescale()
    t = ts.from_datetime(time or datetime.now(timezone.utc))

    eph = load("de421.bsp")
    earth = eph["earth"]
    observer = earth + wgs84.latlon(LAT, LON, elevation_m=ELEV_M)

    # --- Get planet positions (above horizon) ---
    planets = {
        "Moon": eph["moon"],
        "Mercury": eph["mercury"],
        "Venus": eph["venus"],
        "Mars": eph["mars"],
        "Jupiter": eph["jupiter barycenter"],
        "Saturn": eph["saturn barycenter"],
        "Uranus": eph["uranus barycenter"],
        "Neptune": eph["neptune barycenter"],
    }
    planet_positions = {}
    for name, planet in planets.items():
        astrometric = observer.at(t).observe(planet).apparent()
        alt, az, _ = astrometric.altaz()
        # Only include planets above the horizon
        if alt.degrees > 0:
            x, y = stereographic_from_altaz(alt.degrees, az.degrees)
            planet_positions[name] = (x, y)

    # --- Load star catalog and constellations ---
    with load.open(hipparcos.URL) as f:
        stars = hipparcos.load_dataframe(f)

    with load.open("./constellationship.fab") as f:
        constellations = stellarium.parse_constellations(f)

    with load.open("./constellation_abbreviations.json") as f:
        const_names = json.load(f)

    # --- Compute star alt/az ---
    star_positions = observer.at(t).observe(Star.from_dataframe(stars)).apparent()
    alt, az, _ = star_positions.altaz()
    stars["alt_degrees"] = alt.degrees
    stars["az_degrees"] = az.degrees
    visible = stars["alt_degrees"] > 0.0

    # --- Project stars to XY ---
    sx, sy = stereographic_from_altaz(
        stars["alt_degrees"].to_numpy(), stars["az_degrees"].to_numpy()
    )
    stars["x"] = sx
    stars["y"] = sy

    # --- Filter visible constellations and prepare edges ---
    visible_constellations = []
    constellation_edges_xy = []
    name_to_xy_for_label = {}
    constellation_star_ids = set()  # Track stars that are part of constellations

    for name, edges in constellations:
        if not edges:
            continue

        # Get all star IDs in this constellation
        hips = set([h1 for (h1, h2) in edges] + [h2 for (h1, h2) in edges])

        # Skip constellation if no stars are visible
        if not visible.loc[list(hips)].any():
            continue

        visible_constellations.append(name)
        segs = []
        xs_for_label = []
        ys_for_label = []

        # Build line segments for constellation lines
        for h1, h2 in edges:
            if h1 in stars.index and h2 in stars.index:
                if visible.loc[h1] and visible.loc[h2]:
                    x1, y1 = stars.at[h1, "x"], stars.at[h1, "y"]
                    x2, y2 = stars.at[h2, "x"], stars.at[h2, "y"]
                    segs.append([(x1, y1), (x2, y2)])
                    xs_for_label.extend([x1, x2])
                    ys_for_label.extend([y1, y2])
                    # Add these stars to constellation star set
                    constellation_star_ids.add(h1)
                    constellation_star_ids.add(h2)

        if segs:
            constellation_edges_xy.extend(segs)
            # Calculate label position as median of constellation line coordinates
            if xs_for_label:
                name_to_xy_for_label[name] = (
                    np.median(xs_for_label),
                    np.median(ys_for_label),
                )

    # --- Return all data needed for plotting ---
    return {
        "stars": stars,
        "planet_positions": planet_positions,
        "constellation_edges_xy": constellation_edges_xy,
        "name_to_xy_for_label": name_to_xy_for_label,
        "const_names": const_names,
        "visible_constellations": visible_constellations,
        "constellation_star_ids": constellation_star_ids,
        "coords": (LAT, LON),
        "original_coords": original_coords,
        "time": t,
    }


# --- Plot Sky on Single Axis ---
def plot_sky_on_axis(ax, data, limiting_magnitude, title_suffix=""):
    # --- Extract data ---
    stars = data["stars"]
    planet_positions = data["planet_positions"]
    constellation_edges_xy = data["constellation_edges_xy"]
    name_to_xy_for_label = data["name_to_xy_for_label"]
    const_names = data["const_names"]
    visible_constellations = data["visible_constellations"]
    constellation_star_ids = data["constellation_star_ids"]

    # --- Draw constellation lines ---
    if constellation_edges_xy:
        lc = LineCollection(constellation_edges_xy, colors="#ebcc34", linewidths=0.6)
        ax.add_collection(lc)

    # --- Plot visible stars with size ~ brightness ---
    bright = (stars["alt_degrees"] > 0) & (stars["magnitude"] <= limiting_magnitude)
    mag = stars.loc[bright, "magnitude"]
    # Calculate marker size based on magnitude (brighter = larger)
    marker_size = (0.3 + limiting_magnitude - mag) ** 2.0

    # Separate constellation stars from regular stars
    bright_indices = stars.loc[bright].index
    constellation_stars = bright_indices.intersection(constellation_star_ids)
    regular_stars = bright_indices.difference(constellation_star_ids)

    # Plot regular stars first (dimmer, so they appear behind constellation stars)
    if len(regular_stars) > 0:
        regular_mag = stars.loc[regular_stars, "magnitude"]
        regular_size = (0.3 + limiting_magnitude - regular_mag) ** 1.5
        ax.scatter(
            stars.loc[regular_stars, "x"],
            stars.loc[regular_stars, "y"],
            s=regular_size,
            color="#CCCCCC",  # Dimmer gray for regular stars
            zorder=3,
        )

    # Plot constellation stars on top (brighter, more prominent)
    if len(constellation_stars) > 0:
        const_mag = stars.loc[constellation_stars, "magnitude"]
        const_size = (0.3 + limiting_magnitude - const_mag) ** 1.5
        ax.scatter(
            stars.loc[constellation_stars, "x"],
            stars.loc[constellation_stars, "y"],
            s=const_size,
            color="#FFFFFF",  # Bright white for constellation stars
            zorder=4,
        )

    # --- Plot planets and labels ---
    planet_marker_sizes = {
        "Mercury": 25,
        "Venus": 35,
        "Mars": 30,
        "Jupiter": 50,
        "Saturn": 45,
        "Uranus": 35,
        "Neptune": 35,
        "Moon": 80,
    }

    for name, (x, y) in planet_positions.items():
        color = "orange"
        if name == "Moon":
            color = "#FFFFCC"

        # Draw planet marker
        ax.scatter(
            x,
            y,
            color=color,
            s=planet_marker_sizes[name],
            edgecolors="white",
            zorder=5,
        )
        # Add planet label with outline for visibility
        ax.text(
            x,
            y + 0.05,
            name,
            color=color,
            fontsize=5,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=6,
            path_effects=[
                path_effects.Stroke(linewidth=1.5, foreground="black"),
                path_effects.Normal(),
            ],
        )

    # --- Plot constellation labels with overlap detection ---
    used_positions = []  # Track label positions to avoid overlaps

    def too_close(new_x, new_y, min_distance=0.15):
        return any(
            np.sqrt((new_x - x) ** 2 + (new_y - y) ** 2) < min_distance
            for x, y in used_positions
        )

    offset_dist = 0.08
    # Try different offset positions to avoid overlaps
    offsets = [
        (0, 0),
        (0.05, 0),
        (-0.05, 0),
        (0, 0.05),
        (0, -0.05),
        (0.04, 0.04),
        (-0.04, 0.04),
        (0.04, -0.04),
        (-0.04, -0.04),
    ]

    for name, (cx, cy) in name_to_xy_for_label.items():
        # Move label slightly away from constellation center
        vec = np.array([cx, cy])
        norm = np.linalg.norm(vec)
        if norm > 0:
            base_x, base_y = vec + (vec / norm) * offset_dist
        else:
            base_x, base_y = cx, cy

        # Try to place label without overlapping existing ones
        for dx, dy in offsets:
            cx_off = base_x + dx
            cy_off = base_y + dy

            if not too_close(cx_off, cy_off):
                ax.text(
                    cx_off,
                    cy_off,
                    const_names[name],
                    fontsize=5,
                    ha="center",
                    va="center",
                    color="#FFFFFF",
                    fontweight="bold",
                    zorder=7,
                    path_effects=[
                        path_effects.Stroke(linewidth=1.5, foreground="black"),
                        path_effects.Normal(),
                    ],
                )
                used_positions.append((cx_off, cy_off))
                break

        # Cant place = skip

    # --- Draw horizon circle ---
    R = 2.0
    ax.set_xlim(-R * 1.02, R * 1.02)
    ax.set_ylim(-R * 1.02, R * 1.02)
    circle = plt.Circle(
        (0, 0),
        R,
        transform=ax.transData,
        facecolor="#010057",
        alpha=1,
        edgecolor="#FFFFFF",
        linewidth=2,
    )
    ax.add_artist(circle)

    # --- Final formatting ---
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Add subplot title
    ax.text(
        0,
        R + 0.15,
        title_suffix,
        ha="center",
        va="bottom",
        fontsize=14,
        color="#FFFFFF",
        transform=ax.transData,
        fontweight="bold",
    )

    # --- Add cardinal direction labels ---
    label_offset = 0.04
    font_props = dict(color="#ebcc34", fontsize=12, ha="center", va="center")
    # For antipodal view, flip the directions to match original location's reference frame
    if title_suffix == "Not Visible":
        ax.text(0, R + label_offset, "N", **font_props)  # Their N is our S
        ax.text(R + label_offset, 0, "W", **font_props)  # Their E is our W
        ax.text(0, -R - label_offset, "S", **font_props)  # Their S is our N
        ax.text(-R - label_offset, 0, "E", **font_props)  # Their W is our E
    else:
        # Normal directions for visible side
        ax.text(0, R + label_offset, "N", **font_props)
        ax.text(R + label_offset, 0, "E", **font_props)
        ax.text(0, -R - label_offset, "S", **font_props)
        ax.text(-R - label_offset, 0, "W", **font_props)

    # --- Return constellation data ---
    return {i: const_names[i] for i in sorted(set(visible_constellations))}


# --- Main Plotting Function ---
def plot_constellations(
    place="Los Angeles",
    limiting_magnitude=6.0,
    time=None,
    fname=None,
    plot_visible=True,
):
    # --- Side-by-side case ---
    if plot_visible == "both":
        # Get astronomical data for both views
        visible_data = get_astronomical_data(place, time, plot_visible=True)
        not_visible_data = get_astronomical_data(place, time, plot_visible=False)

        # Create side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        fig.patch.set_facecolor("#000000")

        # Plot both views
        visible_constellations = plot_sky_on_axis(
            ax1, visible_data, limiting_magnitude, "Visible"
        )
        not_visible_constellations = plot_sky_on_axis(
            ax2, not_visible_data, limiting_magnitude, "Not Visible"
        )

        # Add main title
        LAT, LON = visible_data["original_coords"]
        current_time = time or datetime.now(timezone.utc)
        main_title = f'Sky View from {place} — {current_time.astimezone().strftime("%Y-%m-%d %H:%M %Z")}\nLat {LAT:.4f}, Lon {LON:.4f}'
        fig.suptitle(main_title, fontsize=20, color="#FFFFFF", y=0.95)

        plt.tight_layout()

        # Save the image
        filename = fname if fname else f"constellations_both_sides_{place}.png"
        print(filename)
        os.makedirs("./images", exist_ok=True)
        plt.savefig(f"./images/{filename}", dpi=200, bbox_inches="tight")

        return visible_constellations, not_visible_constellations

    # --- Single view case ---
    # Get astronomical data
    data = get_astronomical_data(place, time, plot_visible)

    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor("#000000")

    # Plot the view
    visible_constellations = plot_sky_on_axis(ax, data, limiting_magnitude)

    # Add title
    LAT, LON = data["coords"]
    current_time = time or datetime.now(timezone.utc)
    title = f'{"Visible" if plot_visible else "Not Visible"} Constellations {place} — {current_time.astimezone().strftime("%Y-%m-%d %H:%M %Z")}\nLat {LAT:.4f}, Lon {LON:.4f}'
    ax.set_title(title, fontsize=18, color="#FFFFFF", pad=14)

    plt.tight_layout()

    # Save the image
    filename = (
        fname
        if fname
        else f"{'visible' if plot_visible else 'not_visible'}_constellations_{place}.png"
    )
    print(filename)
    os.makedirs("./images", exist_ok=True)
    plt.savefig(f"./images/{filename}", dpi=200, bbox_inches="tight")

    return visible_constellations
