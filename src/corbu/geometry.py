import math
import tqdm
import random
import pandas as pd
import rhino3dm as r3d
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

TOL = 1e-6
random.seed(42)

def brep_to_rect_outline_points(brep, tol=1e-6):
    """Convert a rectangular planar Brep to 5 ordered corner points (closed)."""
    # Collect vertex locations
    verts = [v.Location for v in brep.Vertices]

    # Deduplicate by position (just in case)
    unique = []
    for p in verts:
        if not any(
            abs(p.X - q.X) < tol and
            abs(p.Y - q.Y) < tol and
            abs(p.Z - q.Z) < tol
            for q in unique
        ):
            unique.append(p)

    if len(unique) != 4:
        # Fallback: just return them as-is, closed
        if unique and (
            unique[0].X != unique[-1].X or unique[0].Y != unique[-1].Y \
                or unique[0].Z != unique[-1].Z
            ):
            unique.append(unique[0])
        return unique

    # Sort by Y then X: bottom row first, then top row
    pts_sorted = sorted(unique, key=lambda p: (p.Y, p.X))
    b0, b1, t0, t1 = pts_sorted

    # Ensure left/right order in each row
    if b1.X < b0.X:
        b0, b1 = b1, b0
    if t1.X < t0.X:
        t0, t1 = t1, t0

    ordered = [b0, b1, t1, t0, b0]
    return ordered

def plot_rhino_geometry(
    points=None,
    polylines=None,
    curves=None,
    surfaces=None,
    dim=None,
    show_axes=True,
    colors=None,
    fill_alpha=0.3,
):

    points = points or []
    polylines = polylines or []
    curves = curves or []
    surfaces = surfaces or []

    # Default colors per type
    default_colors = {
        "points": "tab:orange",
        "polylines": "tab:blue",
        "curves": "tab:green",
        "surfaces": "tab:blue",
    }
    if colors is None:
        colors = default_colors
    else:
        merged = default_colors.copy()
        merged.update(colors)
        colors = merged

    # Decide 2D vs 3D if not forced
    if dim is None:
        has_z = False

        for pl in polylines:
            for pt in pl:
                if getattr(pt, "Z", 0.0) != 0.0:
                    has_z = True
                    break
            if has_z:
                break

        if not has_z:
            for pt in points:
                if getattr(pt, "Z", 0.0) != 0.0:
                    has_z = True
                    break

        if not has_z:
            for brep in surfaces:
                for v in brep.Vertices:
                    p = v.Location
                    if getattr(p, "Z", 0.0) != 0.0:
                        has_z = True
                        break
                if has_z:
                    break

        dim = 3 if has_z else 2

    # Create axes
    fig = plt.figure()
    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])
    else:
        ax = fig.add_subplot(111)

    # We'll collect all coordinates for equal 3D scaling
    all_x, all_y, all_z = [], [], []

    # Plot helpers
    def plot_polyline(pl):
        xs = [p.X for p in pl]
        ys = [p.Y for p in pl]
        zs = [getattr(p, "Z", 0.0) for p in pl]

        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)

        if dim == 3:
            ax.plot(xs, ys, zs, linewidth=1, color=colors["polylines"])
        else:
            ax.plot(xs, ys, linewidth=1, color=colors["polylines"])

    def plot_points(pts):
        xs = [p.X for p in pts]
        ys = [p.Y for p in pts]
        zs = [getattr(p, "Z", 0.0) for p in pts]

        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)

        if dim == 3:
            ax.scatter(xs, ys, zs, s=5, color=colors["points"])
        else:
            ax.scatter(xs, ys, s=5, color=colors["points"])

    def plot_curve(cv, samples=50):
        dom = cv.Domain
        t0, t1 = dom.T0, dom.T1

        ts = [t0 + (t1 - t0) * i / (samples - 1) for i in range(samples)]
        pts = [cv.PointAt(t) for t in ts]

        xs = [p.X for p in pts]
        ys = [p.Y for p in pts]
        zs = [getattr(p, "Z", 0.0) for p in pts]

        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)

        if dim == 3:
            ax.plot(xs, ys, zs, linestyle="--", linewidth=1.5, color=colors["curves"])
        else:
            ax.plot(xs, ys, linestyle="--", linewidth=1.5, color=colors["curves"])

    def plot_surface(brep):
        outline_pts = brep_to_rect_outline_points(brep)
        xs = [p.X for p in outline_pts]
        ys = [p.Y for p in outline_pts]
        zs = [p.Z for p in outline_pts]

        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)

        if dim == 3:
            # Filled 3D polygon
            # drop the last repeated point for the polygon vertices
            verts = list(zip(xs[:-1], ys[:-1], zs[:-1]))
            poly3d = Poly3DCollection(
                [verts],
                facecolors=colors["surfaces"],
                edgecolors=colors["polylines"],
                alpha=fill_alpha,
            )
            ax.add_collection3d(poly3d)
        else:
            # 2D filled polygon as before
            poly = Polygon(
                list(zip(xs, ys)),
                closed=True,
                facecolor=colors["surfaces"],
                edgecolor=colors["polylines"],
                alpha=fill_alpha,
            )
            ax.add_patch(poly)

    # Plot order: surfaces first, then polylines/curves/points
    for brep in surfaces:
        plot_surface(brep)

    for pl in polylines:
        plot_polyline(pl)

    for cv in curves:
        plot_curve(cv)

    if points:
        plot_points(points)

    if not show_axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if dim == 3:
            ax.set_zlabel("Z")

    if dim == 2:
        ax.set_aspect("equal", adjustable="box")
    else:
        # Equal scaling in 3D
        if all_x and all_y and all_z:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            z_min, z_max = min(all_z), max(all_z)

            x_mid = 0.5 * (x_min + x_max)
            y_mid = 0.5 * (y_min + y_max)
            z_mid = 0.5 * (z_min + z_max)

            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            if max_range == 0:
                max_range = 1.0  # avoid zero-size

            half = 0.5 * max_range

            ax.set_xlim(x_mid - half, x_mid + half)
            ax.set_ylim(y_mid - half, y_mid + half)
            ax.set_zlim(z_mid - half, z_mid + half)

    plt.tight_layout()
    plt.show()

def cells_to_planar_breps(cells):
    """
    cells: list[r3d.Polyline] in the WorldXY plane (z = 0)
    returns: list[r3d.Brep]
    """
    breps = []
    plane = r3d.Plane.WorldXY()

    for pl in cells:
        # Convert polyline to curve
        crv = r3d.PolylineCurve(pl)

        # Create a planar Brep trimmed by that curve
        brep = r3d.Brep.CreateTrimmedPlane(plane, crv)
        if brep is not None:
            breps.append(brep)

    return breps

def round_half_down(x: float) -> float:
    """Round to nearest integer; if exactly .5, go DOWN."""
    floor_x = math.floor(x)
    frac = x - floor_x
    if frac > 0.5:
        return math.ceil(x)
    else:
        # frac < 0.5 or exactly 0.5 → go down
        return floor_x

def snap_to_3_half_down(v: float) -> float:
    """Snap to nearest multiple of 3; ties go to the lower multiple."""
    return round_half_down(v / 3.0) * 3.0

def get_core_locations(params, plot=False):
    
    # Calculate number of cores so that the maximum distance in x between
    # the core centre and the edge of the building is lower than the max
    # stair distance
    num_cores = int(
        math.ceil(params["building_length"] / (2*params["max_stair_dist"]))
    )

    # Calculate the position of the core(s) in x and y
    raw_core_pts = []
    for i in range(num_cores):

        x = (i+1) * params["building_length"] / num_cores \
                - (params["building_length"]/num_cores)*0.5 \
                    - params["core_length"]/2
    
        y = (params["building_width"] - params["core_width"]) \
            * params["core_y_location"] + params["core_width"] / 2 \
                - params["core_width"]/2
        
        raw_core_pts.append(
            r3d.Point2d(
                snap_to_3_half_down(x),
                snap_to_3_half_down(y),
            )
        )
    
    core_pls = []
    for pt in raw_core_pts:
        pt_list = r3d.Point3dList()
        pt_list.Add(pt.X, pt.Y, 0)
        pt_list.Add(pt.X+params["core_length"], pt.Y, 0)
        pt_list.Add(pt.X+params["core_length"],pt.Y+params["core_width"], 0)
        pt_list.Add(pt.X,pt.Y+params["core_width"], 0)
        pt_list.Add(pt.X, pt.Y, 0) # close polyline
        core_pls.append(r3d.Polyline.CreateFromPoints(pt_list))
    
    if plot:
        plot_rhino_geometry(
            polylines=core_pls,
            dim=2,
            show_axes=True
        )
    
    return core_pls

def rect_bounds_from_polyline(pl):
    xs = [p.X for p in pl]
    ys = [p.Y for p in pl]
    return min(xs), max(xs), min(ys), max(ys)

def point_in_rect(x, y, rect, inclusive=True):
    minx, maxx, miny, maxy = rect
    if inclusive:
        return (minx <= x <= maxx) and (miny <= y <= maxy)
    else:
        return (minx < x < maxx) and (miny < y < maxy)

def build_grid_lines(outer_pl, inner_pls):
    outer_minx, outer_maxx, outer_miny, outer_maxy = rect_bounds_from_polyline(
        outer_pl
    )

    xs = [outer_minx, outer_maxx]
    ys = [outer_miny, outer_maxy]

    for pl in inner_pls:
        minx, maxx, miny, maxy = rect_bounds_from_polyline(pl)
        xs.extend([minx, maxx])
        ys.extend([miny, maxy])

    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return xs, ys

def create_base_surfaces(params, core_pls, plot=False):
    # Make a polyline for the building outline
    pt_list = r3d.Point3dList()
    pt_list.Add(0, 0, 0)
    pt_list.Add(params["building_length"], 0, 0)
    pt_list.Add(params["building_length"], params["building_width"], 0)
    pt_list.Add(0, params["building_width"], 0)
    pt_list.Add(0, 0, 0)  # close polyline
    building_pl = r3d.Polyline.CreateFromPoints(pt_list)

    # Precompute rectangles as (minx, maxx, miny, maxy)
    outer_rect = rect_bounds_from_polyline(building_pl)
    outer_minx, outer_maxx, outer_miny, outer_maxy = outer_rect

    inner_rects = [rect_bounds_from_polyline(pl) for pl in core_pls]

    # X breakpoints from outer + all cores
    xs = [outer_minx, outer_maxx]
    for (minx, maxx, _, _) in inner_rects:
        xs.extend([minx, maxx])
    xs = sorted(set(xs))

    cells = []

    # Loop over X slices
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x1 == x0:
            continue

        # Cores that actually intersect this X interval
        relevant_rects = []
        for r in inner_rects:
            minx, maxx, miny, maxy = r
            # overlap in X? (strict or inclusive, small tolerance)
            if maxx > x0 and minx < x1:
                relevant_rects.append(r)

        # Y breakpoints for THIS slice only:
        y_lines = [outer_miny, outer_maxy]
        for (_, _, miny, maxy) in relevant_rects:
            y_lines.extend([miny, maxy])
        y_lines = sorted(set(y_lines))

        # If no cores intersect this slice, just get one interval
        for j in range(len(y_lines) - 1):
            y0, y1 = y_lines[j], y_lines[j + 1]
            if y1 == y0:
                continue

            # Center point of this candidate cell
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)

            # Must be inside outer
            if not point_in_rect(cx, cy, outer_rect, inclusive=True):
                continue

            # Must NOT be inside any core
            inside_inner = any(
                point_in_rect(cx, cy, r, inclusive=False) for r in inner_rects
            )
            if inside_inner:
                continue

            # Valid cell -> rectangular polyline
            pts = [
                r3d.Point3d(x0, y0, 0.0),
                r3d.Point3d(x1, y0, 0.0),
                r3d.Point3d(x1, y1, 0.0),
                r3d.Point3d(x0, y1, 0.0),
                r3d.Point3d(x0, y0, 0.0),  # close
            ]
            cell_pl = r3d.Polyline(pts)
            cells.append(cell_pl)

    # Turn cells into planar Breps (surfaces)
    cell_surfs = cells_to_planar_breps(cells)

    # Optional plotting
    if plot:
        outline_pls = [building_pl] + core_pls
        plot_rhino_geometry(
            polylines=outline_pls,   # building + cores
            surfaces=cell_surfs,     # base strips
            dim=2,
            colors={
                "polylines": "black",
                "surfaces": "tab:blue",
            },
            fill_alpha=0.4,
        )

    return cells, cell_surfs

def _brep_rect_bounds(brep):
    """
    Get axis-aligned rectangular bounds (minx, maxx, miny, maxy)
    from a planar rectangular Brep.
    """
    xs = []
    ys = []
    for v in brep.Vertices:
        p = v.Location
        xs.append(p.X)
        ys.append(p.Y)
    return min(xs), max(xs), min(ys), max(ys)

def _axis_positions(min_val, max_val, spacing):
    """
    Create positions along one axis between min_val and max_val (inclusive),
    using `spacing` as the MAX bay size.

    Example: length 9, spacing 6
        -> positions [min, min+6, max]  (segments: 6, 3)
    """
    if max_val <= min_val:
        return [min_val]  # degenerate, just in case

    positions = [min_val]
    current = min_val

    # keep stepping by spacing while we can still fit one more full step
    # before hitting or crossing max_val
    while current + spacing < max_val - 1e-9:  # small tolerance
        current += spacing
        positions.append(current)

    # always include the end
    if abs(positions[-1] - max_val) > 1e-9:
        positions.append(max_val)

    return positions

def generate_column_locations(input_params, cell_surfs, plot=False):
    # spacings
    x_spacing = input_params["column_spacing_x"]
    y_spacing = input_params["column_spacing_y"]

    column_locations = []  # [[pts for first cell], [pts for second cell], ...]

    for cell in cell_surfs:
        # rectangular bounds of this cell
        minx, maxx, miny, maxy = _brep_rect_bounds(cell)

        # positions along x and y
        xs = _axis_positions(minx, maxx, x_spacing)
        ys = _axis_positions(miny, maxy, y_spacing)

        # grid of points for this cell
        pts = []
        for x in xs:
            for y in ys:
                pts.append(r3d.Point3d(x, y, 0.0))

        column_locations.append(pts)

    if plot:
        # flatten all points for plotting
        all_pts = [p for cell_pts in column_locations for p in cell_pts]
        plot_rhino_geometry(
            points=all_pts,
            surfaces=cell_surfs,
            dim=2,
            show_axes=True,
            colors={
                "points": "red",
                "surfaces": "tab:blue",
            },
            fill_alpha=0.3,
        )

    return column_locations

def generate_floors(column_locations, plot=True):
    """
    column_locations: list of lists of Point3d
        shape: [[pts for first cell], [pts for second cell], ...]

    Returns:
        floor_locations: list of lists of Breps
        shape: [[brep1, brep2, ... in cell 1], [breps in cell 2], ...]
    """
    floor_locations = []

    for cell_cols in column_locations:
        # If fewer than 4 points, you can't form any panel
        if len(cell_cols) < 4:
            floor_locations.append([])
            continue

        # Unique sorted X and Y coordinates for this cell
        xs = sorted({p.X for p in cell_cols})
        ys = sorted({p.Y for p in cell_cols})

        # If you don't have at least 2 in each direction, no quads
        if len(xs) < 2 or len(ys) < 2:
            floor_locations.append([])
            continue

        # Map (x, y) -> Point3d for quick lookup
        point_map = {(p.X, p.Y): p for p in cell_cols}

        cell_floors = []

        # Loop over each quad in the grid
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = ys[j], ys[j + 1]

                # Get the four corner points; skip if any is missing
                key00 = (x0, y0)
                key10 = (x1, y0)
                key11 = (x1, y1)
                key01 = (x0, y1)

                if not (key00 in point_map and key10 in point_map and
                        key11 in point_map and key01 in point_map):
                    continue

                p00 = point_map[key00]
                p10 = point_map[key10]
                p11 = point_map[key11]
                p01 = point_map[key01]

                # Build a rectangular polyline and then a planar Brep
                pt_list = r3d.Point3dList()
                for p in (p00, p10, p11, p01, p00):
                    pt_list.Add(p.X, p.Y, p.Z)

                pl = r3d.Polyline.CreateFromPoints(pt_list)
                crv = r3d.PolylineCurve(pl)

                brep = r3d.Brep.CreateTrimmedPlane(r3d.Plane.WorldXY(), crv)
                if brep is not None:
                    cell_floors.append(brep)

        floor_locations.append(cell_floors)

    if plot:
        # Flatten all columns and all floors for plotting
        all_pts = [p for cell in column_locations for p in cell]
        all_floors = [b for cell in floor_locations for b in cell]

        plot_rhino_geometry(
            points=all_pts,
            surfaces=all_floors,
            dim=2,
            show_axes=True,
            colors={
                "points": "red",
                "surfaces": "tab:blue",
            },
            fill_alpha=0.3,
        )

    return floor_locations

def _core_segments_from_polylines(core_pls):
    """Extract axis-aligned line segments from core polylines."""
    segments = []
    for pl in core_pls:
        # assume closed, last==first; we go to len(pl)-1
        for i in range(len(pl) - 1):
            p0 = pl[i]
            p1 = pl[i + 1]
            # skip zero-length
            if p0.DistanceTo(p1) < TOL:
                continue
            segments.append((p0, p1))
    return segments

def _trim_beam_against_core_segments(p0, p1, core_segments, tol=TOL):
    """
    Given a beam from p0 to p1 (axis-aligned), trim away portions that are
    coincident with core wall segments. Return a list of (q0, q1) Point3d pairs
    representing the remaining beam pieces (0, 1, or more).
    """
    horizontal = abs(p0.Y - p1.Y) < tol
    vertical   = abs(p0.X - p1.X) < tol

    if not (horizontal or vertical):
        # For safety, if something non-axis-aligned sneaks in,
        # just return it unchanged.
        return [(p0, p1)]

    remaining = []

    if horizontal:
        y = p0.Y
        beam_start = min(p0.X, p1.X)
        beam_end   = max(p0.X, p1.X)

        # Collect overlapping intervals to remove
        to_remove = []
        for c0, c1 in core_segments:
            if abs(c0.Y - c1.Y) > tol:
                continue  # core seg not horizontal
            if abs(c0.Y - y) > tol:
                continue  # not collinear in Y

            seg_start = min(c0.X, c1.X)
            seg_end   = max(c0.X, c1.X)

            overlap_start = max(beam_start, seg_start)
            overlap_end   = min(beam_end, seg_end)
            if overlap_end - overlap_start > tol:
                to_remove.append((overlap_start, overlap_end))

        # If no overlaps, keep whole beam
        if not to_remove:
            return [(p0, p1)]

        # Merge intervals to remove
        to_remove.sort(key=lambda s: s[0])
        merged = []
        for s, e in to_remove:
            if not merged or s > merged[-1][1] + tol:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)

        # Compute complement intervals in [beam_start, beam_end]
        cur = beam_start
        kept_intervals = []
        for s, e in merged:
            if s - cur > tol:
                kept_intervals.append((cur, s))
            cur = max(cur, e)
        if beam_end - cur > tol:
            kept_intervals.append((cur, beam_end))

        # Convert to Point3d pairs, preserving original direction
        forward = p0.X <= p1.X
        for a, b in kept_intervals:
            if forward:
                q0 = r3d.Point3d(a, y, p0.Z)
                q1 = r3d.Point3d(b, y, p0.Z)
            else:
                q0 = r3d.Point3d(b, y, p0.Z)
                q1 = r3d.Point3d(a, y, p0.Z)
            if q0.DistanceTo(q1) > tol:
                remaining.append((q0, q1))

    elif vertical:
        x = p0.X
        beam_start = min(p0.Y, p1.Y)
        beam_end   = max(p0.Y, p1.Y)

        to_remove = []
        for c0, c1 in core_segments:
            if abs(c0.X - c1.X) > tol:
                continue  # not vertical
            if abs(c0.X - x) > tol:
                continue  # not collinear in X

            seg_start = min(c0.Y, c1.Y)
            seg_end   = max(c0.Y, c1.Y)

            overlap_start = max(beam_start, seg_start)
            overlap_end   = min(beam_end, seg_end)
            if overlap_end - overlap_start > tol:
                to_remove.append((overlap_start, overlap_end))

        if not to_remove:
            return [(p0, p1)]

        to_remove.sort(key=lambda s: s[0])
        merged = []
        for s, e in to_remove:
            if not merged or s > merged[-1][1] + tol:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)

        cur = beam_start
        kept_intervals = []
        for s, e in merged:
            if s - cur > tol:
                kept_intervals.append((cur, s))
            cur = max(cur, e)
        if beam_end - cur > tol:
            kept_intervals.append((cur, beam_end))

        forward = p0.Y <= p1.Y
        for a, b in kept_intervals:
            if forward:
                q0 = r3d.Point3d(x, a, p0.Z)
                q1 = r3d.Point3d(x, b, p0.Z)
            else:
                q0 = r3d.Point3d(x, b, p0.Z)
                q1 = r3d.Point3d(x, a, p0.Z)
            if q0.DistanceTo(q1) > tol:
                remaining.append((q0, q1))

    return remaining

def _segment_key(p0, p1, decimals=6):
    """
    Create an order-independent key for a segment p0–p1, rounded to `decimals`
    to avoid floating point noise.
    """
    a = (round(p0.X, decimals), round(p0.Y, decimals), round(p0.Z, decimals))
    b = (round(p1.X, decimals), round(p1.Y, decimals), round(p1.Z, decimals))
    # order-independent: always sort endpoints
    return tuple(sorted((a, b)))

def generate_beams(floor_locations, core_pls, plot=True):

    # Flatten floors
    floors_flat = [brep for cell in floor_locations for brep in cell]

    # Precompute core wall segments
    core_segments = _core_segments_from_polylines(core_pls)

    beams = []           # list of (p0, p1)
    span_dir = []        # "x" or "y" per floor panel
    seen_keys = set()    # to avoid duplicates

    for floor in floors_flat:
        minx, maxx, miny, maxy = _brep_rect_bounds(floor)

        # Get a Z from any vertex (all in same plane)
        v0 = floor.Vertices[0].Location
        z = v0.Z

        len_x = maxx - minx
        len_y = maxy - miny

        # Decide BEAM span direction
        if abs(len_x - len_y) < TOL:
            # square-ish: choose randomly
            direction = random.choice(["x", "y"])
        else:
            direction = "x" if len_x > len_y else "y"
        
        # FLOOR span dir is perpendicular to beam dir
        span_dir.append("x" if direction == "y" else "y")

        if direction == "x":
            # beams along X: bottom and top edges
            candidates = [
                (r3d.Point3d(minx, miny, z), r3d.Point3d(maxx, miny, z)),#bottom
                (r3d.Point3d(minx, maxy, z), r3d.Point3d(maxx, maxy, z)),# top
            ]
        else:
            # beams along Y: left and right edges
            candidates = [
                (r3d.Point3d(minx, miny, z), r3d.Point3d(minx, maxy, z)), # left
                (r3d.Point3d(maxx, miny, z), r3d.Point3d(maxx, maxy, z)),# right
            ]

        # Trim each candidate against core walls
        for p0, p1 in candidates:
            trimmed_segments = _trim_beam_against_core_segments(
                p0, p1, core_segments
            )
            for q0, q1 in trimmed_segments:
                if q0.DistanceTo(q1) < TOL:
                    continue

                key = _segment_key(q0, q1)
                if key in seen_keys:
                    # duplicate beam, skip
                    continue

                seen_keys.add(key)
                beams.append((q0, q1))

    if plot:
        # Convert beams to curves for plotting (different color from cores)
        beam_curves = []
        for p0, p1 in beams:
            pl = r3d.Polyline([p0, p1])
            beam_curves.append(r3d.PolylineCurve(pl))

        # Flatten floors again for plotting surfaces
        all_floors = floors_flat

        plot_rhino_geometry(
            polylines=core_pls,        # cores
            curves=beam_curves,        # beams (different color)
            surfaces=all_floors,       # floor panels
            dim=2,
            show_axes=True,
            colors={
                "polylines": "black",  # cores
                "curves": "red",       # beams
                "surfaces": "tab:blue",
            },
            fill_alpha=0.3,
        )

    return beams, span_dir

def _point_key(p, decimals=6):
    """Order a point into a hashable key for deduplication."""
    return (
        round(p.X, decimals),
        round(p.Y, decimals),
        round(p.Z, decimals),
    )

def generate_columns_1floor(
        input_params, beam_locations, floor_locations, plot=False
    ):
    """
    input_params: input parameters dict
    beam_locations: list of (Point3d, Point3d) from generate_beams
    floor_locations: [[Brep1, Brep2, ...], [Breps in cell 2], ...]
    plot: if True, make a 3D plot with floors (plan) + columns

    Returns:
        columns_1floor: list of (base_point, top_point) Point3d tuples
    """
    column_height = input_params["column_height"]

    # 1. Collect unique column base locations from beam endpoints
    seen = set()
    base_points = []

    for p0, p1 in beam_locations:
        for p in (p0, p1):
            # we consider columns starting at z=0 (first floor)
            # if there is tiny noise, clamp to 0 if close
            z0 = 0.0
            if abs(p.Z - z0) < TOL:
                p_use = r3d.Point3d(p.X, p.Y, z0)
            else:
                p_use = r3d.Point3d(p.X, p.Y, p.Z)

            key = _point_key(p_use)
            if key not in seen:
                seen.add(key)
                base_points.append(p_use)

    # 2. Create column line segments in 3D
    columns_1floor = []
    for bp in base_points:
        base = r3d.Point3d(bp.X, bp.Y, bp.Z)
        top = r3d.Point3d(bp.X, bp.Y, bp.Z + column_height)
        columns_1floor.append((base, top))

    # 3. Optional plotting
    if plot:
        # Floors: flatten the nested list
        floors_flat = [brep for cell in floor_locations for brep in cell]

        # Convert columns to polylines for plotting
        col_pls = []
        for base, top in columns_1floor:
            pl = r3d.Polyline([base, top])
            col_pls.append(pl)

        # 3D plot: floors as surfaces in XY plane, columns as vertical polylines
        plot_rhino_geometry(
            polylines=col_pls,     # columns as red-ish polylines
            surfaces=floors_flat,  # floor panels at z=0
            dim=3,
            show_axes=True,
            colors={
                "polylines": "red",     # columns
                "surfaces": "tab:blue"  # floors
            },
            fill_alpha=0.3,
        )

    return columns_1floor

def _column_base_points(columns_1floor):
    """Extract base points of all columns (lowest Z)."""
    bases = []
    for p0, p1 in columns_1floor:
        base = p0 if p0.Z <= p1.Z else p1
        # snap tiny Z to 0
        z0 = 0.0 if abs(base.Z) < TOL else base.Z
        bases.append(r3d.Point3d(base.X, base.Y, z0))
    return bases

def _make_wall_brep_from_segment(p0, p1, height):
    """
    Create a vertical rectangular wall Brep from segment p0->p1 along XY,
    extruded up by `height`.
    Assumes segment is axis-aligned (horizontal or vertical).
    """
    x0, y0, z0 = p0.X, p0.Y, p0.Z
    x1, y1, z1 = p1.X, p1.Y, p1.Z

    # Force base Z to the same (typically 0)
    base_z = min(z0, z1)
    top_z = base_z + height

    # Four corners (bottom edge, top edge) in order
    bottom0 = r3d.Point3d(x0, y0, base_z)
    bottom1 = r3d.Point3d(x1, y1, base_z)
    top1    = r3d.Point3d(x1, y1, top_z)
    top0    = r3d.Point3d(x0, y0, top_z)

    pl = r3d.Polyline([bottom0, bottom1, top1, top0, bottom0])
    crv = r3d.PolylineCurve(pl)

    # Define plane depending on orientation
    if abs(y0 - y1) < TOL:
        # horizontal segment -> wall plane is parallel to XZ, normal in +Y
        plane = r3d.Plane(bottom0, r3d.Vector3d(0, 1, 0))
    elif abs(x0 - x1) < TOL:
        # vertical segment -> wall plane is parallel to YZ, normal in +X
        plane = r3d.Plane(bottom0, r3d.Vector3d(1, 0, 0))
    else:
        # Fallback: arbitrary plane using segment+Z
        v_seg = r3d.Vector3d(x1 - x0, y1 - y0, 0.0)
        v_z   = r3d.Vector3d(0.0, 0.0, 1.0)
        n     = r3d.Vector3d.CrossProduct(v_seg, v_z)
        plane = r3d.Plane(bottom0, n)

    brep = r3d.Brep.CreateTrimmedPlane(plane, crv)
    return brep

def generate_walls_1floor(
        input_params, core_pls, columns_1floor, floor_locations, plot=False
    ):
    """
    Generate first-floor walls as vertical surfaces from core outlines,
    split where columns sit on the wall.

    input_params: dict with at least "column_height"
    core_pls: list of core boundary polylines at z=0
    columns_1floor: list of (base_point, top_point) for columns
    floor_locations: [[Brep1, Brep2, ...], [Breps in cell 2], ...]
    plot: if True, plot columns, wall surfaces, and floor surfaces (plan)

    Returns:
        walls_1floor: flat list of Breps (wall panels)
    """
    height = input_params["column_height"]
    col_bases = _column_base_points(columns_1floor)

    walls_1floor = []

    for pl in core_pls:
        # Go over each polyline edge
        for i in range(len(pl) - 1):
            p0 = pl[i]
            p1 = pl[i + 1]
            if p0.DistanceTo(p1) < TOL:
                continue

            # Detect orientation
            horizontal = abs(p0.Y - p1.Y) < TOL
            vertical   = abs(p0.X - p1.X) < TOL

            if not (horizontal or vertical):
                # If something weird sneaks in, just make a single wall
                brep = _make_wall_brep_from_segment(p0, p1, height)
                if brep:
                    walls_1floor.append(brep)
                continue

            # Parametric coordinate along the segment (use X or Y)
            if horizontal:
                y = p0.Y
                s0 = min(p0.X, p1.X)
                s1 = max(p0.X, p1.X)
            else:  # vertical
                x = p0.X
                s0 = min(p0.Y, p1.Y)
                s1 = max(p0.Y, p1.Y)

            # Collect split positions from columns sitting *inside* the segment
            split_coords = [s0, s1]

            for cb in col_bases:
                if horizontal:
                    # check same Y, X inside (strict)
                    if abs(cb.Y - y) < TOL and (s0 + TOL) < cb.X < (s1 - TOL):
                        split_coords.append(cb.X)
                else:  # vertical
                    if abs(cb.X - x) < TOL and (s0 + TOL) < cb.Y < (s1 - TOL):
                        split_coords.append(cb.Y)

            # Sort & unique split positions
            split_coords = sorted(set(split_coords))

            # Create sub-walls between consecutive split coords
            for j in range(len(split_coords) - 1):
                a = split_coords[j]
                b = split_coords[j + 1]
                if b - a < TOL:
                    continue

                if horizontal:
                    if p0.X <= p1.X:
                        q0 = r3d.Point3d(a, y, p0.Z)
                        q1 = r3d.Point3d(b, y, p0.Z)
                    else:
                        q0 = r3d.Point3d(b, y, p0.Z)
                        q1 = r3d.Point3d(a, y, p0.Z)
                else:  # vertical
                    if p0.Y <= p1.Y:
                        q0 = r3d.Point3d(x, a, p0.Z)
                        q1 = r3d.Point3d(x, b, p0.Z)
                    else:
                        q0 = r3d.Point3d(x, b, p0.Z)
                        q1 = r3d.Point3d(x, a, p0.Z)

                brep = _make_wall_brep_from_segment(q0, q1, height)
                if brep:
                    walls_1floor.append(brep)

    if plot:
        # Floors: flatten nested list
        floors_flat = [brep for cell in floor_locations for brep in cell]

        # Columns: convert to polylines
        col_pls = []
        for base, top in columns_1floor:
            col_pls.append(r3d.Polyline([base, top]))

        # Plot floors (plan) + walls (vertical surfaces) + columns
        plot_rhino_geometry(
            polylines=col_pls,             # columns as polylines
            surfaces=floors_flat + walls_1floor,
            dim=3,
            show_axes=True,
            colors={
                "polylines": "red",        # columns
                "surfaces": "tab:blue",    # floors + walls (same color for now)
            },
            fill_alpha=0.3,
        )

    return walls_1floor

def _point_key(p, decimals=6):
    return (
        round(p.X, decimals),
        round(p.Y, decimals),
        round(p.Z, decimals),
    )


def _core_pls_to_surfaces(core_pls):
    """Create planar Breps (floors) from core polylines in WorldXY at z=0."""
    plane = r3d.Plane.WorldXY()
    breps = []
    for pl in core_pls:
        crv = r3d.PolylineCurve(pl)
        brep = r3d.Brep.CreateTrimmedPlane(plane, crv)
        if brep:
            breps.append(brep)
    return breps


def generate_full_building(
        input_params, floor_locations, beam_locations, span_dirs,
        columns_1floor, walls_1floor, core_pls, plot=False
    ):
    """
    input_params:
        - "nb_floors": number of storeys
        - "column_height": height of a single level

    floor_locations: [[Brep1, Brep2, ...], [Breps in cell 2], ...] at z=0 (NO cores)
    beam_locations:  list of (Point3d, Point3d) at z=0
    span_dirs:       list of "x"/"y" (same length as flattened floor_locations)
    columns_1floor:  list of (base_point, top_point) for first floor (0 -> h)
    walls_1floor:    list of Breps (0 -> h) from core walls
    core_pls:        list of core rectangular polylines at z=0

    Returns:
        nodes
        floors            # all non-foundation floors incl. roof (no z=0)
        beams             # all non-foundation beams incl. roof
        span_dirs         # span dirs for those beams
        columns           # all column segments for all floors
        walls             # all walls for all floors
        foundation_floors # floors at z=0 (incl. cores)
        foundation_beams  # beams at z=0
        foundation_span_dirs
    """
    nb_floors = int(input_params["nb_floors"])
    h = float(input_params["column_height"])

    # 1. Flatten base floors (z = 0) 
    base_floors = [brep for cell in floor_locations for brep in cell]

    # 2. Core slabs at ground (z=0)
    core_ground_floors = _core_pls_to_surfaces(core_pls)

    # Compute span dirs for core ground floors:
    # floor spans in SHORTEST direction; if equal, choose randomly.
    core_ground_span_dirs = []
    for core_brep in core_ground_floors:
        minx, maxx, miny, maxy = _brep_rect_bounds(core_brep)
        len_x = maxx - minx
        len_y = maxy - miny

        if abs(len_x - len_y) < TOL:
            d = random.choice(["x", "y"])
        else:
            # shortest direction
            d = "x" if len_x < len_y else "y"

        core_ground_span_dirs.append(d)

    # Foundation floors = base floors + core slabs at ground
    foundation_floors = base_floors + core_ground_floors

    # Foundation beams & span dirs = ground floor
    foundation_beams = list(beam_locations)

    # Foundation span dirs = base floor span_dirs + core floor span_dirs
    foundation_span_dirs = list(span_dirs) + core_ground_span_dirs

    # 3. Generate floors above ground (incl. roof)
    floors = []
    floors_span_dirs = []  # floor span directions, same order as 'floors'

    for level_idx in range(1, nb_floors + 1):
        z_off = level_idx * h
        xform = r3d.Transform.Translation(0.0, 0.0, z_off)

        # duplicate the base floors at this level, preserving their span_dirs
        for brep, d in zip(base_floors, span_dirs):
            dup = brep.Duplicate()
            dup.Transform(xform)
            floors.append(dup)
            floors_span_dirs.append(d)

        # ONLY on the roof: add core slabs with span dirs
        if level_idx == nb_floors:
            # same geometry and bounding boxes as ground cores,
            # so we reuse core_ground_span_dirs for consistency
            for core_brep, d_core in zip(core_ground_floors, core_ground_span_dirs):
                dup = core_brep.Duplicate()
                dup.Transform(xform)
                floors.append(dup)
                floors_span_dirs.append(d_core)
    
    # span_dirs_full is per FLOOR
    span_dirs_full = floors_span_dirs

     # 4. Generate beams above ground (incl. roof)
    beams = []
    for level_idx in range(1, nb_floors + 1):
        z_off = level_idx * h
        for (p0, p1) in beam_locations:
            q0 = r3d.Point3d(p0.X, p0.Y, p0.Z + z_off)
            q1 = r3d.Point3d(p1.X, p1.Y, p1.Z + z_off)
            beams.append((q0, q1))

    # 5. Generate columns for all floors
    # Get unique XY bases from columns_1floor
    col_xy_keys = set()
    col_xy_points = []

    for p0, p1 in columns_1floor:
        base = p0 if p0.Z <= p1.Z else p1
        bp = r3d.Point3d(base.X, base.Y, 0.0)
        key = (round(bp.X, 6), round(bp.Y, 6))
        if key not in col_xy_keys:
            col_xy_keys.add(key)
            col_xy_points.append(bp)

    columns = []
    for bp in col_xy_points:
        for level_idx in range(nb_floors):
            z_base = level_idx * h
            z_top = (level_idx + 1) * h
            base = r3d.Point3d(bp.X, bp.Y, z_base)
            top = r3d.Point3d(bp.X, bp.Y, z_top)
            columns.append((base, top))

    # 6. Generate walls for all floors
    walls = []
    # first storey walls
    walls.extend(walls_1floor)
    # walls for floors 2..nb_floors
    for level_idx in range(1, nb_floors):
        z_off = level_idx * h
        xform = r3d.Transform.Translation(0.0, 0.0, z_off)
        for w in walls_1floor:
            dup = w.Duplicate()
            dup.Transform(xform)
            walls.append(dup)

    # 7. Assemble global nodes list (beams, columns, walls, floors)
    node_keys = set()
    nodes = []

    # All beams including foundation
    all_beams_all_levels = foundation_beams + beams
    for p0, p1 in all_beams_all_levels:
        for p in (p0, p1):
            key = _point_key(p)
            if key not in node_keys:
                node_keys.add(key)
                nodes.append(p)

    # Columns
    for base, top in columns:
        for p in (base, top):
            key = _point_key(p)
            if key not in node_keys:
                node_keys.add(key)
                nodes.append(p)

    # Walls
    for w in walls:
        for v in w.Vertices:
            p = v.Location
            key = _point_key(p)
            if key not in node_keys:
                node_keys.add(key)
                nodes.append(p)
    
    # Floors
    all_floors_all_levels = foundation_floors + floors
    for f in all_floors_all_levels:
        for v in f.Vertices:
            p = v.Location
            key = _point_key(p)
            if key not in node_keys:
                node_keys.add(key)
                nodes.append(p)

    # 8. Optional plotting of the entire building
    if plot:
        surfaces_plot = foundation_floors + floors + walls

        # Columns as polylines
        col_pls = [r3d.Polyline([base, top]) for base, top in columns]

        # Beams (incl. foundation) as curves
        beam_curves = []
        for p0, p1 in all_beams_all_levels:
            pl = r3d.Polyline([p0, p1])
            beam_curves.append(r3d.PolylineCurve(pl))

        plot_rhino_geometry(
            polylines=col_pls,        # columns
            curves=beam_curves,       # beams
            surfaces=surfaces_plot,   # floors (incl. cores) + walls
            points=None,
            dim=3,
            show_axes=True,
            colors={
                "polylines": "black",  # columns
                "curves": "red",       # beams
                "surfaces": "tab:blue"
            },
            fill_alpha=0.3,
        )

    return (
        nodes,
        floors,
        beams,
        span_dirs_full,
        columns,
        walls,
        foundation_floors,
        foundation_beams,
        foundation_span_dirs,
    )

def main():

    # Load sampling
    sampling_df = pd.read_csv(
        Path(
            "C:/Users/maxime.pollet/Documents/Code/BuildingLCA/data/sampling/" \
                + "mapped_sampling.csv"
        )
    )
    
    for i in tqdm.tqdm(range(len(sampling_df))):

        input_params = {
            "sample_id": 0,
            "nb_floors": sampling_df.iloc[i]['nb_floors'].item(),
            "building_width": sampling_df.iloc[i]['building_width'].item(),
            "building_length": sampling_df.iloc[i]['building_length'].item(),
            "column_height": 2.5,
            "core_length": sampling_df.iloc[i]['core_length'].item(),
            "core_width": sampling_df.iloc[i]['core_width'].item(),
            "max_stair_dist": sampling_df.iloc[i]['max_stairs_distance'].item(),
            "core_y_location": sampling_df.iloc[i]['core_y_location'].item(),
            "column_spacing_x": sampling_df.iloc[i]['column_spacing_x'].item(),
            "column_spacing_y": sampling_df.iloc[i]['column_spacing_y'].item()
        }

        core_pls = get_core_locations(input_params, plot=True)
        cell_pls, cell_surfs = create_base_surfaces(
            input_params, core_pls, plot=True
        )
        column_locations = generate_column_locations(
            input_params, cell_surfs, plot=True
        )
        floor_locations = generate_floors(column_locations, plot=True)
        beam_locations, span_dirs = generate_beams(
            floor_locations, core_pls, plot=True
        )
        columns_1floor = generate_columns_1floor(
            input_params, beam_locations, floor_locations, plot=True
        )
        walls_1floor = generate_walls_1floor(
            input_params, core_pls, columns_1floor, floor_locations, plot=True
        )
        nodes, floors, beams, span_dirs, columns, walls, foundation_floors, \
        foundation_beams, foundation_span_dirs = generate_full_building(
            input_params, floor_locations, beam_locations, span_dirs,
            columns_1floor, walls_1floor, core_pls, plot=True
        )

if __name__ == "__main__":
    main()