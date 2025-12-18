import math
import os
import numpy as np
import pandas as pd
from copy import deepcopy

def brep_area(brep):
    pts = [
        (
            brep.Vertices[0].Location.X,
            brep.Vertices[0].Location.Y,
            brep.Vertices[0].Location.Z
        ),
        (
            brep.Vertices[1].Location.X,
            brep.Vertices[1].Location.Y,
            brep.Vertices[1].Location.Z
        ),
        (
            brep.Vertices[2].Location.X,
            brep.Vertices[2].Location.Y,
            brep.Vertices[2].Location.Z
        ),
        (
            brep.Vertices[3].Location.X,
            brep.Vertices[3].Location.Y,
            brep.Vertices[3].Location.Z
        ),
    ]

    # Pick one point and find the two adjacent vertices
    p0 = pts[0]
    others = pts[1:]

    # Compute squared distances from p0 to the three other points
    dists = []
    for p in others:
        v = sub(p, p0)
        d2 = dot(v, v)
        dists.append((d2, p, v))  # store squared distance, point, and vector

    # Sort by distance; the two smallest distances from a corner are the sides
    dists.sort(key=lambda t: t[0])
    v1 = dists[0][2]  # vector along one side
    v2 = dists[1][2]  # vector along the other side

    # Area of rectangle = magnitude of cross product of adjacent side vectors
    area = norm(cross(v1, v2))

    return area

def coerce_mixed_to_string(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        inferred = pd.api.types.infer_dtype(df[c], skipna=True)
        if inferred.startswith("mixed") or inferred in {"string", "unicode"}:
            df[c] = df[c].astype("string[pyarrow]")

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def dot(u, v):
    """
    Compute the dot product of two 3‑dimensional vectors.

    Args:
        u (Sequence[float]): First vector; must have exactly three numeric
        components.
        v (Sequence[float]): Second vector; must have exactly three numeric
        components.

    Returns:
        float: The dot product (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]).
    """
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def cross(u, v):
    """
    Compute the cross product of two 3‑dimensional vectors.

    Args:
        u (Sequence[float]): First vector; must have exactly three numeric
        components.
        v (Sequence[float]): Second vector; must have exactly three numeric
        components.

    Returns:
        Tuple[float, float, float]: The cross product vector
            (u[1]*v[2] - u[2]*v[1],
             u[2]*v[0] - u[0]*v[2],
             u[0]*v[1] - u[1]*v[0]).
    """
    return (
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    )

def norm(u):
    """
    Compute the Euclidean norm (length) of a 3‑dimensional vector.

    Args:
        u (Sequence[float]): Vector; must have exactly three numeric components.

    Returns:
        float: The Euclidean norm, defined as sqrt(u[0]² + u[1]² + u[2]²).
    """
    return math.sqrt(dot(u, u))

def segments_coincident(p1, p2, q1, q2, tol=0.001):
    """
    Determine whether two 3D line segments overlap (are coincident) along a
    common line.

    This checks that both segments lie on the same infinite line
    (within a tolerance) and that their projections onto that line have a
    positive-length interval in common.

    Args:
        p1 (object): First endpoint of segment P,
        with numeric attributes `.x`, `.y`, `.z`.
        p2 (object): Second endpoint of segment P,
        with numeric attributes `.x`, `.y`, `.z`.
        q1 (object): First endpoint of segment Q,
        with numeric attributes `.x`, `.y`, `.z`.
        q2 (object): Second endpoint of segment Q,
        with numeric attributes `.x`, `.y`, `.z`.
        tol (float, optional): Tolerance for collinearity and minimum overlap
        length. Defaults to 0.001.

    Returns:
        bool: True if the two segments share a positive-length overlap on the
        same line;
            False otherwise.

    Notes:
        - Returns False if P is degenerate (its length < tol).
        - Uses cross‑product to test collinearity and dot‑product for projection
    """
    # Convert to vectors
    v = (p2.x-p1.x, p2.y-p1.y, p2.z-p1.z)
    w1 = (q1.x-p1.x, q1.y-p1.y, q1.z-p1.z)
    w2 = (q2.x-p1.x, q2.y-p1.y, q2.z-p1.z)

    # Degenerate segment?
    if norm(v) < tol:
        # p1==p2 is a point; can't have positive-length overlap
        return False

    # 1) Check collinearity: both q1 and q2 must lie on the infinite line p1->p2
    if norm(cross(v, w1)) > tol or norm(cross(v, w2)) > tol:
        return False

    # 2) Project q1, q2 onto the p1→p2 line (parametric t)
    v2 = dot(v, v)
    t1 = dot(v, w1) / v2
    t2 = dot(v, w2) / v2

    # Our two parametric intervals:
    #    p‐segment: [0, 1]
    #    q‐segment: [min(t1,t2), max(t1,t2)]
    t_min_q, t_max_q = min(t1, t2), max(t1, t2)

    # Find overlap interval
    t_start = max(0.0, t_min_q)
    t_end   = min(1.0, t_max_q)

    # Positive-length overlap?
    return (t_end - t_start) > tol

def is_point_on_line(pt, p1, p2, tol=0.001):
    """
    Determine whether a point lies on the line segment between two 3D points.

    Checks collinearity within a tolerance and ensures the point’s projection
    falls between the endpoints (inclusive, within tol).

    Args:
        pt (object): Point to test, with numeric attributes `.x`, `.y`, `.z`.
        p1 (object): First endpoint of the line segment, with `.x`, `.y`, `.z`.
        p2 (object): Second endpoint of the line segment, with `.x`, `.y`, `.z`.
        tol (float, optional): Numerical tolerance for comparisons (distance,
            collinearity). Defaults to 0.001.

    Returns:
        bool: True if `pt` lies on the segment `p1`→`p2` (within `tol`), False
        otherwise.

    Notes:
        - If `p1` and `p2` are effectively the same point (within `tol`),
          returns True only if `pt` coincides with that point (within `tol`).
        - Uses cross-product magnitude to test collinearity and dot-product
          bounds to test the segment interval.
    """
    x = pt.x
    y = pt.y
    z = pt.z
    x1 = p1.x
    y1 = p1.y
    z1 = p1.z
    x2 = p2.x
    y2 = p2.y
    z2 = p2.z

    # Handle degenerate case: p1 == p2
    if (abs(x2 - x1) < tol and
        abs(y2 - y1) < tol and
        abs(z2 - z1) < tol):
        return (abs(x - x1) < tol and
                abs(y - y1) < tol and
                abs(z - z1) < tol)

    # Compute vectors
    vx, vy, vz = x2 - x1, y2 - y1, z2 - z1   # p1→p2
    wx, wy, wz = x  - x1, y  - y1, z  - z1   # p1→pt

    # 1) Collinearity: cross(v, w) must be near zero vector
    cx = vy*wz - vz*wy
    cy = vz*wx - vx*wz
    cz = vx*wy - vy*wx
    if math.sqrt(cx*cx + cy*cy + cz*cz) > tol:
        return False

    # 2) Bounding: dot(v, w) between 0 and |v|^2
    dot = vx*wx + vy*wy + vz*wz
    if dot < -tol:
        return False
    len2 = vx*vx + vy*vy + vz*vz
    if dot - len2 > tol:
        return False

    return True

class StructuralElementsDB:
    """
    Database interface for structural element libraries (floors, beams, columns,
    shear walls).

    Loads pre-calculated CSV libraries of floors, beams, and columns from a
    given directory
    and provides methods to query the best-fit element based on material and
    geometric/load criteria.
    """
    def __init__(self, path):
        """
        Initialize the StructuralElementsDB.

        Args:
            path (Path): Directory path containing 'floors_db.csv',
            'beams_db.csv', and 'columns_db.csv'.
        """

        self.path = path

        # load libraries of pre-calculated floors, beams, and columns
        self.floors = pd.read_csv(path.joinpath('floors_db.csv'))
        self.beams = pd.read_csv(path.joinpath('beams_db.csv'))
        self.columns = pd.read_csv(path.joinpath('columns_db.csv'))
    
    def get_column(self, material, height, n_uls, column_id=None):
        """
        Retrieve the lightest column matching the given material, height, and
        ultimate limit state load.

        Args:
            material (str): Material type, one of:
            ['Concrete', 'Steel', 'Timber'].
            height (float): Desired column height in meters.
            n_uls (float): Ultimate limit state load capacity in kN.

        Returns:
            dict: Characteristics of the best-fit column, including:
                - 'Material': dict mapping component names to mass in kg.
                - 'Column thickness': Thickness in mm.
                - 'Section': Section identifier.

        Raises:
            ValueError: If no matching column is found in the database.
        """

        # Find all columns that have the right material, height, and can sustain
        # the load
        possible_columns = self.columns[
            (self.columns['Material'] == material) &
            (self.columns['Column height (m)'].sub(height).abs() < 0.001) &
            (self.columns['ULS limit load (kN)'].sub(n_uls) * -1 < 0.001)
        ]

        # if there's no matching column raise an error
        if len(possible_columns) == 0:
            raise ValueError(
                f'Error: No matching column found for {column_id}: \n' \
                + '    Material: {:s}\n'.format(material) \
                + '    Height = {:f} m\n'.format(height) \
                + '    n_uls = {:f} kN\n'.format(n_uls)
            )

        # Determine which of the possible columns is the best
        # (i.e. the least heavy -> the one that can support the smallest load)
        best_column = possible_columns.loc[
            possible_columns['ULS limit load (kN)'].idxmin()
        ]

        # Format result
        result = {}
        if material == 'Concrete':
            result['Material'] = {
                'Poteau-Béton CEMII': np.round(
                    best_column['Concrete mass (kg)'], 2
                ),
                'Poteau-Ferraillage': np.round(
                    best_column['Reinforcement mass (kg)'], 2
                ),
            }
        elif material == 'Steel':
            result['Material'] = {
                'Poteau-Acier-S235': np.round(
                    best_column['Steel mass (kg)'], 2
                ),
                'Poteau-Acier-Assemblage': np.round(
                    best_column['Steel mass (kg)']*0.1, 2
                ),
                'Poteau-Flocage': np.round(
                    best_column['Coating mass (kg)'], 2
                ),
            }
        elif material == 'Timber':
            result['Material'] = {
                'Poteau-Bois-GL24h': np.round(
                    best_column['Timber mass (kg)'], 2
                ),
                'Poteau-Acier-Assemblage': np.round(
                    best_column['Timber mass (kg)']*0.1, 2
                )
            }
        
        result['Column thickness'] = best_column['Column thickness (mm)']
        result['Section'] = best_column['Section']

        return result
    
    def get_beam(self, material, span, q_lin, g_lin, beam_id=None):
        """
        Retrieve the lightest beam matching the given material, span, and
        lineloads.

        Args:
            material (str): Material type, one of:
            ['Concrete', 'Steel', 'Timber'].
            span (float): Beam span in meters.
            q_lin (float): Live load lineload in kN/m.
            g_lin (float): Dead load lineload in kN/m.

        Returns:
            dict: Characteristics of the best-fit beam, including:
                - 'Material': dict mapping component names to mass in kg.
                - 'Height': Height in mm.
                - 'Section': Section identifier.

        Raises:
            ValueError: If no matching beam is found in the database.
        """

        # Check if there are matching beams
        possible_beams = self.beams[
            (self.beams['Material'] == material) &
            (self.beams['Beam span (m)'].sub(span).abs() < 0.001) &
            (self.beams['Q lin. (kN/ml)'].sub(q_lin) * -1 < 0.001) &
            (self.beams['G lin. (kN/ml)'].sub(g_lin) * -1 < 0.001)
        ].copy()

        # if there's no matching beam raise an error
        if len(possible_beams) == 0:
            raise ValueError(
                f'Error: No matching beam found for {beam_id}: \n' \
                + '    Material: {:s}\n'.format(material) \
                + '    Span = {:f}\n'.format(span) \
                + '    Qlin = {:f}\n'.format(q_lin) \
                + '    Glin = {:f}\n'.format(g_lin)
            )
        
        # Determine which of the possible beams is the best
        # (i.e. the least heavy)
        possible_beams['Total mass (kg)'] = possible_beams[
            [
                'Timber mass (kg)', 'Steel mass (kg)', 'Coating mass (kg)',
                'Concrete mass (kg)', 'Reinforcement mass (kg)'
            ]
        ].sum(1)
        best_beam = possible_beams.loc[
            possible_beams['Total mass (kg)'].idxmin()
        ]

        # Prepare result dict
        result = {}
        if material == 'Concrete':
            result['Material'] = {
                'Poutre-Béton CEMII': np.round(
                    best_beam['Concrete mass (kg)'], 2
                ),
                'Poutre-Ferraillage': np.round(
                    best_beam['Reinforcement mass (kg)'], 2
                )
            }
        # Else if material is steel
        elif material == 'Steel':
            result['Material'] = {
                'Poutre-Acier-S235': np.round(
                    best_beam['Steel mass (kg)'], 2
                ),
                'Poutre-Acier-Assemblage': np.round(
                    best_beam['Steel mass (kg)']*0.1, 2
                ),
                'Poutre-Flocage': np.round(
                    best_beam['Coating mass (kg)'], 2
                ),
            }
        # Else if material is timber
        elif material == 'Timber':
            result['Material'] = {
                'Poutre-Bois-GL24h': np.round(
                    best_beam['Timber mass (kg)'], 2
                ),
                'Poutre-Acier-Assemblage': np.round(
                    best_beam['Timber mass (kg)']*0.1, 2
                )
            }
        
        result['Height'] = best_beam['Height (mm)']
        result['Section'] = best_beam['Section']

        return result
    
    def get_floor(self, typology, function, floor_span, floor_id=None):
        """
        Retrieve the floor configuration matching typology, function, and span.

        Args:
            typology (str): Floor typology identifier.
            function (str): Usage function (e.g., 'Logement', 'Bureau').
            floor_span (float): Span in meters.

        Returns:
            dict: Characteristics of the floor, including:
                - 'Material': dict mapping component names to kg/m2 quantities.
                - 'Height': Floor thickness in cm.
                - 'G': Dead load in kN/m2.

        Raises:
            ValueError: If no matching floor is found in the database.
        """

        # Check if there is a matching floor
        best_floor = self.floors[
            (self.floors['Typology'] == typology) &
            (self.floors['Span (m)'].sub(floor_span).abs() < 0.001) &
            (self.floors['Function'] == function)
        ].squeeze(0)

        # if there's no matching floor raise an error
        if len(best_floor) == 0:
            raise ValueError(
                f'Error: No matching floor found for {floor_id}: \n' \
                + '    Typology: {:s}\n'.format(typology) \
                + '    Span = {:f}\n'.format(floor_span) \
                + '    Function: {:s}'.format(function)
            )
        
        # format results
        result = {
            'Material': {},
            'Height': best_floor['Height (cm)'],
            'G': best_floor['G (kN/m2)']
        }
        for i in range(4):
            if pd.isna(best_floor['Comp. {:d} name'.format(i+1)]):
                break
            result['Material'][best_floor['Comp. {:d} name'.format(i+1)]] \
                = best_floor['Comp. {:d} quantity (kg/m2)'.format(i+1)]
        
        return result
    
    def get_shear_wall(self, length, height, material, G, Q, V):
        """
        Estimate concrete and reinforcement mass for a shear wall.

        Args:
            length (float): Wall length in meters.
            height (float): Wall height in meters.
            material (str): Wall material (currently unused parameter).
            G (float): Dead load lineload in kN/m (unused).
            Q (float): Live load lineload in kN/m (unused).
            V (float): Shear force in kN (unused).

        Returns:
            dict: Characteristics of the shear wall, including:
                - 'Material': dict with 'Mur-Béton CEMII' and 'Mur-Ferraillage'
                masses in kg.
                - 'Width': Wall thickness in meters.
        """
        concrete_mass = np.round(0.20 * length * height * 2400, 1) # in kg
        reinforcement_mass = np.round(25 * concrete_mass / 2400, 1) # in kg

        result = {
            'Material': {
                'Mur-Béton CEMII': concrete_mass,
                'Mur-Ferraillage': reinforcement_mass
            },
            'Width': 0.2
        }
        return result

class Node:
    """
    Represents a 3D point in a structural model and tracks connected elements.

    Attributes:
        id (str): Unique identifier for the node.
        x (float): X-coordinate, rounded to 3 decimals.
        y (float): Y-coordinate, rounded to 3 decimals.
        z (float): Z-coordinate, rounded to 3 decimals.
        connected_columns (list[str]): IDs of connected column elements.
        connected_beams (list[str]): IDs of connected beam elements.
        connected_floors (list[str]): IDs of connected floor elements.
        connected_walls (list[str]): IDs of connected shear-wall elements.
    """
    def __init__(self, id, x, y, z):
        """
        Initialize a Node with coordinates and empty connection lists.

        Args:
            id (str): Unique node identifier.
            x (float): X-coordinate.
            y (float): Y-coordinate.
            z (float): Z-coordinate.
        """
        self.id = id
        self.x = round(x, 3)
        self.y = round(y, 3)
        self.z = round(z, 3)
        self.connected_columns = []
        self.connected_floors = []
        self.connected_beams = []
        self.connected_walls = []
    
    def __str__(self):
        return f'{self.id}: x = {self.x}, y = {self.y}, z = {self.z}'
    
    def __repr__(self):
        return self.id
    
    def get_connected_columns(self, all_columns, all_nodes):
        """
        Populate connected_columns with IDs of columns incident on this node.

        Args:
            all_columns (dict[str, object]): Mapping of column IDs to column
            objects.
            all_nodes (dict[str, Node]): Mapping of node IDs to Node instances.
        """
        for column in all_columns.values():
            if all_nodes[column.startpoint].id == self.id \
                or all_nodes[column.endpoint].id == self.id:
                self.connected_columns.append(column.id)
    
    def get_connected_beams(self, all_beams, all_nodes):
        """
        Populate connected_beams with IDs of beams incident on this node.

        Args:
            all_beams (dict[str, object]): Mapping of beam IDs to beam objects.
            all_nodes (dict[str, Node]): Mapping of node IDs to Node instances.
        """
        for beam in all_beams.values():
            if all_nodes[beam.startpoint].id == self.id \
                or all_nodes[beam.endpoint].id == self.id:
                self.connected_beams.append(beam.id)
    
    def get_connected_floors(self, all_floors, all_nodes):
        """
        Populate connected_floors with IDs of floor elements incident on this
        node.

        Args:
            all_floors (dict[str, object]): Mapping of floor IDs to floor
            objects.
            all_nodes (dict[str, Node]): Mapping of node IDs to Node instances.
        """
        for floor in all_floors.values():
            if all_nodes[floor.node_1].id == self.id \
                or all_nodes[floor.node_2].id == self.id \
                    or all_nodes[floor.node_3].id == self.id \
                        or all_nodes[floor.node_4].id == self.id:
                self.connected_floors.append(floor.id)
    
    def get_connected_walls(self, all_walls, all_nodes):
        """
        Populate connected_walls with IDs of shear-wall elements incident on
        this node.

        Args:
            all_walls (dict[str, object]): Mapping of wall IDs to wall objects.
            all_nodes (dict[str, Node]): Mapping of node IDs to Node instances.
        """
        for wall in all_walls.values():
            if all_nodes[wall.node_1].id == self.id \
                or all_nodes[wall.node_2].id == self.id \
                    or all_nodes[wall.node_3].id == self.id \
                        or all_nodes[wall.node_4].id == self.id:
                self.connected_walls.append(wall.id)
    
    def distance_to_node(self, node):
        """
        Compute the Euclidean distance between this node and another node.

        Args:
            node (Node): The target node to measure distance to.

        Returns:
            float: Euclidean distance between the two nodes.
        """
        return np.sqrt(
            (self.x - node.x)**2 + (self.y - node.y)**2 + (self.z - node.z)**2
        )

class FloorSystem:
    """
    Represents a floor system in a structural model, defining geometry, loading,
    and material requirements.

    Attributes:
        id (str): Unique identifier for the floor system.
        typology (str): Floor typology identifier used for database lookup.
        beam_material (str): Mapped material for beams
            ('Concrete', 'Steel', or 'Timber').
        surface (float): Floor surface area in square meters.
        span_dir (str): Direction of span: 'x', 'y', or 'xy'.
        elements_db (StructuralElementsDB): Database interface for retrieving
            element data.
        building_function (str): Use case of the floor
            ('Logement', 'Bureau', 'Gare').
        calc_secondary_beams (bool): Whether to calculate secondary beam layout.
        node_1, node_2, node_3, node_4 (str): Ordered corner node IDs defining
            the floor corners.
        span (float): Distance between node_1 and node_3 (longitudinal span)
            in meters.
        width (float): Distance between node_1 and node_2 (transverse width)
            in meters.
        q (float): Live load in kN/m2 based on building_function.
        g_superimposed (float): Superimposed dead load in kN/m2 (constant 1.2).
        g_floor (float): Dead load of the structural floor element in kN/m2.
        g (float): Total dead load (floor + superimposed) in kN/m2.
        height (float): Floor thickness in meters.
        elements (dict): Material quantities in kg for each component over the
            entire surface.
    """
    def __init__(self, id, typology, beam_material, surface, span_dir,
                 node_1, node_2, node_3, node_4, existing_nodes, elements_db,
                 building_function, calc_secondary_beams):
        """
        Initialize the FloorSystem and identify corner node order and dimensions.

        Args:
            id (str): Unique floor system identifier.
            typology (str): Floor typology for database queries.
            beam_material (str): Material label ('Beton', 'Acier', 'Bois')
                mapped internally.
            surface (float): Floor area in m2.
            span_dir (str): Span direction: 'x', 'y', or 'xy'.
            node_1, node_2, node_3, node_4 (Node): Corner Node objects.
            existing_nodes (dict[str, Node]): Map of node IDs to Node instances.
            elements_db (StructuralElementsDB): Database for retrieving floor
                elements.
            building_function (str): 'Logement', 'Bureau', or 'Gare'.
            calc_secondary_beams (bool): If True, calculate secondary beams

        Raises:
            ValueError: If any of the provided corner Nodes are not found in
            existing_nodes.
        """

        # Set basic attributes
        self.id = id
        self.typology = typology
        self.surface = surface
        material_mappings = {
            'Beton': 'Concrete',
            'Acier': 'Steel',
            'Bois': 'Timber',
        }
        self.beam_material = material_mappings[beam_material]
        self.span_dir = span_dir # can be 'x', 'y'; or 'xy'
        self.elements_db = elements_db
        self.building_function = building_function
        self.calc_secondary_beams = calc_secondary_beams

        # Find endpoints in list of existing nodes
        p1_found = False
        p2_found = False
        p3_found = False
        p4_found = False
        for node in existing_nodes.values():
            if abs(node.x - node_1.x) < 0.001 \
                and abs(node.y - node_1.y) < 0.001 \
                    and abs(node.z - node_1.z) < 0.001:
                node_a = node.id
                p1_found = True
            elif abs(node.x - node_2.x) < 0.001 \
                and abs(node.y - node_2.y) < 0.001 \
                    and abs(node.z - node_2.z) < 0.001:
                node_b = node.id
                p2_found = True
            elif abs(node.x - node_3.x) < 0.001 \
                and abs(node.y - node_3.y) < 0.001 \
                    and abs(node.z - node_3.z) < 0.001:
                node_c = node.id
                p3_found = True
            elif abs(node.x - node_4.x) < 0.001 \
                and abs(node.y - node_4.y) < 0.001 \
                    and abs(node.z - node_4.z) < 0.001:
                node_d = node.id
                p4_found = True
            if p1_found and p2_found and p3_found and p4_found:
                break
        if not (p1_found and p2_found and p3_found and p4_found):
            raise ValueError(
                f'Error: {self.id} corner(s) not found:\n' \
                + f'    P1 found? {p1_found}\n' \
                + f'    P2 found? {p2_found}\n' \
                + f'    P3 found? {p3_found}\n' \
                + f'    P4 found? {p4_found}\n' \
            )
        
        # Order nodes so that the couples node_1-node_2 and node_3-node_4 define
        # two beams supporitng the floor
        sorted_nodes = sorted(
            [node_a, node_b, node_c, node_d],
            key=lambda n_id: (existing_nodes[n_id].x, existing_nodes[n_id].y)
        )
        if span_dir == 'x':
            self.node_1 = sorted_nodes[0]
            self.node_2 = sorted_nodes[1]
            self.node_3 = sorted_nodes[2]
            self.node_4 = sorted_nodes[3]
        elif span_dir == 'y':
            self.node_1 = sorted_nodes[0]
            self.node_2 = sorted_nodes[2]
            self.node_3 = sorted_nodes[1]
            self.node_4 = sorted_nodes[3]
        
        self.span = existing_nodes[self.node_1].distance_to_node(
            existing_nodes[self.node_3]
        )
        self.width = existing_nodes[self.node_1].distance_to_node(
            existing_nodes[self.node_2]
        )
    
    def __str__(self):
        return f'{self.id}:\n   - node 1: {self.node_1}\n' \
            + f'   - node 2: {self.node_2}\n   - node 3: {self.node_3}\n' \
            + f'   - node 4: {self.node_4}\n'
    
    def __repr__(self):
        return self.id
    
    def design(self):
        """
        Design the floor system by determining loads and selecting floor
        elements.

        Steps:
            1. Determine loads supported by the floor
            2. Determine floor used and if relevant the floor-beam system.
            3. Compute material quantities for the entire floor.

        Attributes set:
            q (float): Live load in kN/m2.
            g_superimposed (float): Superimposed dead load.
            g_floor (float): Dead load from selected floor element.
            g (float): Total dead load (g_floor + g_superimposed).
            height (float): Floor thickness in m.
            elements (dict[str, float]): Material mass in kg.
        """
        # 1. Determine loads supported by floor
        # 1.1. Live loads
        live_loads_specs = { # in kN/m2
            'Logement': 1.5,
            'Bureau': 2.5,
            'Gare': 5
        }
        self.q = live_loads_specs[self.building_function]
        # 1.2. Dead load
        self.g_superimposed = 1.2 # kN/m2 permanent load other than floor

        # 2. Determine floor system to use and calculate total dead load (g)
        # Option A: if secondary beam system is considered, then determine
        # optimal subspan and # interior beams configuration
        if self.calc_secondary_beams:
            # Start by calculating full floor, if span possible
            # (i.e. is 6 or lower as max span is 6m)
            if round(self.span) <= 6:
                # Find adequate floor in DB
                designed_full_floor = self.elements_db.get_floor(
                    self.typology, self.building_function, self.span, self.id
                )
                g_full_floor = designed_full_floor['G']
            # if span is higher than 6 then it is impossible to have a full
            # floor, so we set g_full_floor to +inf so that the subfloor system
            # will be the lightest one (this way it gets chosen)
            else:
                g_full_floor = math.inf

            # Then look at a subdivision of the floor with subspans of 3 and
            # interior beams, if the width of the floor is larger than 3
            if round(self.width) > 3:
                num_subfloors = round(self.width / 3)
                subfloor_span = 3
                subfloor_width = self.span
                # Get designed subfloors and its resulting loading
                designed_subfloor = self.elements_db.get_floor(
                    self.typology, self.building_function,
                    subfloor_span, self.id + '_subfloor'
                )
                g_subfloor = designed_subfloor['G'] # in kN/m2
                
                # Get designed secondary beams
                num_secondary_beams = num_subfloors + 1
                num_edge_beams = 2
                num_interior_beams = num_secondary_beams - num_edge_beams
                # Get designed edge beam (which only supports half a subfloor)
                edge_beam = self.elements_db.get_beam(
                    self.beam_material, subfloor_width,
                    self.q * subfloor_span / 2,
                    (g_subfloor + self.g_superimposed) * subfloor_span / 2,
                    self.id + '_edge_beam'
                )
                # Get designed interior beam (which supports two half subfloors)
                interior_beam = self.elements_db.get_beam(
                    self.beam_material, subfloor_width, self.q * subfloor_span,
                    (g_subfloor + self.g_superimposed) * subfloor_span,
                    self.id + '_interior_beam'
                )

                # Calculate total permanent load (G) of the subfloors and
                # and secondary beams, and average the load as kN/m2 of total
                # floor area
                g_edge_beam = 0 # in kN/ml
                for material_quantity in edge_beam['Material'].values():
                    g_edge_beam += material_quantity * 10 \
                        / (1000 * subfloor_width)
                g_edge_beam *= num_edge_beams
                g_interior_beam = 0 # in kN/ml
                for material_quantity in interior_beam['Material'].values():
                    g_interior_beam += material_quantity * 10 \
                        / (1000 * subfloor_width)
                g_interior_beam *= num_interior_beams
                # Calc total self weight of subfloor + secondary beams in kN/m2
                # (assuming it is ok to turn beams point loads into distributed
                # loads)
                g_floor_beam_system = g_subfloor + (
                    g_edge_beam + g_interior_beam
                ) * subfloor_width / self.surface
            # if the width is equal to 3, then nop subdivisions are possible
            # -> we pick the full floor. To do so, we set g_floor_beam_system to
            # math.inf
            else:
                g_floor_beam_system = math.inf
            
            # If having a full floor is lighter or equal to having secondary
            # beams, we select the full floor
            if g_full_floor <= g_floor_beam_system:
                self.secondary_beams = False
                self.g_floor = g_full_floor
                self.g = self.g_floor + self.g_superimposed
                self.height = designed_full_floor['Height'] * 0.01 # turn cm into m
                self.g_subfloor = 0
                self.num_subfloors = 0
                self.subfloor_span = 0
                self.subfloor_width = 0
                self.g_edge_beam = 0
                self.g_interior_beam = 0
                self.num_edge_beams = 0
                self.num_interior_beams = 0
                self.edge_beam_section = 0
                self.interior_beam_section = 0
            # Otherwise, we select the system with secondary beams, and we store
            # all relevant information
            else:
                self.secondary_beams = True
                self.g_floor = g_floor_beam_system
                self.g = self.g_floor + self.g_superimposed
                self.height = designed_subfloor['Height'] * 0.01 #turn cm into m
                self.g_subfloor = g_subfloor
                self.num_subfloors = num_subfloors
                self.subfloor_span = subfloor_span
                self.subfloor_width = subfloor_width
                self.g_edge_beam = g_edge_beam
                self.g_interior_beam = g_interior_beam
                self.num_edge_beams = num_edge_beams
                self.num_interior_beams = num_interior_beams
                self.edge_beam_section = edge_beam['Section']
                self.interior_beam_section = interior_beam['Section']
        
        # Option B: if no secondary beam system, simply design floor itself,
        # with self.g = g_floor + g_superimposed
        else:
            self.secondary_beams = False
            # Find adequate floor in DB
            designed_floor = self.elements_db.get_floor(
                self.typology, self.building_function, self.span, self.id
            )
            self.g_floor = designed_floor['G']
            self.g = self.g_floor + self.g_superimposed
            self.height = designed_floor['Height'] * 0.01 # turn cm into m
            self.g_subfloor = 0
            self.num_subfloors = 0
            self.subfloor_span = 0
            self.subfloor_width = 0
            self.g_edge_beam = 0
            self.g_interior_beam = 0
            self.num_edge_beams = 0
            self.num_interior_beams = 0
            self.edge_beam_section = 0
            self.interior_beam_section = 0
        
        # 3. Get elements composing the floor system, with material quantities
        if self.calc_secondary_beams:
            # treat materials differently dependingh on type of floor system
            if self.secondary_beams:
                # store material in each subfloor
                self.elements = {
                    material_type: quantity * self.surface \
                        for material_type, quantity \
                            in designed_subfloor['Material'].items()
                } # in kg
                # store material in edge beams
                for material_type, quantity in edge_beam['Material'].items():
                    if material_type not in self.elements.keys():
                        self.elements[material_type] = quantity \
                            * self.num_edge_beams
                    else:
                        self.elements[material_type] += quantity \
                            * self.num_edge_beams
                # store material in interior beams
                for material_type, quantity in interior_beam['Material'].items():
                    if material_type not in self.elements.keys():
                        self.elements[material_type] = quantity \
                            * self.num_interior_beams
                    else:
                        self.elements[material_type] += quantity \
                            * self.num_interior_beams
                
            else:
                # store total amount of material (convert from kg/m2 to kg)
                self.elements = {
                    material_type: quantity * self.surface \
                        for material_type, quantity \
                            in designed_full_floor['Material'].items()
                } # in kg
        else:
            # store total amount of material (convert from kg/m2 to kg)
            self.elements = {
                material_type: quantity * self.surface \
                    for material_type, quantity \
                        in designed_floor['Material'].items()
            } # in kg

class PrincipalBeam:
    """
    Represents a primary beam in a structural model, defining its geometry,
    connected floors, and design actions.

    Attributes:
        id (str): Unique identifier for the beam.
        material (str): Beam material ('Concrete', 'Steel', or 'Timber').
        length (float): Beam length in meters.
        startpoint (str): Node ID at the start of the beam.
        endpoint (str): Node ID at the end of the beam.
        elements_db (object): Database interface for retrieving beam elements.
        connected_floors (list[str]): IDs of floors intersecting this beam.
        supported_floors (list[str]): IDs of floors this beam supports based on
            span direction.
        supported_floors_side_1 (list[str]): IDs of supported floors on one side
        supported_floors_side_2 (list[str]): IDs of supported floors on the
            other side.
        g_superimposed_side_1 (float): Superimposed dead load from side 1 in kN.
        q_side_1 (float): Live load from side 1 in kN.
        g_superimposed_side_2 (float): Superimposed dead load from side 2 in kN.
        q_side_2 (float): Live load from side 2 in kN.
        g_superimposed (float): Total superimposed dead load in kN.
        q (float): Total live load in kN.
        height (float): Beam section height in meters.
        section (any): Beam section identifier.
        g_beam (float): Self-weight of the beam expressed as kN per meter.
        elements (dict): Material quantities in kg for each component of the
            beam.

    Methods:
        get_connected_floors(all_floors, all_nodes):
            Identify all floors intersecting this beam and categorize support.
        design(all_floors):
            Compute loads from connected floors and select/design beam sizing.
    """
    def __init__(self, id, material, length, node_1, node_2, \
                 existing_nodes, elements_db):
        """
        Initialize a PrincipalBeam and determine its endpoints based on Node
        positions.

        Args:
            id (str): Unique beam identifier.
            material (str): Material label ('Beton', 'Acier', 'Bois') mapped
                internally.
            length (float): Beam length in meters.
            node_1, node_2 (Node): Node objects defining beam endpoints.
            existing_nodes (dict[str, Node]): Map of node IDs to Node instances.
            elements_db (object): Database interface for retrieving beam
                elements.

        Raises:
            ValueError: If endpoints are not found in existing_nodes.
        """

        self.id = id
        material_mappings = {
            'Beton': 'Concrete',
            'Acier': 'Steel',
            'Bois': 'Timber',
        }
        self.material = material_mappings[material]
        self.length = length
        self.elements_db = elements_db

        # Find endpoints in list of existing nodes
        p1_found = False
        p2_found = False
        for node in existing_nodes.values():
            if abs(node.x - node_1.x) < 0.001 \
                and abs(node.y - node_1.y) < 0.001 \
                    and abs(node.z - node_1.z) < 0.001:
                node_1_id = node.id
                p1_found = True
            elif abs(node.x - node_2.x) < 0.001 \
                and abs(node.y - node_2.y) < 0.001 \
                    and abs(node.z - node_2.z) < 0.001:
                node_2_id = node.id
                p2_found = True
            if p1_found and p2_found:
                break
        if not (p1_found and p2_found):
            raise ValueError(
                'Error: Beam endpoint(s) not found:\n' \
                + '    P1 found? {:b}\n'.format(p1_found) \
                + '    P2 found? {:b}\n'.format(p2_found)
            ) 

        # Arrange points in ascending order based on x/y coordinates
        if node_2.x >= node_1.x and node_2.y >= node_1.y:
            self.startpoint = node_1_id
            self.endpoint = node_2_id
        else:
            self.startpoint = node_2_id
            self.endpoint = node_1_id
    
    def __str__(self):
        return f'{self.id}:\n   - startpoint: {self.startpoint}\n' \
            + f'   - endpoint: {self.endpoint}'
    
    def __repr__(self):
        return self.id
            
    def get_connected_floors(self, all_floors, all_nodes):
        """
        Identify floor segments coincident with this beam and classify supported
        floors.

        Args:
            all_floors (dict[str, object]): Map of floor IDs to floor objects.
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
        """
        self.connected_floors = list(set(
            [
                floor.id for floor in all_floors.values() \
                    if (
                        segments_coincident(
                            all_nodes[self.startpoint],
                            all_nodes[self.endpoint],
                            all_nodes[floor.node_1],
                            all_nodes[floor.node_2]
                        )
                        or
                        segments_coincident(
                            all_nodes[self.startpoint],
                            all_nodes[self.endpoint],
                            all_nodes[floor.node_2],
                            all_nodes[floor.node_4]
                        )
                        or
                        segments_coincident(
                            all_nodes[self.startpoint],
                            all_nodes[self.endpoint],
                            all_nodes[floor.node_4],
                            all_nodes[floor.node_3]
                        )
                        or
                        segments_coincident(
                            all_nodes[self.startpoint],
                            all_nodes[self.endpoint],
                            all_nodes[floor.node_3],
                            all_nodes[floor.node_1]
                        )
                    )
            ]
        ))

        # Determine which connected floors are supported by the beams
        self.supported_floors = []
        for floor_id in self.connected_floors:
            floor = all_floors[floor_id]
            if floor.span_dir == 'x' and abs(
                all_nodes[self.startpoint].x - all_nodes[self.endpoint].x
            ) < 0.001:
                self.supported_floors.append(floor.id)
            elif floor.span_dir == 'y' and abs(
                all_nodes[self.startpoint].y - all_nodes[self.endpoint].y
            ) < 0.001:
                self.supported_floors.append(floor.id)
            elif floor.span_dir == 'xy':
                self.supported_floors.append(floor.id)
        
        # Separate supported floors based on which side of the beam they're from
        self.supported_floors_side_1 = []
        self.supported_floors_side_2 = []
        for floor_id in self.supported_floors:
            floor = all_floors[floor_id]
            floor_centroid = Node(
                -1,
                (
                        all_nodes[floor.node_1].x + all_nodes[floor.node_2].x \
                        + all_nodes[floor.node_3].x + all_nodes[floor.node_4].x
                )/4,
                (
                        all_nodes[floor.node_1].y + all_nodes[floor.node_2].y \
                        + all_nodes[floor.node_3].y + all_nodes[floor.node_4].y
                )/4,
                (
                        all_nodes[floor.node_1].z + all_nodes[floor.node_2].z \
                        + all_nodes[floor.node_3].z + all_nodes[floor.node_4].z
                )/4,
            )
            # the 'or' enables accounting for the right dir when floor.span_dir
            # is 'xy'
            if floor.span_dir == 'x' or abs(
                    all_nodes[self.startpoint].x - all_nodes[self.endpoint].x
                ) < 0.001:
                if floor_centroid.x > all_nodes[self.startpoint].x:
                    self.supported_floors_side_1.append(floor_id)
                else:
                    self.supported_floors_side_2.append(floor_id)
            elif floor.span_dir == 'y' or abs(
                    all_nodes[self.startpoint].y - all_nodes[self.endpoint].y
                ) < 0.001:
                if floor_centroid.y > all_nodes[self.startpoint].y:
                    self.supported_floors_side_1.append(floor_id)
                else:
                    self.supported_floors_side_2.append(floor_id)

    def design(self, all_floors):
        """
        Design the beam by aggregating loads from supported floors and selecting
        beam specs.

        Args:
            all_floors (dict[str, object]): Map of floor IDs to floor objects.

        Sets:
            g_superimposed_side_1 (float), q_side_1 (float),
            g_superimposed_side_2 (float), q_side_2 (float),
            g_superimposed (float), q (float), height (float), section, g_beam
                (float), elements (dict).

        Raises:
            ValueError: If supported floors impose varying loads along beam
                length.
        """

        # 1. Determine loading
        # Add load of all supported floors (q_superimposed and g_superimposed)
        # start with side 1
        self.g_superimposed_side_1 = 0
        self.q_side_1 = 0
        for i, floor_id in enumerate(self.supported_floors_side_1):
            floor = all_floors[floor_id]
            # only add first floor in list (there can't be superposed floors ..)
            if i == 0:
                self.g_superimposed_side_1 += floor.g * floor.span / 2
                self.q_side_1 += floor.q * floor.span / 2
            # for the other floors, simple check that the loads are the same
            # because designed beams can only take uniform loads
            elif floor.g * floor.span / 2 != self.g_superimposed_side_1 \
                or floor.q * floor.span / 2 != self.q_side_1:
                raise ValueError(
                    'Error: Beam loaded with varying load along its length!\n' \
                    + f'    Floors responsible: {self.supported_floors_side_1}'
                )
        # Then side 2
        self.g_superimposed_side_2 = 0
        self.q_side_2 = 0
        for i, floor_id in enumerate(self.supported_floors_side_2):
            floor = all_floors[floor_id]
            # only add first floor in list (there can't be superposed floors ..)
            if i == 0:
                self.g_superimposed_side_2 += floor.g * floor.span / 2
                self.q_side_2 += floor.q * floor.span / 2
            # for the other floors, simple check that the loads are the same
            # because designed beams can only take uniform loads
            elif floor.g * floor.span / 2 != self.g_superimposed_side_2 \
                or floor.q * floor.span / 2 != self.q_side_2:
                raise ValueError(
                    'Error: Beam loaded with varying load along its length!\n' \
                    + f'    Floors responsible: {self.supported_floors_side_2}'
                )
        # Sum loads from both sides
        self.g_superimposed = self.g_superimposed_side_1 \
            + self.g_superimposed_side_2
        self.q = self.q_side_1 + self.q_side_2
        
        # 2. find fitting beam based on loads
        beam = self.elements_db.get_beam(
            self.material, self.length, self.q, self.g_superimposed, self.id
        )
        self.height = beam['Height'] * 0.001 # turn mm into m
        self.section = beam['Section'] # keep mm, as is standard

        # 3. Determine self-weight of beam by getting each element mass and
        # converting it to kN/ml
        self.g_beam = 0
        for material_quantity in beam['Material'].values():
            self.g_beam += material_quantity * 10 / (1000 * self.length)
        self.g = self.g_superimposed + self.g_beam

        # 4. get elements composing the beams with material quantities
        self.elements = beam['Material'] # in kg

class Column:
    """
    Represents a vertical column in a structural model, managing connectivity
    and loading design.

    Attributes:
        id (str): Unique identifier for the column.
        material (str): Column material ('Concrete', 'Steel', or 'Timber').
        length (float): Column length in meters.
        startpoint (str): Node ID at the base of the column.
        endpoint (str): Node ID at the top of the column.
        elements_db (StructuralElementsDB): Database interface for retrieving
            column specs.
        connected_beams (list[str]): IDs of beams connected at base or top nodes
        supported_beams (list[str]): IDs of beams supported by the top node.
        connected_columns (list[str]): IDs of adjacent columns at same nodes.
        supported_columns (list[str]): IDs of columns supported above the top
            node.
        g_upper_floors (float): Combined permanent load from floors above in kN.
        q_upper_floors (float): Combined live load from floors above in kN.
        q_upper_floors_reduced (float): Reduced live load per code if >2 floors.
        g_level (float): Permanent load from current level beams in kN.
        q_level (float): Live load from current level beams in kN.
        g_superimposed (float): Total permanent load on column in kN.
        q (float): Total live load on column in kN.
        q_reduced (float): Total reduced live load on column in kN.
        n_uls (float): Ultimate limit state axial load in kN.
        width (float): Column thickness in meters.
        section (any): Column section identifier.
        g_column (float): Self-weight of column in kN.
        elements (dict): Material quantities in kg for the column.
    """
    def __init__(self, id, material, length, node_1, node_2, \
                 existing_nodes, elements_db):
        """
        Initialize a Column and determine its base and top node IDs.

        Args:
            id (str): Unique column identifier.
            material (str): Material label ('Beton', 'Acier', 'Bois') mapped
                internally.
            length (float): Column length in meters.
            node_1, node_2 (Node): Node objects defining column endpoints.
            existing_nodes (dict[str, Node]): Map of node IDs to Node instances.
            elements_db (StructuralElementsDB): Database interface for
                retrieving column specs.

        Raises:
            ValueError: If endpoints are not found in existing_nodes.
        """

        self.id = id
        material_mappings = {
            'Beton': 'Concrete',
            'Acier': 'Steel',
            'Bois': 'Timber',
        }
        self.material = material_mappings[material]
        self.length = length
        self.elements_db = elements_db
        
        # Find endpoints in list of existing nodes
        p1_found = False
        p2_found = False
        for node in existing_nodes.values():
            if abs(node.x - node_1.x) < 0.001 \
                and abs(node.y - node_1.y) < 0.001 \
                    and abs(node.z - node_1.z) < 0.001:
                node_1_id = node.id
                p1_found = True
            elif abs(node.x - node_2.x) < 0.001 \
                and abs(node.y - node_2.y) < 0.001 \
                    and abs(node.z - node_2.z) < 0.001:
                node_2_id = node.id
                p2_found = True
            if p1_found and p2_found:
                break
        if not (p1_found and p2_found):
            raise ValueError(
                'Error: Column endpoint(s) not found:\n' \
                + '    P1 found? {:b}\n'.format(p1_found) \
                + '    P2 found? {:b}\n'.format(p2_found)
            ) 

        # Arrange points in ascending order based on z coordinate
        if node_2.z > node_1.z:
            self.startpoint = node_1_id
            self.endpoint = node_2_id
        else:
            self.startpoint = node_2_id
            self.endpoint = node_1_id
    
    def __str__(self):
        return f'{self.id}:\n   - startpoint: {self.startpoint}\n' \
            + f'   - endpoint: {self.endpoint}'
    
    def __repr__(self):
        return self.id
    
    def get_connected_beams(self, all_nodes):
        """
        Identify beams connected to this column at its base and top nodes.

        Args:
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
        """

        # Get all beams connected to the top point
        self.connected_beams = list(
            set(
                [
                    beam_id for beam_id in \
                        all_nodes[self.startpoint].connected_beams \
                            + all_nodes[self.endpoint].connected_beams
                ]
            )
        )

        # and beams actually supported by the column
        self.supported_beams = [
            beam_id for beam_id in all_nodes[self.endpoint].connected_beams
        ]

    def get_connected_columns(self, all_nodes):
        """
        Identify adjacent columns connected at the same nodes.

        Args:
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
        """
        # Get all connected columns
        self.connected_columns = list(
            set(
                [
                    column_id for column_id in \
                        all_nodes[self.startpoint].connected_columns \
                            + all_nodes[self.endpoint].connected_columns if \
                                (column_id != self.id)
                ]
            )
        )

        # and columns actually supported by the column
        self.supported_columns = [
            column_id for column_id in \
                all_nodes[self.endpoint].connected_columns if \
                    (column_id != self.id)
        ]
            
    def design(self, all_columns, all_beams):
        """
        Design the column by aggregating loads from supported beams and columns
        above.

        Args:
            all_columns (dict[str, Column]): Map of column IDs to Column
                instances.
            all_beams (dict[str, PrincipalBeam]): Map of beam IDs to
                PrincipalBeam instances.

        Sets:
            n_upper_floors (int), g_upper_floors (float), q_upper_floors (float)
            q_upper_floors_reduced (float), g_level (float), q_level (float),
            g_superimposed (float), q (float), q_reduced (float), n_uls (float),
            width (float), section, g_column (float), elements (dict).
        """

        # 1. Calculate loading
        # 1.1. Start with loading from upper floors

        # Determine number of upper floors
        ref_column = self
        n_upper_floors = 0
        while True:
            if len(ref_column.supported_columns) == 0:
                break
            for column_id in ref_column.supported_columns:
                n_upper_floors += 1
                ref_column = all_columns[column_id]
                break
        
        # Get g and q from upper floors
        if n_upper_floors != 0:
            for column_id in self.supported_columns:
                column = all_columns[column_id]
                self.g_upper_floors = column.g
                self.q_upper_floors = column.q
        else:
            self.g_upper_floors = 0.
            self.q_upper_floors = 0.
        
        # if n_upper_floors > 2, apply reductio factor to q_upper_floors
        if n_upper_floors > 2:
            # Formula in NF EN 1991-1-1 §6.3.1.2(11):
            # alpha_n = (2 + (n - 2) * psi_0)/n
            # where n = number of floors (n > 2)
            # psi_0 comes from NF EN 1990-1-1 Annexe A1 Table A1.1
            # psi_0 = 0.7 for both housing (category A) and offices (category B)
            self.q_upper_floors_reduced = self.q_upper_floors * \
                ((2 + (n_upper_floors - 2) * 0.7)/n_upper_floors)
        else:
            self.q_upper_floors_reduced = deepcopy(self.q_upper_floors)

        # 1.2. Calculate loading from this floor
        # Permanent loads
        self.g_level = 0
        self.q_level = 0
        # iterate over beams connected to the top of the column and get loads
        for beam_id in self.supported_beams:
            beam = all_beams[beam_id]
            self.g_level += beam.g * beam.length / 2
            self.q_level += beam.q * beam.length / 2
        
        self.g_superimposed = self.g_upper_floors + self.g_level
        self.q = self.q_upper_floors + self.q_level
        self.q_reduced = self.q_upper_floors_reduced + self.q_level
        
        # Load combination, according to NF EN 1990/NA §6.4.3.2(3):
        # E_d = gamma_g * G + gamma_q * Q
        # where gamma_g = 1.35 and gamma_q = 1.5 according to NF EN 1990/NA
        # Table A1.2(B)(NF) for ULS (STR)
        self.n_uls = 1.35 * self.g_superimposed + 1.5 * self.q_reduced
        
        # 2. Find a fitting column based on load, material and geometry
        column = self.elements_db.get_column(
            self.material, self.length, self.n_uls, self.id
        )
        self.width = column['Column thickness'] * 0.001 # turn mm to m
        self.section = column['Section'] # keep mm as is standard

        # 3. Determine self-weight of column by getting each element mass and
        # converting it to kN
        self.g_column = 0
        for material_quantity in column['Material'].values():
            self.g_column += material_quantity * 10 / 1000
        self.g = self.g_superimposed + self.g_column

        # 4. get elements composing the columns with material quantities
        self.elements = column['Material'] # in kg

class ShearWall:
    """
    Represents a shear wall in a structural model, managing geometry,
    connectivity, and load-based design.

    Attributes:
        id (str): Unique wall identifier.
        material (str): Wall material label (e.g., 'Beton', 'Acier', 'Bois').
        node_1, node_2, node_3, node_4 (str): Node IDs defining the wall corners
        length (float): Wall horizontal length in meters.
        height (float): Wall vertical height in meters.
        elements_db (StructuralElementsDB): Interface to retrieve shear wall
            specs.
        connected_nodes (list[str]): IDs of nodes lying along the top edge of
            the wall.
        connected_beams (list[str]): IDs of beams connected to the wall top
            corners.
        connected_floors (list[str]): IDs of floors intersecting the wall top
            edge.
        supported_floors (list[str]): IDs of floors this wall supports based on
            span direction.
        supported_floors_side_1 (list[str]): Supported floor IDs on one side of
            the wall.
        supported_floors_side_2 (list[str]): Supported floor IDs on the other
            side of the wall.
        g_upper_floors (float): Combined permanent load from walls above in kN.
        q_upper_floors (float): Combined live load from walls above in kN.
        q_upper_floors_reduced (float): Reduced live load per code if >2 walls
            above.
        g_level_side_1 (float): Permanent load from floors on side 1 in kN.
        q_level_side_1 (float): Live load from floors on side 1 in kN.
        g_level_side_2 (float): Permanent load from floors on side 2 in kN.
        q_level_side_2 (float): Live load from floors on side 2 in kN.
        g_level (float): Total permanent load from floors on both sides in kN.
        q_level (float): Total live load from floors on both sides in kN.
        g_superimposed (float): Total permanent load on the wall in kN.
        q (float): Total live load on the wall in kN.
        q_reduced (float): Total reduced live load on the wall in kN.
        thickness (float): Wall thickness in meters.
        g_wall (float): Self-weight of the wall expressed as kN per meter length
        elements (dict): Material quantities in kg for the wall.
    """
    def __init__(self, id, material, node_1, node_2, node_3, node_4,
                 existing_nodes, elements_db):
        """
        Initialize a ShearWall and determine its corner node IDs and dimensions.

        Args:
            id (str): Unique wall identifier.
            material (str): Material label (e.g., 'Beton', 'Acier', 'Bois').
            node_1, node_2, node_3, node_4 (Node): Corner Node objects.
            existing_nodes (dict[str, Node]): Map of node IDs to Node instances.
            elements_db (StructuralElementsDB): Interface for wall specs
                retrieval.

        Raises:
            ValueError: If any of the provided corner nodes are not found in
                existing_nodes.
        """
        self.id = id
        self.material = material
        self.elements_db = elements_db

        # Find endpoints in list of existing nodes
        p1_found = False
        p2_found = False
        p3_found = False
        p4_found = False
        for node in existing_nodes.values():
            if abs(node.x - node_1.x) < 0.001 \
                and abs(node.y - node_1.y) < 0.001 \
                    and abs(node.z - node_1.z) < 0.001:
                node_a = node.id
                p1_found = True
            elif abs(node.x - node_2.x) < 0.001 \
                and abs(node.y - node_2.y) < 0.001 \
                    and abs(node.z - node_2.z) < 0.001:
                node_b = node.id
                p2_found = True
            elif abs(node.x - node_3.x) < 0.001 \
                and abs(node.y - node_3.y) < 0.001 \
                    and abs(node.z - node_3.z) < 0.001:
                node_c = node.id
                p3_found = True
            elif abs(node.x - node_4.x) < 0.001 \
                and abs(node.y - node_4.y) < 0.001 \
                    and abs(node.z - node_4.z) < 0.001:
                node_d = node.id
                p4_found = True
            if p1_found and p2_found and p3_found and p4_found:
                break
        if not (p1_found and p2_found and p3_found and p4_found):
            raise ValueError(
                'Error: Shear wall corner(s) not found:\n' \
                + '    P1 found? {:b}\n'.format(p1_found) \
                + '    P2 found? {:b}\n'.format(p2_found) \
                + '    P3 found? {:b}\n'.format(p3_found) \
                + '    P4 found? {:b}\n'.format(p4_found) \
            )
        
        # Sort nodes so that the two first ones are the bottom ones
        # Sort nodes by z coordinate ascending (lowest first)
        sorted_nodes = sorted(
            [node_a, node_b, node_c, node_d],
            key=lambda n_id: existing_nodes[n_id].z
        )
        bottom_nodes = sorted(
            sorted_nodes[:2],
            key=lambda n_id: (existing_nodes[n_id].x, existing_nodes[n_id].y)
        )
        top_nodes = sorted(
            sorted_nodes[2:],
            key=lambda n_id: (existing_nodes[n_id].x, existing_nodes[n_id].y)
        )
        self.node_1 = bottom_nodes[0]
        self.node_2 = bottom_nodes[1]
        self.node_3 = top_nodes[0]
        self.node_4 = top_nodes[1]
        
        self.length = existing_nodes[self.node_1].distance_to_node(
            existing_nodes[self.node_2]
        )
        self.height = existing_nodes[self.node_1].distance_to_node(
            existing_nodes[self.node_3]
        )

    def __str__(self):
        return f'{self.id}:\n   - node 1: {self.node_1}\n' \
            + f'   - node 2: {self.node_2}\n   - node 3: {self.node_3}\n' \
            + f'   - node 4: {self.node_4}\n'
    
    def __repr__(self):
        return self.id
    
    def get_connected_nodes(self, all_nodes):
        """
        Identify nodes lying along the top edge of the wall.

        Args:
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
        """
        self.connected_nodes = [
            node_id for node_id, node in all_nodes.items() \
                if is_point_on_line(
                    node, all_nodes[self.node_3], all_nodes[self.node_4]
                )
        ]
    
    def get_connected_beams(self, all_nodes):
        """
        Identify beams connected to the wall's top corners.

        Args:
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
        """
        # Get all beams connected to the top points
        self.connected_beams = list(
            set(
                [
                    beam_id for beam_id in \
                        all_nodes[self.node_3].connected_beams \
                        + all_nodes[self.node_4].connected_beams
                ]
            )
        )

        # TODO: Get beams acting as point load in te middle of wall
        # But for now we're placing a column under each beam -> no point load

    def get_connected_floors(self, all_floors, all_nodes):
        """
        Identify floors intersecting the wall top edge and classify support.

        Args:
            all_floors (dict[str, object]): Map of floor IDs to floor objects.
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
        """

        # Get all floors that has one edge coincident for a portion with top of
        # wall. However, don't take those that have only one point in common
        self.connected_floors = list(set(
            [
                floor.id for floor in all_floors.values() \
                    if (
                        segments_coincident(
                            all_nodes[self.node_3], all_nodes[self.node_4],
                            all_nodes[floor.node_1], all_nodes[floor.node_2]
                        )
                        or
                        segments_coincident(
                            all_nodes[self.node_3], all_nodes[self.node_4],
                            all_nodes[floor.node_2], all_nodes[floor.node_4]
                        )
                        or
                        segments_coincident(
                            all_nodes[self.node_3], all_nodes[self.node_4],
                            all_nodes[floor.node_4], all_nodes[floor.node_3]
                        )
                        or
                        segments_coincident(
                            all_nodes[self.node_3], all_nodes[self.node_4],
                            all_nodes[floor.node_3], all_nodes[floor.node_1]
                        )
                    )
            ]
        ))

        # Determine which connected floors are supported by the shear wall
        self.supported_floors = []
        for floor_id in self.connected_floors:
            floor = all_floors[floor_id]
            if floor.span_dir == 'x' and all_nodes[self.node_3].y \
                != all_nodes[self.node_4].y:
                self.supported_floors.append(floor.id)
            elif floor.span_dir == 'y' and all_nodes[self.node_3].x \
                != all_nodes[self.node_4].x:
                self.supported_floors.append(floor.id)
            elif floor.span_dir == 'xy':
                self.supported_floors.append(floor.id)
        
        # Separate supported floors based on which side of the wall they're from
        self.supported_floors_side_1 = []
        self.supported_floors_side_2 = []
        for floor_id in self.supported_floors:
            floor = all_floors[floor_id]
            floor_centroid = Node(
                -1,
                (
                        all_nodes[floor.node_1].x + all_nodes[floor.node_2].x \
                        + all_nodes[floor.node_3].x + all_nodes[floor.node_4].x
                )/4,
                (
                        all_nodes[floor.node_1].y + all_nodes[floor.node_2].y \
                        + all_nodes[floor.node_3].y + all_nodes[floor.node_4].y
                )/4,
                (
                        all_nodes[floor.node_1].z + all_nodes[floor.node_2].z \
                        + all_nodes[floor.node_3].z + all_nodes[floor.node_4].z
                )/4,
            )
            # the 'or' enables accounting for the right dir when floor.span_dir
            # is 'xy'
            if floor.span_dir == 'x' or abs(
                    all_nodes[self.node_3].x - all_nodes[self.node_4].x
                ) < 0.001:
                if floor_centroid.x > all_nodes[self.node_3].x:
                    self.supported_floors_side_1.append(floor_id)
                else:
                    self.supported_floors_side_2.append(floor_id)
            elif floor.span_dir == 'y' or abs(
                    all_nodes[self.node_3].x - all_nodes[self.node_4].x
                ) < 0.001:
                if floor_centroid.y > all_nodes[self.node_3].y:
                    self.supported_floors_side_1.append(floor_id)
                else:
                    self.supported_floors_side_2.append(floor_id)
    
    def get_connected_walls(self, all_nodes):
        """
        Identify adjacent shear walls connected at the top edge corners.

        Args:
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
        """
        # Get all walls connected to the top points
        self.connected_walls = list(
            set(
                [
                    wall_id for wall_id in \
                        all_nodes[self.node_3].connected_walls \
                        + all_nodes[self.node_4].connected_walls \
                            if (wall_id != self.id)
                ]
            )
        )

        self.supported_walls = [
            wall_id for wall_id in all_nodes[self.node_3].connected_walls if (
                (wall_id in all_nodes[self.node_4].connected_walls)
                and (wall_id != self.id)
            )
        ]
    
    def design(self, all_walls, all_floors):
        """
        Design the shear wall by aggregating floor loads and selecting wall
        thickness.

        Args:
            all_walls (dict[str, ShearWall]): Map of wall IDs to ShearWall
                instances.
            all_floors (dict[str, object]): Map of floor IDs to floor objects.
        """
        # 1. Calculate loading
        # 1.1. Start with loading from upper floors
        # Determine number of upper floors
        ref_wall = self
        n_upper_floors = 0
        while True:
            if len(ref_wall.supported_walls) == 0:
                break
            for wall_id in ref_wall.supported_walls:
                n_upper_floors += 1
                ref_wall = all_walls[wall_id]
        
        # Get g and q from upper floors
        if n_upper_floors != 0:
            for wall_id in self.supported_walls:
                wall = all_walls[wall_id]
                self.g_upper_floors = wall.g
                self.q_upper_floors = wall.q
        else:
            self.g_upper_floors = 0.
            self.q_upper_floors = 0.
        
        # if n_upper_floors > 2, apply reduction factor to q_upper_floors
        if n_upper_floors > 2:
            # Formula in NF EN 1991-1-1 §6.3.1.2(11):
            # alpha_n = (2 + (n - 2) * psi_0)/n
            # where n = number of floors (n > 2)
            # psi_0 comes from NF EN 1990-1-1 Annexe A1 Table A1.1
            # psi_0 = 0.7 for both housing (category A) and offices (category B)
            self.q_upper_floors_reduced = self.q_upper_floors * \
                ((2 + (n_upper_floors - 2) * 0.7)/n_upper_floors)
        else:
            self.q_upper_floors_reduced = deepcopy(self.q_upper_floors)

        # 1.2. Calculate loading from this floor
        # side 1
        self.g_level_side_1 = 0
        self.q_level_side_1 = 0
        for i, floor_id in enumerate(self.supported_floors_side_1):
            floor = all_floors[floor_id]
            # only add first floor in list (there can't be superposed floors ..)
            if i == 0:
                self.g_level_side_1 += floor.g * floor.span / 2
                self.q_level_side_1 += floor.q * floor.span / 2
            # for the other floors, simple check that the loads are the same
            # because designed walls can only take uniform loads
            elif abs(floor.g * floor.span / 2 - self.g_level_side_1) > 0.001 \
                or abs(floor.q * floor.span / 2 - self.q_level_side_1) > 0.001:
                raise ValueError(
                    'Error: Wall loaded with varying load along its length!\n' \
                    + '    Floors responsible: {self.supported_floors_side_1}'
                )
        # side 2
        self.g_level_side_2 = 0
        self.q_level_side_2 = 0
        for i, floor_id in enumerate(self.supported_floors_side_2):
            floor = all_floors[floor_id]
            # only add first floor in list (there can't be superposed floors ..)
            if i == 0:
                self.g_level_side_2 += floor.g * floor.span / 2
                self.q_level_side_2 += floor.q * floor.span / 2
            # for the other floors, simple check that the loads are the same
            # because designed walls can only take uniform loads
            elif abs(floor.g * floor.span / 2 - self.g_level_side_2) > 0.001 \
                or abs(floor.q * floor.span / 2 - self.q_level_side_2) > 0.001:
                raise ValueError(
                    'Error: Wall loaded with varying load along its length!\n' \
                    + '    Floors responsible: {self.supported_floors_side_2}'
                )
        # sum loads from the two sides
        self.g_level = self.g_level_side_1 + self.g_level_side_2
        self.q_level = self.q_level_side_1 + self.q_level_side_2
        
        self.g_superimposed = self.g_upper_floors + self.g_level
        self.q = self.q_upper_floors + self.q_level
        self.q_reduced = self.q_upper_floors_reduced + self.q_level
        
        # 2. Calculate amount of concrete and reinforcement
        wall = self.elements_db.get_shear_wall(
            self.length, self.height, self.material,
            self.g_superimposed, self.q_reduced, 0
        )
        self.thickness = wall['Width'] # in m

        # 3. Determine self-weight of column by getting each element mass and
        # converting it to kN/ml
        self.g_wall = 0
        for material_quantity in wall['Material'].values():
            self.g_wall += material_quantity * 10 / (1000 * self.length)
        self.g = self.g_superimposed + self.g_wall

        # 4. get elements composing the wall with material quantities
        self.elements = wall['Material'] # in kg

class IsolatedFooting:
    """
    Represents an isolated square footing supporting a column, designed based on
    soil capacity and column loads.

    Attributes:
        id (str): Unique identifier for the footing.
        connected_column (str): ID of the column supported by this footing.
        soil (str): Soil type affecting bearing capacity (default 'Argile et
            limons mous').
        supported_foundation_beams (list): supported foundation beams
        n_axial (float): Ultimate axial load on footing in kN.
        width_column (float): Width of the connected column in meters.
        width (float): Width of the square footing in meters.
        height (float): Thickness (height) of the footing in meters.
        volume (float): Concrete volume of the footing in cubic meters.
        elements (dict): Material quantities in kg, keys 'Fondation-Beton' and
            'Fondation-Ferraillage'.
    """
    def __init__(self, id, column, soil='Argile et limons mous'):
        """
        Initialize an IsolatedFooting for a given column and soil type.

        Args:
            id (str): Unique footing identifier.
            column (Column): Column object this footing supports.
            soil (str, optional): Soil type, one of:
                - 'Argile et limons mous'
                - 'Sables et graves compacts'
                - 'Marnes et marno-calcaires compacts'
                Defaults to 'Argile et limons mous'.

        Attributes set:
            connected_column (str): ID of the supported column.
        """
        self.id = id
        self.connected_column = column.id
        self.soil = soil
    
    def __str__(self):
        return f'{self.id}, at the bottom of Column {self.connected_column}'
    
    def __repr__(self):
        return self.id
    
    def get_connected_foundation_beams(self, all_nodes, all_columns):
        """
        Identify foundation beams connected to this footing.

        Args:
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
            all_columns (dict[str, Node]): Map of columns IDs to COLUMN
            instances.
        """

        # Get all foundation beams supported by footing
        self.supported_foundation_beams = list(
            set(
                [
                    foundation_beam_id for foundation_beam_id in \
                        all_nodes[
                            all_columns[self.connected_column].startpoint
                        ].connected_beams
                ]
            )
        )
    
    def design(self, all_columns, all_foundation_beams):
        """
        Design the footing by determining required width, thickness, and
        material quantities.

        Uses the connected column's ultimate load and soil bearing parameters to
        1. Compute ultimate axial load (n_axial) from supported foundation beams
           and column permanent and live loads.
        2. Determine soil bearing capacity factors based on soil type.
        3. Iteratively find minimum square footing width to satisfy bearing
            pressure.
        4. Compute footing thickness (height), concrete volume, and
            reinforcement weight.

        Args:
            all_columns (dict[str, Column]): Mapping of column IDs to Column
                instances.

        Sets:
            n_axial (float): Ultimate axial load in kN.
            width_column (float): Width of the supported column in m.
            width (float): Width of the footing in m.
            height (float): Footing thickness in m.
            volume (float): Concrete volume in m^3.
            elements (dict): Masses in kg: 'Fondation-Beton' and
                'Fondation-Ferraillage'.

        Raises:
            ValueError: If an invalid soil type is provided.
        """
        
        # Get loading coming from foundation beams and floors
        self.g_level = 0
        self.q_level = 0
        for foundation_beam_id in self.supported_foundation_beams:
            foundation_beam = all_foundation_beams[foundation_beam_id]
            self.g_level += foundation_beam.g * foundation_beam.length / 2
            self.q_level += foundation_beam.q * foundation_beam.length / 2
        
        # Get loads from above floors
        self.g_upper_floors = all_columns[self.connected_column].g
        self.q_upper_floors = all_columns[self.connected_column].q
        
        # Get loading coming from abve floors (i.e. from the above column)
        self.n_axial = 1.35 * (self.g_upper_floors + self.g_level) \
            + 1.5 * (self.q_upper_floors + self.q_level)

        self.width_column = all_columns[self.connected_column].width

        # Constants
        concrete_density = 2400 # kg/m3
        p0 = 0
        d = 0
        q0 = 0
        f_global_uls = 1.68
        if self.soil == 'Argile et limons mous':
            pl = 0.7
            kp = 1.30
        elif self.soil == 'Sables et graves compacts':
            pl = 1.5
            kp = 1.75
        elif self.soil == 'Marnes et marno-calcaires compacts':
            pl = 4.5
            kp = 1.68
        else:
            raise ValueError('Error: Invalid soil type')
        pl_etoile = pl - p0
        ple_etoile = pl_etoile
        qnet = kp * ple_etoile

        # Start by determining width of the square footing
        footing_width = 0.5
        width_found = False
        while not width_found:
            val1 = self.n_axial - footing_width**2 * q0 * 1000
            val2 = footing_width**2 * qnet * 1000 / f_global_uls # 1000x for MPa
            if val1 <= val2:
                width_found = True
            else:
                footing_width += 0.1
        self.width = footing_width

        # Determine remaining footing characteristics
        self.height = max(
            ((footing_width - self.width_column)/4) + 0.05, 0.25
        )
        self.volume = self.width**2 * self.height
        concrete_weight = concrete_density * self.volume
        reinforcement_weight = 50 * self.volume #TODO:find where this comes from
        self.elements = {
            'Fondation-Beton': concrete_weight,
            'Fondation-Ferraillage': reinforcement_weight
        } # in kg

class ContinuousFooting:
    """
    Represents a continuous footing supporting a shear wall, designed based on
    soil capacity and wall loads per meter length.

    Attributes:
        id (str): Unique identifier for the footing.
        connected_wall (str): ID of the wall supported by this footing.
        soil (str): Soil type affecting bearing capacity
            (default 'Argile et limons mous').
        connected_foundation_floors: connected foundation floors
        supported_foundation_floors: supported foundation floors
        n_uls (float): Ultimate limit state load on the footing strip in kN per
            meter.
        wall_length (float): Length of the connected wall in meters.
        wall_width (float): Thickness of the connected wall in meters.
        n_axial (float): Equivalent axial load on a strip equal to wall
            thickness in kN.
        width (float): Width of the square strip footing in meters.
        height (float): Thickness of the footing in meters.
        volume (float): Concrete volume of the footing strip in cubic meters.
        elements (dict): Material quantities in kg, keys 'Fondation-Beton' and
            'Fondation-Ferraillage'.
    """
    def __init__(self, id, wall, soil='Argile et limons mous'):
        """
        Initialize a ContinuousFooting for a given wall and soil type.

        Args:
            id (str): Unique footing identifier.
            wall (ShearWall): ShearWall object this footing supports.
            soil (str, optional): Soil type, defaults to 'Argile et limons mous'

        Attributes set:
            connected_wall (str): ID of the supported wall.
        """
        self.id = id
        self.connected_wall = wall.id
        self.soil = soil
    
    def __str__(self):
        return f'{self.id}, at the bottom of Wall {self.connected_wall}'
    
    def __repr__(self):
        return self.id
    
    def get_connected_foundation_floors(
            self, all_foundation_floors, all_nodes, all_walls):
        """
        Identify foundation floor segments coincident with this footing and
        classify supported floors.

        Args:
            all_foundation_floors (dict[str, object]):
                Map of floor IDs to floor objects.
            all_nodes (dict[str, Node]): Map of node IDs to Node instances.
            all_walls (dict[str, Node]): Map of wall IDs to ShearWall instances.
        """
        self.connected_foundation_floors = list(set(
            [
                foundation_floor.id for foundation_floor \
                    in all_foundation_floors.values() \
                        if (
                            segments_coincident(
                                all_nodes[
                                    all_walls[self.connected_wall].node_1
                                ],
                                all_nodes[
                                    all_walls[self.connected_wall].node_2
                                ],
                                all_nodes[foundation_floor.node_1],
                                all_nodes[foundation_floor.node_2]
                            )
                            or
                            segments_coincident(
                                all_nodes[
                                    all_walls[self.connected_wall].node_1
                                ],
                                all_nodes[
                                    all_walls[self.connected_wall].node_2
                                ],
                                all_nodes[foundation_floor.node_2],
                                all_nodes[foundation_floor.node_4]
                            )
                            or
                            segments_coincident(
                                all_nodes[
                                    all_walls[self.connected_wall].node_1
                                ],
                                all_nodes[
                                    all_walls[self.connected_wall].node_2
                                ],
                                all_nodes[foundation_floor.node_4],
                                all_nodes[foundation_floor.node_3]
                            )
                            or
                            segments_coincident(
                                all_nodes[
                                    all_walls[self.connected_wall].node_1
                                ],
                                all_nodes[
                                    all_walls[self.connected_wall].node_2
                                ],
                                all_nodes[foundation_floor.node_3],
                                all_nodes[foundation_floor.node_1]
                            )
                        )
            ]
        ))

        # Determine which connected floors are supported by the footing
        self.supported_foundation_floors = []
        for foundation_floor_id in self.connected_foundation_floors:
            foundation_floor = all_foundation_floors[foundation_floor_id]
            if foundation_floor.span_dir == 'x' \
                and abs(
                    all_nodes[all_walls[self.connected_wall].node_1].x \
                    - all_nodes[all_walls[self.connected_wall].node_2].x
                ) < 0.001:
                self.supported_foundation_floors.append(foundation_floor.id)
            elif foundation_floor.span_dir == 'y' \
                and abs(
                    all_nodes[all_walls[self.connected_wall].node_1].y \
                    - all_nodes[all_walls[self.connected_wall].node_2].y
                ) < 0.001:
                self.supported_foundation_floors.append(foundation_floor.id)
            elif foundation_floor.span_dir == 'xy':
                self.supported_foundation_floors.append(foundation_floor.id)
        
        # Separate supported floors based on which side of the beam they're from
        self.supported_foundation_floors_side_1 = []
        self.supported_foundation_floors_side_2 = []
        for foundation_floor_id in self.supported_foundation_floors:
            foundation_floor = all_foundation_floors[foundation_floor_id]
            foundation_floor_centroid = Node(
                -1,
                (
                    all_nodes[foundation_floor.node_1].x \
                    + all_nodes[foundation_floor.node_2].x \
                    + all_nodes[foundation_floor.node_3].x \
                    + all_nodes[foundation_floor.node_4].x
                )/4,
                (
                    all_nodes[foundation_floor.node_1].y \
                    + all_nodes[foundation_floor.node_2].y \
                    + all_nodes[foundation_floor.node_3].y \
                    + all_nodes[foundation_floor.node_4].y
                )/4,
                (
                    all_nodes[foundation_floor.node_1].z \
                    + all_nodes[foundation_floor.node_2].z \
                    + all_nodes[foundation_floor.node_3].z \
                    + all_nodes[foundation_floor.node_4].z
                )/4,
            )
            # the 'or' enables accounting for the right dir when floor.span_dir
            # is 'xy'
            if foundation_floor.span_dir == 'x' or abs(
                    all_nodes[all_walls[self.connected_wall].node_1].x \
                    - all_nodes[all_walls[self.connected_wall].node_2].x
                ) < 0.001:
                if foundation_floor_centroid.x \
                    > all_nodes[all_walls[self.connected_wall].node_1].x:
                    self.supported_foundation_floors_side_1.append(
                        foundation_floor_id
                    )
                else:
                    self.supported_foundation_floors_side_2.append(
                        foundation_floor_id
                    )
            elif foundation_floor.span_dir == 'y' or abs(
                    all_nodes[all_walls[self.connected_wall].node_1].y \
                    - all_nodes[all_walls[self.connected_wall].node_2].y
                ) < 0.001:
                if foundation_floor_centroid.y \
                    > all_nodes[all_walls[self.connected_wall].node_1].y:
                    self.supported_foundation_floors_side_1.append(
                        foundation_floor_id
                    )
                else:
                    self.supported_foundation_floors_side_2.append(
                        foundation_floor_id
                    )

    def design(self, all_walls, all_foundation_floors):
        """
        Design the footing by determining required width, thickness, and
            material quantities along the wall strip.

        Args:
            all_walls (dict[str, ShearWall]): Mapping of wall IDs to ShearWall
                instances.
            all_foundation_floors (dict[str, ShearWall]): Mapping of foundation
                floor IDs to FloorSystem instances.

        Sets:
            n_uls (float): ULS load in kN/m.
            wall_length (float): Wall length in m.
            wall_width (float): Wall thickness in m.
            n_axial (float): Axial load on strip in kN.
            width (float): Footing strip width in m.
            height (float): Footing thickness in m.
            volume (float): Concrete volume per strip in m^3.
            elements (dict): Masses in kg: 'Fondation-Beton' and
                'Fondation-Ferraillage'.

        Raises:
            ValueError: If an invalid soil type is provided.
        """

        # Get load of all supported foundation floors (q_level and g_level)
        # start with side 1
        self.g_level_side_1 = 0
        self.q_level_side_1 = 0
        for i, foundation_floor_id in \
            enumerate(self.supported_foundation_floors_side_1):
            foundation_floor = all_foundation_floors[foundation_floor_id]
            # only add first floor in list (there can't be superposed floors ..)
            if i == 0:
                self.g_level_side_1 += foundation_floor.g \
                    * foundation_floor.span / 2
                self.q_level_side_1 += foundation_floor.q \
                    * foundation_floor.span / 2
            # for the other floors, simply check that the loads are the same
            # because designed footings can only take uniform loads
            elif (foundation_floor.g * foundation_floor.span / 2) \
                != self.g_level_side_1 or \
                    (foundation_floor.q * foundation_floor.span / 2) \
                        != self.q_level_side_1:
                raise ValueError(
                    'Error: Continuous footing loaded with varying load along '\
                    + 'its length!\n' + '    Floors responsible: '\
                    + f'{self.supported_foundation_floors_side_1}'
                )
        # Then side 2
        self.g_level_side_2 = 0
        self.q_level_side_2 = 0
        for i, foundation_floor_id in \
            enumerate(self.supported_foundation_floors_side_2):
            foundation_floor = all_foundation_floors[foundation_floor_id]
            # only add first floor in list (there can't be superposed floors ..)
            if i == 0:
                self.g_level_side_2 += foundation_floor.g \
                    * foundation_floor.span / 2
                self.q_level_side_2 += foundation_floor.q \
                    * foundation_floor.span / 2
            # for the other floors, simply check that the loads are the same
            # because designed footings can only take uniform loads
            elif (foundation_floor.g * foundation_floor.span / 2) \
                != self.g_level_side_2 or \
                    (foundation_floor.q * foundation_floor.span / 2) \
                        != self.q_level_side_2:
                raise ValueError(
                    'Error: Continuous footing loaded with varying load along '\
                    + 'its length!\n' + '    Floors responsible: '\
                    + f'{self.supported_foundation_floors_side_2}'
                )
        # Sum loads from both sides (kN/ml)
        self.g_level = self.g_level_side_1 + self.g_level_side_2
        self.q_level = self.q_level_side_1 + self.q_level_side_2

        # Get loads from above floors
        self.g_upper_floors = all_walls[self.connected_wall].g
        self.q_upper_floors = all_walls[self.connected_wall].q

        # g and q loads in kN/ml
        self.n_uls = 1.35 * (self.g_upper_floors + self.g_level) \
            + 1.5 * (self.q_upper_floors + self.q_level)
        self.wall_length = all_walls[self.connected_wall].length
        self.wall_width = all_walls[self.connected_wall].thickness
        # Calculate amunt of load on a strip of length wall_width to model
        #  a square column of size 
        self.n_axial = self.n_uls * self.wall_width

       # Constants
        concrete_density = 2400 # kg/m3
        p0 = 0
        d = 0
        q0 = 0
        f_global_uls = 1.68
        if self.soil == 'Argile et limons mous':
            pl = 0.7
            kp = 1.30
        elif self.soil == 'Sables et graves compacts':
            pl = 1.5
            kp = 1.75
        elif self.soil == 'Marnes et marno-calcaires compacts':
            pl = 4.5
            kp = 1.68
        else:
            raise ValueError('Error: Invalid soil type')
        pl_etoile = pl - p0
        ple_etoile = pl_etoile
        qnet = kp * ple_etoile

        # Start by determining width of the square footing
        footing_width = 0.5
        width_found = False
        while not width_found:
            val1 = self.n_axial - footing_width**2 * q0 * 1000
            val2 = footing_width**2 * qnet * 1000 / f_global_uls # 1000x for MPa
            if val1 <= val2:
                width_found = True
            else:
                footing_width += 0.1
        self.width = footing_width
        
        # Determine remaining footing characteristics for one strip
        self.height = max(
            (self.width - self.wall_width)/4 + 0.05,
            0.25
        )
        self.volume = self.width**2 * self.height
        concrete_weight = concrete_density * self.volume
        reinforcement_weight = 50 * self.volume #TODO:find where this comes from
        # Calculate total material quantities for the strip along the wall
        self.elements = {
            'Fondation-Beton': concrete_weight * self.wall_length \
                / self.wall_width,
            'Fondation-Ferraillage': reinforcement_weight * self.wall_length \
                / self.wall_width
        } # in kg

class Structure:
    """
    Encapsulates a complete structural model, including geometry, element
    instantiation, connectivity, and design execution.

    Attributes:
        path (Path): Directory containing parquet files for nodes, decks, beams,
            columns, walls, and foundations.
        id (int): id of structure in samples.parquet
        building_function (str): Use case for live loads
            ('Logement', 'Bureau', 'Gare').
        soil_type (str): Soil classification for foundation design.
        elements_db (StructuralElementsDB): Interface for retrieving element
            specifications.
        all_nodes (dict[str, Node]): Node instances keyed by ID.
        all_columns (dict[str, Column]): Column instances keyed by ID.
        sorted_columns_ids (list[str]): Column IDs sorted top-down by elevation.
        all_beams (dict[str, PrincipalBeam]): Beam instances keyed by ID.
        all_walls (dict[str, ShearWall]): ShearWall instances keyed by ID.
        sorted_walls_ids (list[str]): Wall IDs sorted top-down by elevation.
        all_floors (dict[str, FloorSystem]): FloorSystem instances for levels.
        all_foundation_floors (dict[str, FloorSystem]): FloorSystem instances
            for foundation slabs.
        all_foundation_beams (dict[str, PrincipalBeam]): PrincipalBeam instances
            for grade beams.
        all_isolated_footings (dict[str, IsolatedFooting]): IsolatedFooting
            instances for columns at grade.
        all_continuous_footings (dict[str, ContinuousFooting]):
            ContinuousFooting instances for walls at grade.
        intermediate_floors (dict[str, FloorSystem]): Floors below the roof
            level.
        roof_floors (dict[str, FloorSystem]): Floors at the roof level.
        intermediate_beams (dict[str, PrincipalBeam]): Beams below the roof
            level.
        roof_beams (dict[str, PrincipalBeam]): Beams at the roof level.
        result (dict): Aggregated design outputs for all structural components.
    """
    def __init__(
            self, path, id, building_function, soil_type, elements_db,
            geom_objects=None, params=None
        ):
        """
        Initialize the Structure by loading geometry PARQUETs and instantiating
        elements.

        Args:
            path (Path): Directory path containing PARQUET files:
                'samples.parquet','nodes.parquet', 'decks.parquet',
                'beams.parquet', 'columns.parquet', 'core_walls.parquet',
                'foundation_decks.parquet', 'foundation_beams.parquet'
            id (int): id of structure in samples.parquet
            building_function (str): Live-load classification
                ('Logement', 'Bureau', 'Gare').
            soil_type (str): Soil type for foundation design.
            elements_db (StructuralElementsDB): Database interface for element
                properties.

        Actions:
            - Reads PARQUETs for nodes, decks, beams, columns, shear walls, and
                foundation data.
            - Instantiates Node, Column, PrincipalBeam, ShearWall, FloorSystem,
                IsolatedFooting, and ContinuousFooting objects.
            - Sorts columns and walls by elevation for top-down design.
            - Computes connectivity: which elements connect at each node, and
                element-to-element relations.
        """
        self.path = path
        self.id = id
        self.building_function = building_function
        self.soil_type = soil_type
        self.elements_db = elements_db
        self.geom_objects = geom_objects

        # Load all geometries
        if self.geom_objects is not None:
            nodes = [
                [
                    self.id, f"node_{i}", node.X, node.Y, node.Z
                ] for i, node in enumerate(self.geom_objects["nodes"])
            ]
            columns = [
                [
                    self.id, f"column_{i}", params["column_material"],
                    column[0].DistanceTo(column[1]), column[0].X, column[0].Y,
                    column[0].Z, column[1].X, column[1].Y, column[1].Z
                ] for i, column in enumerate(self.geom_objects["columns"])
            ]
            beams = [
                [
                    self.id, f"beam_{i}", params["beam_material"],
                    beam[0].DistanceTo(beam[1]), beam[0].X, beam[0].Y,
                    beam[0].Z, beam[1].X, beam[1].Y, beam[1].Z
                ] for i, beam in enumerate(self.geom_objects["beams"])
            ]
            walls = [
                [
                    self.id, f"wall_{i}", "Beton",
                    wall.Vertices[0].Location.X,
                    wall.Vertices[0].Location.Y,
                    wall.Vertices[0].Location.Z,
                    wall.Vertices[1].Location.X,
                    wall.Vertices[1].Location.Y,
                    wall.Vertices[1].Location.Z,
                    wall.Vertices[2].Location.X,
                    wall.Vertices[2].Location.Y,
                    wall.Vertices[2].Location.Z,
                    wall.Vertices[3].Location.X,
                    wall.Vertices[3].Location.Y,
                    wall.Vertices[3].Location.Z
                ] for i, wall in enumerate(self.geom_objects["walls"])
            ]
            floors = [
                [
                    self.id, f"floor_{i}", params["floor_material"],
                    params["beam_material"], brep_area(floor),
                    floor.Vertices[0].Location.X,
                    floor.Vertices[0].Location.Y,
                    floor.Vertices[0].Location.Z,
                    floor.Vertices[1].Location.X,
                    floor.Vertices[1].Location.Y,
                    floor.Vertices[1].Location.Z,
                    floor.Vertices[2].Location.X,
                    floor.Vertices[2].Location.Y,
                    floor.Vertices[2].Location.Z,
                    floor.Vertices[3].Location.X,
                    floor.Vertices[3].Location.Y,
                    floor.Vertices[3].Location.Z,
                    self.geom_objects["span_dirs"][i]
                ] for i, floor in enumerate(self.geom_objects["floors"])
            ]
            foundation_floors = [
                [
                    self.id, f"foundation_floor_{i}", "Coulee-En-Place",
                    "Beton", brep_area(floor),
                    floor.Vertices[0].Location.X,
                    floor.Vertices[0].Location.Y,
                    floor.Vertices[0].Location.Z,
                    floor.Vertices[1].Location.X,
                    floor.Vertices[1].Location.Y,
                    floor.Vertices[1].Location.Z,
                    floor.Vertices[2].Location.X,
                    floor.Vertices[2].Location.Y,
                    floor.Vertices[2].Location.Z,
                    floor.Vertices[3].Location.X,
                    floor.Vertices[3].Location.Y,
                    floor.Vertices[3].Location.Z,
                    self.geom_objects["foundation_span_dirs"][i]
                ] for i, floor in enumerate(
                    self.geom_objects["foundation_floors"]
                )
            ]
            foundation_beams = [
                [
                    self.id, f"beam_{i}", "Beton",
                    beam[0].DistanceTo(beam[1]), beam[0].X, beam[0].Y,
                    beam[0].Z, beam[1].X, beam[1].Y, beam[1].Z
                ] for i, beam in enumerate(
                    self.geom_objects["foundation_beams"]
                )
            ]

        else:
            nodes = pd.read_parquet(
                self.path.joinpath('./nodes.parquet'),
                filters=[('sample_id', '==', self.id)]
            )
            nodes = nodes.values.tolist()
            floors = pd.read_parquet(
                self.path.joinpath('./floors.parquet'),
                filters=[('sample_id', '==', self.id)]
            )
            floors = floors.values.tolist()
            beams = pd.read_parquet(
                self.path.joinpath('./beams.parquet'),
                filters=[('sample_id', '==', self.id)]
            )
            beams = beams.values.tolist()
            columns = pd.read_parquet(
                self.path.joinpath('./columns.parquet'),
                filters=[('sample_id', '==', self.id)]
            )
            columns = columns.values.tolist()
            walls = pd.read_parquet(
                self.path.joinpath('./walls.parquet'),
                filters=[('sample_id', '==', self.id)]
            )
            walls = walls.values.tolist()
            foundation_floors = pd.read_parquet(
                self.path.joinpath('./foundation_floors.parquet'),
                filters=[('sample_id', '==', self.id)]
            )
            foundation_floors = foundation_floors.values.tolist()
            foundation_beams = pd.read_parquet(
                self.path.joinpath('./foundation_beams.parquet'),
                filters=[('sample_id', '==', self.id)]
            )
            foundation_beams = foundation_beams.values.tolist()

        # Initialise structural elements
        # Nodes
        self.all_nodes = {
            nodes[i][1]: Node(
                nodes[i][1], nodes[i][2], nodes[i][3], nodes[i][4]
            ) for i in range(len(nodes))
        }
        # Calc levels heights
        self.level_heights = sorted(
            list(
                set(
                    [round(node.z,2) for node in self.all_nodes.values()]
                )
            )
        )
        # Columns
        self.all_columns = {
            columns[i][1]: Column(
                columns[i][1], columns[i][2], columns[i][3],
                Node(-1, columns[i][4], columns[i][5], columns[i][6]),
                Node(-1, columns[i][7], columns[i][8], columns[i][9]),
                self.all_nodes, self.elements_db
            ) for i in range(len(columns))
        }
        # Sort columns ids frm the highest to lowest
        self.sorted_columns_ids = sorted(
            self.all_columns,
            key=lambda cid: self.all_nodes[self.all_columns[cid].endpoint].z,
            reverse=True 
        )
        # beams
        self.all_beams = {
            beams[i][1]: PrincipalBeam(
                beams[i][1], beams[i][2], beams[i][3],
                Node(-1, beams[i][4], beams[i][5], beams[i][6]),
                Node(-1, beams[i][7], beams[i][8], beams[i][9]),
                self.all_nodes, self.elements_db
            ) for i in range(len(beams))
        }
        # Walls
        self.all_walls = {
            walls[i][1]: ShearWall(
                walls[i][1], walls[i][2],
                Node(-1, walls[i][3], walls[i][4], walls[i][5]),
                Node(-1, walls[i][6], walls[i][7], walls[i][8]),
                Node(-1, walls[i][9], walls[i][10], walls[i][11]),
                Node(-1, walls[i][12], walls[i][13], walls[i][14]),
                self.all_nodes, self.elements_db
            ) for i in range(len(walls))
        }
        # Sort columns ids frm the highest to lowest
        self.sorted_walls_ids = sorted(
            self.all_walls,
            key=lambda cid: self.all_nodes[self.all_walls[cid].node_3].z,
            reverse=True 
        )
        # Level floors
        self.all_floors = {
            floors[i][1]: FloorSystem(
                floors[i][1], floors[i][2], floors[i][3],
                floors[i][4], floors[i][17],
                Node(-1, floors[i][5], floors[i][6], floors[i][7]),
                Node(-1, floors[i][8], floors[i][9], floors[i][10]),
                Node(-1, floors[i][11], floors[i][12], floors[i][13]),
                Node(-1, floors[i][14], floors[i][15], floors[i][16]),
                self.all_nodes, self.elements_db, self.building_function, True
            ) for i in range(len(floors))
        }
        # Groundfloors
        # Generate groundfloors as foundations with grade beams
        # and concrete slabs
        
        self.all_foundation_floors = {
            foundation_floors[i][1]: FloorSystem(
                foundation_floors[i][1], foundation_floors[i][2],
                foundation_floors[i][3], foundation_floors[i][4],
                foundation_floors[i][17],
                Node(
                    -1, foundation_floors[i][5], foundation_floors[i][6],
                    foundation_floors[i][7]
                ),
                Node(
                    -1, foundation_floors[i][8], foundation_floors[i][9],
                    foundation_floors[i][10]
                ),
                Node(
                    -1, foundation_floors[i][11], foundation_floors[i][12],
                    foundation_floors[i][13]
                ),
                Node(
                    -1, foundation_floors[i][14], foundation_floors[i][15],
                    foundation_floors[i][16]
                ),
                self.all_nodes, self.elements_db, self.building_function, True
            ) for i in range(len(foundation_floors))
        }
        # ground floors beams
        self.all_foundation_beams = {
            foundation_beams[i][1]: PrincipalBeam(
                foundation_beams[i][1], foundation_beams[i][2],
                foundation_beams[i][3],
                Node(
                    -1, foundation_beams[i][4], foundation_beams[i][5],
                    foundation_beams[i][6]
                ),
                Node(-1, foundation_beams[i][7], foundation_beams[i][8],
                     foundation_beams[i][9]
                ),
                self.all_nodes, self.elements_db
            ) for i in range(len(foundation_beams))
        }
        # Isolated footings
        self.all_isolated_footings = {}
        i = 0
        for column in self.all_columns.values():
            if abs(self.all_nodes[column.startpoint].z - self.level_heights[0])\
                  < 0.001:
                self.all_isolated_footings[f'isolated_footing_{i}'] \
                    = IsolatedFooting(
                        f'isolated_footing_{i}', column, soil=self.soil_type
                    )
                i += 1
        # Continuous footings
        self.all_continuous_footings = {}
        i = 0
        for wall in self.all_walls.values():
            if abs(self.all_nodes[wall.node_1].z - self.level_heights[0])\
                  < 0.001:
                self.all_continuous_footings[f'continuous_footing_{i}'] \
                    = ContinuousFooting(
                        f'continuous_footing_{i}', wall, soil=self.soil_type
                    )
                i += 1
        
        # Compute connectivity between structural elements
        # Node connectivity
        for node in self.all_nodes.values():
            node.get_connected_columns(self.all_columns, self.all_nodes)
            node.get_connected_beams(self.all_beams, self.all_nodes)
            node.get_connected_beams(self.all_foundation_beams, self.all_nodes)
            node.get_connected_floors(self.all_floors, self.all_nodes)
            node.get_connected_floors(
                self.all_foundation_floors, self.all_nodes
            )
            node.get_connected_walls(self.all_walls, self.all_nodes)
        # Beam-to-Floor connectivity
        for beam in self.all_beams.values():
            beam.get_connected_floors(self.all_floors, self.all_nodes)
        for foundation_beam in self.all_foundation_beams.values():
            foundation_beam.get_connected_floors(
                self.all_foundation_floors, self.all_nodes
            )
        # Wall-to-Beam/Floors/Walls connectivity
        for wall in self.all_walls.values():
            wall.get_connected_nodes(self.all_nodes)
            wall.get_connected_beams(self.all_nodes)
            wall.get_connected_floors(self.all_floors, self.all_nodes)
            wall.get_connected_walls(self.all_nodes)
        # Column to Beam/Columns connectivity
        for column in self.all_columns.values():
            column.get_connected_beams(self.all_nodes)
            column.get_connected_columns(self.all_nodes)
        # Isolated Footing to foundation beam connectivity
        for isolated_footing in self.all_isolated_footings.values():
            isolated_footing.get_connected_foundation_beams(
                self.all_nodes, self.all_columns
            )
        # Continuous Footing to foundation floor connectivity
        for continuous_footing in self.all_continuous_footings.values():
            continuous_footing.get_connected_foundation_floors(
                self.all_foundation_floors, self.all_nodes, self.all_walls
            )
    
    def design(self):
        """
        Perform design calculations and material quantity extraction for all
        structural elements.

        Steps:
            1. Design floor systems for all levels and foundation slabs.
            2. Design primary beams at levels and grade beams.
            3. Design vertical columns top-down.
            4. Design shear walls top-down.
            5. Design footings (isolated and continuous) at grade.
            6. Segregate roof vs intermediate floors and beams.
            7. Aggregate results into self.result.
        """
        # Structural elements design and material quantity determination
        # Floors
        for floor in self.all_floors.values():
            floor.design()
        # Beams
        for beam in self.all_beams.values():
            beam.design(self.all_floors)
        # Columns
        for column_id in self.sorted_columns_ids:
            column = self.all_columns[column_id]
            column.design(self.all_columns, self.all_beams)
        # Walls
        for wall_id in self.sorted_walls_ids:
            wall = self.all_walls[wall_id]
            wall.design(self.all_walls, self.all_floors)
        # Foundation floors
        for foundation_floor in self.all_foundation_floors.values():
            foundation_floor.design()
        # Foundation beams
        for foundation_beam in self.all_foundation_beams.values():
            foundation_beam.design(self.all_foundation_floors)
        # Isolated footings
        for isolated_footing in self.all_isolated_footings.values():
            isolated_footing.design(
                self.all_columns, self.all_foundation_beams
            )
        # Continuous footings
        for continuous_footing in self.all_continuous_footings.values():
            continuous_footing.design(
                self.all_walls, self.all_foundation_floors
            )
        
        # Separate floors/beams into roof floors/beams and intermediate
        # floors/beams
        self.intermediate_floors = {}
        self.roof_floors = {}
        self.intermediate_beams = {}
        self.roof_beams = {}
        for floor_id, floor in self.all_floors.items():
            if abs(self.all_nodes[floor.node_1].z - self.level_heights[-1]) \
                < 0.001:
                self.roof_floors[floor_id] = floor
            else:
                self.intermediate_floors[floor_id] = floor
        for beam_id, beam in self.all_beams.items():
            if abs(self.all_nodes[beam.startpoint].z - self.level_heights[-1]) \
                < 0.001:
                self.roof_beams[beam_id] = beam
            else:
                self.intermediate_beams[beam_id] = beam

        self.result = {
            'Nodes': self.all_nodes,
            'Beams': self.all_beams,
            'Beams': self.all_floors,
            'Columns': self.all_columns,
            'Walls': self.all_walls,
            'Foundation floors': self.all_foundation_floors,
            'Foundation beams': self.all_foundation_beams,
            'Isolated footings': self.all_isolated_footings,
            'Continuous footings': self.all_continuous_footings,
            'Roof floors': self.roof_floors,
            'Roof beams': self.roof_beams,
            'Intermediate floors': self.intermediate_floors,
            'Intermediate beams': self.intermediate_beams,
        }
    
    def compute_material_quantities(self):

        all_floor_materials = list(
            set(
                key for floor in {
                    **self.all_floors, **self.all_foundation_floors
                }.values()
                for key in floor.elements.keys()
            )
        )
        floor_data = {
            'Structure id': [],
            'Floor id': [],
            'Typology': [],
            'Span (m)': [],
            'Width (m)': [],
            'Thickness (m)': [],
            'G (kN/m2)': [],
            'Q (kN/m2)': [],
            'G superimposed (kN/m2)': [],
            'G floor (kN/m2)': [],
            'Location': [],
            'Secondary beams?': [],
            'Num. subfloors': [],
            'Subfloor span (m)': [],
            'Subfloor width (m)': [],
            'Num. edge beams': [],
            'Num. interior beams': [],
            'Edge beam section': [],
            'Interior beam section': [],
            'G subfloor (kN/m2)': [],
            'G edge beam (kN/ml)': [],
            'G interior beam (kN/ml)': [],
        }
        for mat in all_floor_materials:
            floor_data[f'{mat} (kg)'] = []
        for floor_id, floor in self.all_floors.items():
            floor_data['Structure id'].append(self.id)
            floor_data['Floor id'].append(floor.id)
            floor_data['Typology'].append(floor.typology)
            floor_data['Span (m)'].append(floor.span)
            floor_data['Width (m)'].append(floor.width)
            floor_data['Thickness (m)'].append(floor.height)
            floor_data['G (kN/m2)'].append(floor.g)
            floor_data['Q (kN/m2)'].append(floor.q)
            floor_data['G superimposed (kN/m2)'].append(floor.g_superimposed)
            floor_data['G floor (kN/m2)'].append(floor.g_floor)
            floor_data['Secondary beams?'].append(floor.secondary_beams)
            floor_data['Num. subfloors'].append(floor.num_subfloors)
            floor_data['Subfloor span (m)'].append(floor.subfloor_span)
            floor_data['Subfloor width (m)'].append(floor.subfloor_width)
            floor_data['Num. edge beams'].append(floor.num_edge_beams)
            floor_data['Num. interior beams'].append(floor.num_interior_beams)
            floor_data['Edge beam section'].append(floor.edge_beam_section)
            floor_data['Interior beam section'].append(
                floor.interior_beam_section
            )
            floor_data['G subfloor (kN/m2)'].append(floor.g_subfloor)
            floor_data['G edge beam (kN/ml)'].append(floor.g_edge_beam)
            floor_data['G interior beam (kN/ml)'].append(floor.g_interior_beam)

            if floor_id in self.roof_floors.keys():
                floor_data['Location'].append('Roof')
            else:
                floor_data['Location'].append('Intermediate')
            
            for mat in all_floor_materials:
                if mat in floor.elements:
                    floor_data[mat + ' (kg)'].append(floor.elements[mat])
                else:
                    floor_data[mat + ' (kg)'].append(0)

        # Foundation floors
        # Save floors
        for floor_id, floor in self.all_foundation_floors.items():
            floor_data['Structure id'].append(self.id)
            floor_data['Floor id'].append(floor.id)
            floor_data['Typology'].append(floor.typology)
            floor_data['Span (m)'].append(floor.span)
            floor_data['Width (m)'].append(floor.width)
            floor_data['Thickness (m)'].append(floor.height)
            floor_data['G (kN/m2)'].append(floor.g)
            floor_data['Q (kN/m2)'].append(floor.q)
            floor_data['G superimposed (kN/m2)'].append(floor.g_superimposed)
            floor_data['G floor (kN/m2)'].append(floor.g_floor)
            floor_data['Location'].append('Foundation')
            floor_data['Secondary beams?'].append(floor.secondary_beams)
            floor_data['Num. subfloors'].append(floor.num_subfloors)
            floor_data['Subfloor span (m)'].append(floor.subfloor_span)
            floor_data['Subfloor width (m)'].append(floor.subfloor_width)
            floor_data['Num. edge beams'].append(floor.num_edge_beams)
            floor_data['Num. interior beams'].append(floor.num_interior_beams)
            floor_data['Edge beam section'].append(floor.edge_beam_section)
            floor_data['Interior beam section'].append(
                floor.interior_beam_section
            )
            floor_data['G subfloor (kN/m2)'].append(floor.g_subfloor)
            floor_data['G edge beam (kN/ml)'].append(floor.g_edge_beam)
            floor_data['G interior beam (kN/ml)'].append(floor.g_interior_beam)

            for mat in all_floor_materials:
                if mat in floor.elements:
                    floor_data[mat + ' (kg)'].append(floor.elements[mat])
                else:
                    floor_data[mat + ' (kg)'].append(0)
        floor_data = pd.DataFrame(floor_data)

        # Save beams
        beam_data = {
            'Structure id': [],
            'Beam id': [],
            'Material type': [],
            'Length (m)': [],
            'Section (mm*mm)': [],
            'G (kN/ml)': [],
            'Q (kN/ml)': [],
            'G superimposed (kN/ml)': [],
            'G beam (kN/ml)': [],
            'Location': []
        }
        all_beam_materials = list(
            set(
                key for beam in {
                    **self.all_beams, **self.all_foundation_beams
                }.values()
                for key in beam.elements.keys()
            )
        )
        for mat in all_beam_materials:
            beam_data[f'{mat} (kg)'] = []
        for beam_id, beam in self.all_beams.items():
            beam_data['Structure id'].append(self.id)
            beam_data['Beam id'].append(beam.id)
            beam_data['Material type'].append(beam.material)
            beam_data['Length (m)'].append(beam.length)
            beam_data['Section (mm*mm)'].append(beam.section)
            beam_data['G (kN/ml)'].append(beam.g)
            beam_data['Q (kN/ml)'].append(beam.q)
            beam_data['G superimposed (kN/ml)'].append(beam.g_superimposed)
            beam_data['G beam (kN/ml)'].append(beam.g_beam)
            if beam_id in self.roof_beams.keys():
                beam_data['Location'].append('Roof')
            else:
                beam_data['Location'].append('Intermediate')

            for mat in all_beam_materials:
                if mat in beam.elements:
                    beam_data[mat + ' (kg)'].append(beam.elements[mat])
                else:
                    beam_data[mat + ' (kg)'].append(0)
        # Foundation beams
        for beam_id, beam in self.all_foundation_beams.items():
            beam_data['Structure id'].append(self.id)
            beam_data['Beam id'].append(beam.id)
            beam_data['Material type'].append(beam.material)
            beam_data['Length (m)'].append(beam.length)
            beam_data['Section (mm*mm)'].append(beam.section)
            beam_data['G (kN/ml)'].append(beam.g)
            beam_data['Q (kN/ml)'].append(beam.q)
            beam_data['G superimposed (kN/ml)'].append(beam.g_superimposed)
            beam_data['G beam (kN/ml)'].append(beam.g_beam)
            beam_data['Location'].append('Foundation')

            for mat in all_beam_materials:
                if mat in beam.elements:
                    beam_data[mat + ' (kg)'].append(beam.elements[mat])
                else:
                    beam_data[mat + ' (kg)'].append(0)
        beam_data = pd.DataFrame(beam_data)

        # Save columns
        column_data = {
            'Structure id': [],
            'Column id': [],
            'Material type': [],
            'Length (m)': [],
            'Section (mm*mm)': [],
            'N ULS (kN)': [],
            'G (kN)': [],
            'Q (kN)': [],
            'Q reduced (kN)': [],
            'G upper floors (kN)': [],
            'Q upper floors (kN)': [],
            'G level (kN)': [],
            'Q level (kN)': [],
            'G superimposed (kN)': [],
            'G column (kN)': []
        }
        all_column_materials = list(
            set(
                key for column in self.all_columns.values()
                for key in column.elements.keys()
            )
        )
        for mat in all_column_materials:
            column_data[f'{mat} (kg)'] = []
        for column_id, column in self.all_columns.items():
            column_data['Structure id'].append(self.id)
            column_data['Column id'].append(column.id)
            column_data['Material type'].append(column.material)
            column_data['Length (m)'].append(column.length)
            column_data['Section (mm*mm)'].append(column.section)
            column_data['N ULS (kN)'].append(column.n_uls)
            column_data['G (kN)'].append(column.g)
            column_data['Q (kN)'].append(column.q)
            column_data['Q reduced (kN)'].append(column.q_reduced)
            column_data['G upper floors (kN)'].append(column.g_upper_floors)
            column_data['Q upper floors (kN)'].append(column.q_upper_floors)
            column_data['G level (kN)'].append(column.g_level)
            column_data['Q level (kN)'].append(column.q_level)
            column_data['G superimposed (kN)'].append(column.g_superimposed)
            column_data['G column (kN)'].append(column.g_column)

            for mat in all_column_materials:
                if mat in column.elements:
                    column_data[mat + ' (kg)'].append(column.elements[mat])
                else:
                    column_data[mat + ' (kg)'].append(0)
        column_data = pd.DataFrame(column_data)
        
        # Save walls
        wall_data = {
            'Structure id': [],
            'Wall id': [],
            'Material type': [],
            'Length (m)': [],
            'Thickness (m)': [],
            'G (kN/ml)': [],
            'Q (kN/ml)': [],
            'Q reduced (kN/ml)': [],
            'G upper floors (kN/ml)': [],
            'Q upper floors (kN/ml)': [],
            'G level (kN/ml)': [],
            'Q level (kN/ml)': [],
            'G superimposed (kN/ml)': [],
            'G wall (kN/ml)': []
        }
        all_wall_materials = list(
            set(
                key for wall in self.all_walls.values()
                for key in wall.elements.keys()
            )
        )
        for mat in all_wall_materials:
            wall_data[f'{mat} (kg)'] = []
        for wall_id, wall in self.all_walls.items():
            wall_data['Structure id'].append(self.id)
            wall_data['Wall id'].append(wall.id)
            wall_data['Material type'].append(wall.material)
            wall_data['Length (m)'].append(wall.length)
            wall_data['Thickness (m)'].append(wall.thickness)
            wall_data['G (kN/ml)'].append(wall.g)
            wall_data['Q (kN/ml)'].append(wall.q)
            wall_data['Q reduced (kN/ml)'].append(wall.q_reduced)
            wall_data['G upper floors (kN/ml)'].append(wall.g_upper_floors)
            wall_data['Q upper floors (kN/ml)'].append(wall.q_upper_floors)
            wall_data['G level (kN/ml)'].append(wall.g_level)
            wall_data['Q level (kN/ml)'].append(wall.q_level)
            wall_data['G superimposed (kN/ml)'].append(wall.g_superimposed)
            wall_data['G wall (kN/ml)'].append(wall.g_wall)

            for mat in all_wall_materials:
                if mat in wall.elements:
                    wall_data[mat + ' (kg)'].append(wall.elements[mat])
                else:
                    wall_data[mat + ' (kg)'].append(0)
        wall_data = pd.DataFrame(wall_data)
        
        # Save isolated footings
        isolated_footing_data = {
            'Structure id': [],
            'Isolated footing id': [],
            'Ref. column': [],
            'Width column (m)': [],
            'Width (m)': [],
            'Height (m)': [],
            'N ULS (kN)': [],
            'G upper floors (kN)': [],
            'Q upper floors (kN)': [],
            'G level (kN)': [],
            'Q level (kN)': []
        }
        all_isolated_footing_materials = list(
            set(
                key for isolated_footing in self.all_isolated_footings.values()
                for key in isolated_footing.elements.keys()
            )
        )
        for mat in all_isolated_footing_materials:
            isolated_footing_data[f'{mat} (kg)'] = []
        for isolated_footing_id, isolated_footing \
            in self.all_isolated_footings.items():
            isolated_footing_data['Structure id'].append(self.id)
            isolated_footing_data['Isolated footing id'].append(
                isolated_footing.id
            )
            isolated_footing_data['Ref. column'].append(
                isolated_footing.connected_column
            )
            isolated_footing_data['Width column (m)'].append(
                isolated_footing.width_column
            )
            isolated_footing_data['Width (m)'].append(isolated_footing.width)
            isolated_footing_data['Height (m)'].append(isolated_footing.height)
            isolated_footing_data['N ULS (kN)'].append(isolated_footing.n_axial)
            isolated_footing_data['G upper floors (kN)'].append(
                isolated_footing.g_upper_floors
            )
            isolated_footing_data['Q upper floors (kN)'].append(
                isolated_footing.q_upper_floors
            )
            isolated_footing_data['G level (kN)'].append(
                isolated_footing.g_level
            )
            isolated_footing_data['Q level (kN)'].append(
                isolated_footing.q_level
            )

            for mat in all_isolated_footing_materials:
                if mat in isolated_footing.elements:
                    isolated_footing_data[mat + ' (kg)'].append(
                        isolated_footing.elements[mat]
                    )
                else:
                    isolated_footing_data[mat + ' (kg)'].append(0)
        isolated_footing_data = pd.DataFrame(isolated_footing_data)

        # Save continuous footings
        continuous_footing_data = {
            'Structure id': [],
            'Continuous footing id': [],
            'Ref. wall': [],
            'Width wall (m)': [],
            'Length wall (m)': [],
            'Width (m)': [],
            'Height (m)': [],
            'N axial (kN)': [],
            'N ULS (kN/ml)': [],
            'G upper floors (kN/ml)': [],
            'Q upper floors (kN/ml)': [],
            'G level (kN/ml)': [],
            'Q level (kN/ml)': []
        }
        all_continuous_footing_materials = list(
            set(
                key for continuous_footing in 
                self.all_continuous_footings.values()
                for key in continuous_footing.elements.keys()
            )
        )
        for mat in all_continuous_footing_materials:
            continuous_footing_data[f'{mat} (kg)'] = []
        for continuous_footing_id, continuous_footing \
            in self.all_continuous_footings.items():
            continuous_footing_data['Structure id'].append(self.id)
            continuous_footing_data['Continuous footing id'].append(
                continuous_footing.id
            )
            continuous_footing_data['Ref. wall'].append(
                continuous_footing.connected_wall
            )
            continuous_footing_data['Width wall (m)'].append(
                continuous_footing.wall_width
            )
            continuous_footing_data['Length wall (m)'].append(
                continuous_footing.wall_length
            )
            continuous_footing_data['Width (m)'].append(
                continuous_footing.width
            )
            continuous_footing_data['Height (m)'].append(
                continuous_footing.height
            )
            continuous_footing_data['N axial (kN)'].append(
                continuous_footing.n_axial
            )
            continuous_footing_data['N ULS (kN/ml)'].append(
                continuous_footing.n_uls
            )
            continuous_footing_data['G upper floors (kN/ml)'].append(
                continuous_footing.g_upper_floors
            )
            continuous_footing_data['Q upper floors (kN/ml)'].append(
                continuous_footing.q_upper_floors
            )
            continuous_footing_data['G level (kN/ml)'].append(
                continuous_footing.g_level
            )
            continuous_footing_data['Q level (kN/ml)'].append(
                continuous_footing.q_level
            )

            for mat in all_continuous_footing_materials:
                if mat in continuous_footing.elements:
                    continuous_footing_data[mat + ' (kg)'].append(
                        continuous_footing.elements[mat]
                    )
                else:
                    continuous_footing_data[mat + ' (kg)'].append(0)
        continuous_footing_data = pd.DataFrame(continuous_footing_data)
        
        # Compute material quantities
        # Find all materials used
        floor_material_cols = [
            col for col in floor_data.columns if col not in [
                'Structure id',
                'Floor id',
                'Typology',
                'Span (m)',
                'Width (m)',
                'Thickness (m)',
                'G (kN/m2)',
                'Q (kN/m2)',
                'G superimposed (kN/m2)',
                'G floor (kN/m2)',
                'Location',
                'Secondary beams?',
                'Num. subfloors',
                'Subfloor span (m)',
                'Subfloor width (m)',
                'Num. edge beams',
                'Num. interior beams',
                'Edge beam section',
                'Interior beam section',
                'G subfloor (kN/m2)',
                'G edge beam (kN/ml)',
                'G interior beam (kN/ml)'
            ]
        ]
        beam_material_cols = [
            col for col in beam_data.columns if col not in [
                'Structure id',
                'Beam id',
                'Material type',
                'Length (m)',
                'Section (mm*mm)',
                'G (kN/ml)',
                'Q (kN/ml)',
                'G superimposed (kN/ml)',
                'G beam (kN/ml)',
                'Location'
            ]
        ]
        column_material_cols = [
            col for col in column_data.columns if col not in [
                'Structure id',
                'Column id',
                'Material type',
                'Length (m)',
                'Section (mm*mm)',
                'N ULS (kN)',
                'G (kN)',
                'Q (kN)',
                'Q reduced (kN)',
                'G upper floors (kN)',
                'Q upper floors (kN)',
                'G level (kN)',
                'Q level (kN)',
                'G superimposed (kN)',
                'G column (kN)'
            ]
        ]
        wall_material_cols = [
            col for col in wall_data.columns if col not in [
                'Structure id',
                'Wall id',
                'Material type',
                'Length (m)',
                'Thickness (m)',
                'G (kN/ml)',
                'Q (kN/ml)',
                'Q reduced (kN/ml)',
                'G upper floors (kN/ml)',
                'Q upper floors (kN/ml)',
                'G level (kN/ml)',
                'Q level (kN/ml)',
                'G superimposed (kN/ml)',
                'G wall (kN/ml)'
            ]
        ]
        isolated_footing_material_cols = [
            col for col in isolated_footing_data.columns if col not in [
                'Structure id',
                'Isolated footing id',
                'Ref. column',
                'Width column (m)',
                'Width (m)',
                'Height (m)',
                'N ULS (kN)',
                'G upper floors (kN)',
                'Q upper floors (kN)',
                'G level (kN)',
                'Q level (kN)'
            ]
        ]
        continuous_footing_material_cols = [
            col for col in continuous_footing_data.columns if col not in [
                'Structure id',
                'Continuous footing id',
                'Ref. wall',
                'Width wall (m)',
                'Length wall (m)',
                'Width (m)',
                'Height (m)',
                'N axial (kN)',
                'N ULS (kN/ml)',
                'G upper floors (kN/ml)',
                'Q upper floors (kN/ml)',
                'G level (kN/ml)',
                'Q level (kN/ml)'
            ]
        ]
        # get floor material quantities
        intermediate_floor_material_quantities = {}
        roof_floor_material_quantities = {}
        foundation_floor_material_quantities = {}
        floor_material_quantities = {}
        for mat in floor_material_cols:
            intermediate_floor_material_quantities[mat] = float(
                floor_data[
                    floor_data['Location'] == 'Intermediate'
                ][mat].sum()
            )
            roof_floor_material_quantities[mat] = float(
                floor_data[
                    floor_data['Location'] == 'Roof'
                ][mat].sum()
            )
            foundation_floor_material_quantities[mat] = float(
                floor_data[
                    floor_data['Location'] == 'Foundation'
                ][mat].sum()
            )
            floor_material_quantities[mat] = float(floor_data[mat].sum())
        intermediate_floor_material_quantities = {
            k:v for k, v in intermediate_floor_material_quantities.items() \
                if v > 0.0
        }
        roof_floor_material_quantities = {
            k:v for k, v in roof_floor_material_quantities.items() if v > 0.0
        }
        foundation_floor_material_quantities = {
            k:v for k, v in foundation_floor_material_quantities.items() \
                if v > 0.0
        }
        floor_material_quantities = {
            k:v for k, v in floor_material_quantities.items() if v > 0.0
        }
        
        # get beam material quantities
        intermediate_beam_material_quantities = {}
        roof_beam_material_quantities = {}
        foundation_beam_material_quantities = {}
        beam_material_quantities = {}
        for mat in beam_material_cols:
            intermediate_beam_material_quantities[mat] = float(
                beam_data[
                    beam_data['Location'] == 'Intermediate'
                ][mat].sum()
            )
            roof_beam_material_quantities[mat] = float(
                beam_data[
                    beam_data['Location'] == 'Roof'
                ][mat].sum()
            )
            foundation_beam_material_quantities[mat] = float(
                beam_data[
                    beam_data['Location'] == 'Foundation'
                ][mat].sum()
            )
            beam_material_quantities[mat] = float(beam_data[mat].sum())
        intermediate_beam_material_quantities = {
            k:v for k, v in intermediate_beam_material_quantities.items() \
                if v > 0.0
        }
        roof_beam_material_quantities = {
            k:v for k, v in roof_beam_material_quantities.items() if v > 0.0
        }
        foundation_beam_material_quantities = {
            k:v for k, v in foundation_beam_material_quantities.items() \
                if v > 0.0
        }
        beam_material_quantities = {
            k:v for k, v in beam_material_quantities.items() if v > 0.0
        }

        # get column material quantities
        column_material_quantities = {}
        for mat in column_material_cols:
            column_material_quantities[mat] = float(column_data[mat].sum())
        column_material_quantities = {
            k:v for k, v in column_material_quantities.items() if v > 0.0
        }
        
        # get wall material quantities
        wall_material_quantities = {}
        for mat in wall_material_cols:
            wall_material_quantities[mat] = float(wall_data[mat].sum())
        wall_material_quantities = {
            k:v for k, v in wall_material_quantities.items() if v > 0.0
        }
        
        # get isolated footing material quantities
        isolated_footing_material_quantities = {}
        for mat in isolated_footing_material_cols:
            isolated_footing_material_quantities[mat] \
                = float(isolated_footing_data[mat].sum())
        isolated_footing_material_quantities = {
            k:v for k, v in isolated_footing_material_quantities.items() \
                if v > 0.0
        }
        
        # get continuous footing material quantities
        continuous_footing_material_quantities = {}
        for mat in continuous_footing_material_cols:
            continuous_footing_material_quantities[mat] \
                = float(continuous_footing_data[mat].sum())
        continuous_footing_material_quantities = {
            k:v for k, v in continuous_footing_material_quantities.items() \
                if v > 0.0
        }
        
        # Calculate the total material quantities
        total_material_quantities = {}
        for mat_dict in [
            floor_material_quantities, beam_material_quantities,
            column_material_quantities, wall_material_quantities,
            isolated_footing_material_quantities,
            continuous_footing_material_quantities
        ]:
            for mat, quantity in mat_dict.items():
                if mat not in total_material_quantities.keys():
                    total_material_quantities[mat] = quantity
                else:
                    total_material_quantities[mat] += quantity
        
        # Collect final results in a dict
        self.material_quantities = {
            'Floors': floor_material_quantities,
            'Beams': beam_material_quantities,
            'Columns': column_material_quantities,
            'Walls': wall_material_quantities,
            'Foundation floors': foundation_floor_material_quantities,
            'Foundation beams': foundation_beam_material_quantities,
            'Isolated footings': isolated_footing_material_quantities,
            'Continuous footings': continuous_footing_material_quantities,
            'Roof floors': roof_floor_material_quantities,
            'Roof beams': roof_beam_material_quantities,
            'Total': total_material_quantities
        }
    
    def save_results(self, results_folder):
        os.mkdir(
            self.path.joinpath(
                "./data/results/{:s}/designed_structure_{:d}".format(
                    results_folder, self.id
                )
            )
        )

        # Prep floors df
        floor_data = {
            'Floor id': [],
            'Typology': [],
            'Span (m)': [],
            'Width (m)': [],
            'Thickness (m)': [],
            'p1_x': [],
            'p1_y': [],
            'p1_z': [],
            'p2_x': [],
            'p2_y': [],
            'p2_z': [],
            'p3_x': [],
            'p3_y': [],
            'p3_z': [],
            'p4_x': [],
            'p4_y': [],
            'p4_z': [],
        }
        for floor_id, floor in self.all_floors.items():
            floor_data['Floor id'].append(floor.id)
            floor_data['Typology'].append(floor.typology)
            floor_data['Span (m)'].append(floor.span)
            floor_data['Width (m)'].append(floor.width)
            floor_data['Thickness (m)'].append(floor.height)
            floor_data['p1_x'].append(self.all_nodes[floor.node_1].x)
            floor_data['p1_y'].append(self.all_nodes[floor.node_1].y)
            floor_data['p1_z'].append(self.all_nodes[floor.node_1].z)
            floor_data['p2_x'].append(self.all_nodes[floor.node_2].x)
            floor_data['p2_y'].append(self.all_nodes[floor.node_2].y)
            floor_data['p2_z'].append(self.all_nodes[floor.node_2].z)
            floor_data['p3_x'].append(self.all_nodes[floor.node_3].x)
            floor_data['p3_y'].append(self.all_nodes[floor.node_3].y)
            floor_data['p3_z'].append(self.all_nodes[floor.node_3].z)
            floor_data['p4_x'].append(self.all_nodes[floor.node_4].x)
            floor_data['p4_y'].append(self.all_nodes[floor.node_4].y)
            floor_data['p4_z'].append(self.all_nodes[floor.node_4].z)
        floor_data = pd.DataFrame(floor_data)
        
        foundation_floor_data = {
            'Floor id': [],
            'Typology': [],
            'Span (m)': [],
            'Width (m)': [],
            'Thickness (m)': [],
            'p1_x': [],
            'p1_y': [],
            'p1_z': [],
            'p2_x': [],
            'p2_y': [],
            'p2_z': [],
            'p3_x': [],
            'p3_y': [],
            'p3_z': [],
            'p4_x': [],
            'p4_y': [],
            'p4_z': [],
        }
        for floor_id, floor in self.all_foundation_floors.items():
            foundation_floor_data['Floor id'].append(floor.id)
            foundation_floor_data['Typology'].append(floor.typology)
            foundation_floor_data['Span (m)'].append(floor.span)
            foundation_floor_data['Width (m)'].append(floor.width)
            foundation_floor_data['Thickness (m)'].append(floor.height)
            foundation_floor_data['p1_x'].append(self.all_nodes[floor.node_1].x)
            foundation_floor_data['p1_y'].append(self.all_nodes[floor.node_1].y)
            foundation_floor_data['p1_z'].append(self.all_nodes[floor.node_1].z)
            foundation_floor_data['p2_x'].append(self.all_nodes[floor.node_2].x)
            foundation_floor_data['p2_y'].append(self.all_nodes[floor.node_2].y)
            foundation_floor_data['p2_z'].append(self.all_nodes[floor.node_2].z)
            foundation_floor_data['p3_x'].append(self.all_nodes[floor.node_3].x)
            foundation_floor_data['p3_y'].append(self.all_nodes[floor.node_3].y)
            foundation_floor_data['p3_z'].append(self.all_nodes[floor.node_3].z)
            foundation_floor_data['p4_x'].append(self.all_nodes[floor.node_4].x)
            foundation_floor_data['p4_y'].append(self.all_nodes[floor.node_4].y)
            foundation_floor_data['p4_z'].append(self.all_nodes[floor.node_4].z)
        foundation_floor_data = pd.DataFrame(foundation_floor_data)
        
        # Prep beams df
        beam_data = {
            'Beam id': [],
            'Material type': [],
            'Length (m)': [],
            'Section (mm*mm)': [],
            'p1_x': [],
            'p1_y': [],
            'p1_z': [],
            'p2_x': [],
            'p2_y': [],
            'p2_z': [],
        }
        for beam_id, beam in self.all_beams.items():
            beam_data['Beam id'].append(beam.id)
            beam_data['Material type'].append(beam.material)
            beam_data['Length (m)'].append(beam.length)
            beam_data['Section (mm*mm)'].append(beam.section)
            beam_data['p1_x'].append(self.all_nodes[beam.startpoint].x)
            beam_data['p1_y'].append(self.all_nodes[beam.startpoint].y)
            beam_data['p1_z'].append(self.all_nodes[beam.startpoint].z)
            beam_data['p2_x'].append(self.all_nodes[beam.endpoint].x)
            beam_data['p2_y'].append(self.all_nodes[beam.endpoint].y)
            beam_data['p2_z'].append(self.all_nodes[beam.endpoint].z)
        beam_data = pd.DataFrame(beam_data)

        foundation_beam_data = {
            'Beam id': [],
            'Material type': [],
            'Length (m)': [],
            'Section (mm*mm)': [],
            'p1_x': [],
            'p1_y': [],
            'p1_z': [],
            'p2_x': [],
            'p2_y': [],
            'p2_z': [],
        }
        for beam_id, beam in self.all_foundation_beams.items():
            foundation_beam_data['Beam id'].append(beam.id)
            foundation_beam_data['Material type'].append(beam.material)
            foundation_beam_data['Length (m)'].append(beam.length)
            foundation_beam_data['Section (mm*mm)'].append(beam.section)
            foundation_beam_data['p1_x'].append(
                self.all_nodes[beam.startpoint].x
            )
            foundation_beam_data['p1_y'].append(
                self.all_nodes[beam.startpoint].y
            )
            foundation_beam_data['p1_z'].append(
                self.all_nodes[beam.startpoint].z
            )
            foundation_beam_data['p2_x'].append(self.all_nodes[beam.endpoint].x)
            foundation_beam_data['p2_y'].append(self.all_nodes[beam.endpoint].y)
            foundation_beam_data['p2_z'].append(self.all_nodes[beam.endpoint].z)
        foundation_beam_data = pd.DataFrame(foundation_beam_data)
        
        # Prep columns df
        column_data = {
            'Column id': [],
            'Material type': [],
            'Length (m)': [],
            'Section (mm*mm)': [],
            'p1_x': [],
            'p1_y': [],
            'p1_z': [],
            'p2_x': [],
            'p2_y': [],
            'p2_z': [],
        }
        for column_id, column in self.all_columns.items():
            column_data['Column id'].append(column.id)
            column_data['Material type'].append(column.material)
            column_data['Length (m)'].append(column.length)
            column_data['Section (mm*mm)'].append(column.section)
            column_data['p1_x'].append(self.all_nodes[column.startpoint].x)
            column_data['p1_y'].append(self.all_nodes[column.startpoint].y)
            column_data['p1_z'].append(self.all_nodes[column.startpoint].z)
            column_data['p2_x'].append(self.all_nodes[column.endpoint].x)
            column_data['p2_y'].append(self.all_nodes[column.endpoint].y)
            column_data['p2_z'].append(self.all_nodes[column.endpoint].z)
        column_data = pd.DataFrame(column_data)
        
        # Save walls
        wall_data = {
            'Wall id': [],
            'Material type': [],
            'Length (m)': [],
            'Thickness (m)': [],
            'p1_x': [],
            'p1_y': [],
            'p1_z': [],
            'p2_x': [],
            'p2_y': [],
            'p2_z': [],
            'p3_x': [],
            'p3_y': [],
            'p3_z': [],
            'p4_x': [],
            'p4_y': [],
            'p4_z': [],
        }
        for wall_id, wall in self.all_walls.items():
            wall_data['Wall id'].append(wall.id)
            wall_data['Material type'].append(wall.material)
            wall_data['Length (m)'].append(wall.length)
            wall_data['Thickness (m)'].append(wall.thickness)
            wall_data['p1_x'].append(self.all_nodes[wall.node_1].x)
            wall_data['p1_y'].append(self.all_nodes[wall.node_1].y)
            wall_data['p1_z'].append(self.all_nodes[wall.node_1].z)
            wall_data['p2_x'].append(self.all_nodes[wall.node_2].x)
            wall_data['p2_y'].append(self.all_nodes[wall.node_2].y)
            wall_data['p2_z'].append(self.all_nodes[wall.node_2].z)
            wall_data['p3_x'].append(self.all_nodes[wall.node_3].x)
            wall_data['p3_y'].append(self.all_nodes[wall.node_3].y)
            wall_data['p3_z'].append(self.all_nodes[wall.node_3].z)
            wall_data['p4_x'].append(self.all_nodes[wall.node_4].x)
            wall_data['p4_y'].append(self.all_nodes[wall.node_4].y)
            wall_data['p4_z'].append(self.all_nodes[wall.node_4].z)

        wall_data = pd.DataFrame(wall_data)

        # Save isolated footings
        isolated_footing_data = {
            'Isolated footing id': [],
            'Ref. column': [],
            'Width (m)': [],
            'Height (m)': [],
            'p_x': [],
            'p_y': [],
            'p_z': [],
        }
        for isolated_footing_id, isolated_footing \
            in self.all_isolated_footings.items():
            isolated_footing_data['Isolated footing id'].append(
                isolated_footing.id
            )
            isolated_footing_data['Ref. column'].append(
                isolated_footing.connected_column
            )
            isolated_footing_data['Width (m)'].append(isolated_footing.width)
            isolated_footing_data['Height (m)'].append(isolated_footing.height)
            isolated_footing_data['p_x'].append(
                self.all_nodes[
                    self.all_columns[
                        isolated_footing.connected_column
                    ].startpoint
                ].x  
            )
            isolated_footing_data['p_y'].append(
                self.all_nodes[
                    self.all_columns[
                        isolated_footing.connected_column
                    ].startpoint
                ].y
            )
            isolated_footing_data['p_z'].append(
                self.all_nodes[
                    self.all_columns[
                        isolated_footing.connected_column
                    ].startpoint
                ].z
            )
        isolated_footing_data = pd.DataFrame(isolated_footing_data)

        # Save continuous footings
        continuous_footing_data = {
            'Continuous footing id': [],
            'Ref. wall': [],
            'Width wall (m)': [],
            'Length wall (m)': [],
            'Width (m)': [],
            'Height (m)': [],
            'p1_x': [],
            'p1_y': [],
            'p1_z': [],
            'p2_x': [],
            'p2_y': [],
            'p2_z': [],
        }
        for continuous_footing_id, continuous_footing \
            in self.all_continuous_footings.items():
            continuous_footing_data['Continuous footing id'].append(
                continuous_footing.id
            )
            continuous_footing_data['Ref. wall'].append(
                continuous_footing.connected_wall
            )
            continuous_footing_data['Width wall (m)'].append(
                continuous_footing.wall_width
            )
            continuous_footing_data['Length wall (m)'].append(
                continuous_footing.wall_length
            )
            continuous_footing_data['Width (m)'].append(
                continuous_footing.width
            )
            continuous_footing_data['Height (m)'].append(
                continuous_footing.height
            )
            continuous_footing_data['p1_x'].append(
                self.all_nodes[
                    self.all_walls[
                        continuous_footing.connected_wall
                    ].node_1
                ].x
            )
            continuous_footing_data['p1_y'].append(
                self.all_nodes[
                    self.all_walls[
                        continuous_footing.connected_wall
                    ].node_1
                ].y
            )
            continuous_footing_data['p1_z'].append(
                self.all_nodes[
                    self.all_walls[
                        continuous_footing.connected_wall
                    ].node_1
                ].z
            )
            continuous_footing_data['p2_x'].append(
                self.all_nodes[
                    self.all_walls[
                        continuous_footing.connected_wall
                    ].node_2
                ].x
            )
            continuous_footing_data['p2_y'].append(
                self.all_nodes[
                    self.all_walls[
                        continuous_footing.connected_wall
                    ].node_2
                ].y
            )
            continuous_footing_data['p2_z'].append(
                self.all_nodes[
                    self.all_walls[
                        continuous_footing.connected_wall
                    ].node_2
                ].z
            )

        continuous_footing_data = pd.DataFrame(continuous_footing_data)

        # save all dfs
        floor_data.to_csv(
            self.path.joinpath(
                f"./data/results/{results_folder}/designed_structure_{self.id}/floors.csv"
            ), index=False
        )
        foundation_floor_data.to_csv(
            self.path.joinpath(
                f"./data/results/{results_folder}/designed_structure_{self.id}/foundation_floors.csv"
            ), index=False
        )
        beam_data.to_csv(
            self.path.joinpath(
                f"./data/results/{results_folder}/designed_structure_{self.id}/beams.csv"
            ), index=False
        )
        foundation_beam_data.to_csv(
            self.path.joinpath(
                f"./data/results/{results_folder}/designed_structure_{self.id}/foundation_beams.csv"
            ), index=False
        )
        column_data.to_csv(
            self.path.joinpath(
                f"./data/results/{results_folder}/designed_structure_{self.id}/columns.csv"
            ), index=False
        )
        wall_data.to_csv(
            self.path.joinpath(
                f"./data/results/{results_folder}/designed_structure_{self.id}/walls.csv"
            ), index=False
        )
        isolated_footing_data.to_csv(
            self.path.joinpath(
                f"./data/results/{results_folder}/designed_structure_{self.id}/isolated_footings.csv"
            ), index=False
        )
        continuous_footing_data.to_csv(
            self.path.joinpath(
                f"./data/results/{results_folder}/designed_structure_{self.id}/continuous_footings.csv"
            ), index=False
        )