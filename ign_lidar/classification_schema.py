"""Classification Schema for IGN LiDAR HD Dataset

This module consolidates all classification schemas:
- ASPRS LAS 1.4 standard classification codes (0-31)
- ASPRS extended codes for BD TOPO integration (32-255)
- LOD2/LOD3 building-focused classification schemas
- Mapping utilities and conversion functions

This replaces and consolidates:
- asprs_classes.py (ASPRS codes and BD TOPO mappings)
- classes.py (LOD2/LOD3 building classes)

Author: IGN LiDAR HD Development Team
Date: October 22, 2025
Version: 3.1.0 - Consolidated schema
"""

from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# ASPRS LAS 1.4 Standard Classification Codes (0-255)
# ============================================================================


class ASPRSClass(IntEnum):
    """
    ASPRS LAS 1.4 classification codes.

    Standard codes (0-31) are reserved by ASPRS specification.
    Extended codes (32-255) are user-defined, here mapped to IGN BD TOPO features.

    Reference: ASPRS LAS Specification Version 1.4 - R15
    """

    # ========================================================================
    # Standard ASPRS Classifications (0-31) - Reserved by ASPRS
    # ========================================================================

    CREATED_NEVER_CLASSIFIED = 0  # Created, never classified
    UNCLASSIFIED = 1  # Unclassified
    GROUND = 2  # Ground
    LOW_VEGETATION = 3  # Low Vegetation
    MEDIUM_VEGETATION = 4  # Medium Vegetation
    HIGH_VEGETATION = 5  # High Vegetation
    BUILDING = 6  # Building
    LOW_POINT = 7  # Low Point (noise)
    RESERVED_8 = 8  # Reserved (formerly Model Key-point)
    WATER = 9  # Water
    RAIL = 10  # Rail
    ROAD_SURFACE = 11  # Road Surface
    RESERVED_12 = 12  # Reserved (formerly Wire - Guard)
    WIRE_GUARD = 13  # Wire - Guard (Shield)
    WIRE_CONDUCTOR = 14  # Wire - Conductor (Phase)
    TRANSMISSION_TOWER = 15  # Transmission Tower
    WIRE_STRUCTURE_CONNECTOR = 16  # Wire-structure Connector (Insulator)
    BRIDGE_DECK = 17  # Bridge Deck
    HIGH_NOISE = 18  # High Noise
    OVERHEAD_STRUCTURE = 19  # Overhead Structure
    IGNORED_GROUND = 20  # Ignored Ground (Breakline Proximity)
    SNOW = 21  # Snow
    TEMPORAL_EXCLUSION = 22  # Temporal Exclusion
    # 23-31 reserved for future ASPRS use

    # ========================================================================
    # Extended Classifications (32-255) - IGN BD TOPO® Specific
    # ========================================================================

    # Road types (32-49)
    ROAD_MOTORWAY = 32  # Autoroute
    ROAD_PRIMARY = 33  # Route principale
    ROAD_SECONDARY = 34  # Route secondaire
    ROAD_TERTIARY = 35  # Route tertiaire
    ROAD_RESIDENTIAL = 36  # Rue résidentielle
    ROAD_SERVICE = 37  # Route de service
    ROAD_PEDESTRIAN = 38  # Zone piétonne
    ROAD_CYCLEWAY = 39  # Piste cyclable
    ROAD_PARKING = 40  # Parking
    ROAD_BRIDGE = 41  # Pont routier (deprecated, use BRIDGE_DECK)
    ROAD_TUNNEL = 42  # Tunnel routier
    ROAD_ROUNDABOUT = 43  # Rond-point

    # Special area types (40-49, overlaps intentionally with roads)
    PARKING = 40  # Aires de stationnement (BD TOPO)
    SPORTS_FACILITY = 41  # Équipements sportifs (BD TOPO)
    CEMETERY = 42  # Cimetières (BD TOPO)
    POWER_LINE = 43  # Lignes électriques (BD TOPO)
    AGRICULTURE = 44  # Terres agricoles (RPG)

    # Building types (50-69)
    BUILDING_RESIDENTIAL = 50  # Bâtiment résidentiel
    BUILDING_COMMERCIAL = 51  # Bâtiment commercial
    BUILDING_INDUSTRIAL = 52  # Bâtiment industriel
    BUILDING_RELIGIOUS = 53  # Bâtiment religieux
    BUILDING_PUBLIC = 54  # Bâtiment public
    BUILDING_AGRICULTURAL = 55  # Bâtiment agricole
    BUILDING_SPORTS = 56  # Bâtiment sportif
    BUILDING_HISTORIC = 57  # Bâtiment historique
    BUILDING_ROOF = 58  # Toit
    BUILDING_WALL = 59  # Mur
    BUILDING_FACADE = 60  # Façade
    BUILDING_CHIMNEY = 61  # Cheminée
    BUILDING_BALCONY = 62  # Balcon

    # LOD3 Roof subtypes (63-69) - v3.1
    BUILDING_ROOF_FLAT = 63  # Toit plat
    BUILDING_ROOF_GABLED = 64  # Toit à pignon (2 pentes)
    BUILDING_ROOF_HIPPED = 65  # Toit à 4 pentes
    BUILDING_ROOF_COMPLEX = 66  # Toit complexe (mansarde, etc.)
    BUILDING_ROOF_RIDGE = 67  # Faîtage (ligne de crête)
    BUILDING_ROOF_EDGE = 68  # Bordure de toit
    BUILDING_DORMER = 69  # Lucarne
    # Note: IGN uses classes 64-65 in some tiles (now mapped to roof types)

    # Vegetation types (70-79)
    VEGETATION_TREE = 70  # Arbre
    VEGETATION_BUSH = 71  # Buisson
    VEGETATION_GRASS = 72  # Herbe
    VEGETATION_HEDGE = 73  # Haie
    VEGETATION_FOREST = 74  # Forêt
    VEGETATION_VINEYARD = 75  # Vignoble
    VEGETATION_ORCHARD = 76  # Verger

    # Water types (80-89)
    WATER_RIVER = 80  # Rivière
    WATER_LAKE = 81  # Lac
    WATER_POND = 82  # Étang
    WATER_CANAL = 83  # Canal
    WATER_FOUNTAIN = 84  # Fontaine
    WATER_SWIMMING_POOL = 85  # Piscine

    # Infrastructure (90-109)
    RAILWAY_TRACK = 90  # Voie ferrée
    RAILWAY_PLATFORM = 91  # Quai de gare
    RAILWAY_BRIDGE = 92  # Pont ferroviaire
    RAILWAY_TUNNEL = 93  # Tunnel ferroviaire
    POWER_LINE_OVERHEAD = 94  # Ligne électrique aérienne
    POWER_PYLON = 95  # Pylône électrique
    ANTENNA = 96  # Antenne
    STREET_LIGHT = 97  # Lampadaire
    TRAFFIC_SIGN = 98  # Panneau de signalisation
    FENCE = 99  # Clôture
    WALL_STANDALONE = 100  # Mur indépendant

    # Urban furniture (110-119)
    BENCH = 110  # Banc
    BIN = 111  # Poubelle
    SHELTER = 112  # Abri
    BOLLARD = 113  # Borne
    BARRIER = 114  # Barrière

    # Terrain (120-129)
    TERRAIN_BARE = 120  # Sol nu
    TERRAIN_GRAVEL = 121  # Gravier
    TERRAIN_SAND = 122  # Sable
    TERRAIN_ROCK = 123  # Roche
    TERRAIN_CLIFF = 124  # Falaise
    TERRAIN_QUARRY = 125  # Carrière

    # Vehicles (130-139)
    VEHICLE_CAR = 130  # Voiture
    VEHICLE_TRUCK = 131  # Camion
    VEHICLE_BUS = 132  # Bus
    VEHICLE_TRAIN = 133  # Train
    VEHICLE_BOAT = 134  # Bateau
    VEHICLE_AIRCRAFT = 135  # Avion


# ============================================================================
# LOD2 Building-Focused Classification (15 classes)
# ============================================================================


class LOD2Class(IntEnum):
    """
    LOD2 (Level of Detail 2) building-focused classification.

    Simplified building taxonomy for architectural analysis.
    Focus on main structural elements and roof types.
    """

    # Structural elements
    WALL = 0

    # Roof types
    ROOF_FLAT = 1
    ROOF_GABLE = 2
    ROOF_HIP = 3

    # Roof details
    CHIMNEY = 4
    DORMER = 5

    # Facades
    BALCONY = 6
    OVERHANG = 7

    # Foundation
    FOUNDATION = 8

    # Context (non-building)
    GROUND = 9
    VEGETATION_LOW = 10
    VEGETATION_HIGH = 11
    WATER = 12
    VEHICLE = 13
    OTHER = 14


# ============================================================================
# LOD3 Extended Building Classification (30 classes)
# ============================================================================


class LOD3Class(IntEnum):
    """
    LOD3 (Level of Detail 3) extended building classification.

    Detailed building taxonomy including windows, doors, and facade elements.
    For high-resolution architectural modeling.
    """

    # Structural elements (walls with openings)
    WALL_PLAIN = 0
    WALL_WITH_WINDOWS = 1
    WALL_WITH_DOOR = 2

    # Roof types (detailed)
    ROOF_FLAT = 3
    ROOF_GABLE = 4
    ROOF_HIP = 5
    ROOF_MANSARD = 6
    ROOF_GAMBREL = 7

    # Roof details
    CHIMNEY = 8
    DORMER_GABLE = 9
    DORMER_SHED = 10
    SKYLIGHT = 11
    ROOF_EDGE = 12

    # Windows and doors
    WINDOW = 13
    DOOR = 14
    GARAGE_DOOR = 15

    # Facades
    BALCONY = 16
    BALUSTRADE = 17
    OVERHANG = 18
    PILLAR = 19
    CORNICE = 20

    # Foundation
    FOUNDATION = 21
    BASEMENT_WINDOW = 22

    # Context (non-building)
    GROUND = 23
    VEGETATION_LOW = 24
    VEGETATION_HIGH = 25
    WATER = 26
    VEHICLE = 27
    STREET_FURNITURE = 28
    OTHER = 29


# ============================================================================
# Classification Mode
# ============================================================================


class ClassificationMode:
    """
    Classification mode for LAS output.

    Modes:
    - ASPRS_STANDARD: Standard ASPRS codes (0-31) only
    - ASPRS_EXTENDED: ASPRS codes + BD TOPO extended codes (32-255)
    - LOD2: LOD2 building-focused classes (for training)
    - LOD3: LOD3 detailed building classes (for training)
    """

    ASPRS_STANDARD = "asprs_standard"
    ASPRS_EXTENDED = "asprs_extended"
    LOD2 = "lod2"
    LOD3 = "lod3"


# ============================================================================
# ASPRS to LOD Mappings
# ============================================================================

ASPRS_TO_LOD2: Dict[int, int] = {
    0: LOD2Class.OTHER,  # Never classified
    1: LOD2Class.OTHER,  # Unclassified
    2: LOD2Class.GROUND,  # Ground
    3: LOD2Class.VEGETATION_LOW,  # Low Vegetation
    4: LOD2Class.VEGETATION_LOW,  # Medium Vegetation
    5: LOD2Class.VEGETATION_HIGH,  # High Vegetation
    6: LOD2Class.WALL,  # Building (requires refinement)
    7: LOD2Class.VEGETATION_LOW,  # Low Point (noise)
    8: LOD2Class.OTHER,  # Model Key-point
    9: LOD2Class.WATER,  # Water
    10: LOD2Class.OTHER,  # Rail
    11: LOD2Class.OTHER,  # Road Surface
    12: LOD2Class.OTHER,  # Reserved
    13: LOD2Class.OTHER,  # Wire - Guard
    14: LOD2Class.OTHER,  # Wire - Conductor
    15: LOD2Class.OTHER,  # Transmission Tower
    16: LOD2Class.OTHER,  # Wire-structure Connector
    17: LOD2Class.VEHICLE,  # Bridge Deck (temporary)
    18: LOD2Class.VEGETATION_HIGH,  # High Noise
    64: LOD2Class.OTHER,  # Unknown
    65: LOD2Class.OTHER,  # Unknown
    67: LOD2Class.OTHER,  # Unknown
}

ASPRS_TO_LOD3: Dict[int, int] = {
    0: LOD3Class.OTHER,  # Never classified
    1: LOD3Class.OTHER,  # Unclassified
    2: LOD3Class.GROUND,  # Ground
    3: LOD3Class.VEGETATION_LOW,  # Low Vegetation
    4: LOD3Class.VEGETATION_LOW,  # Medium Vegetation
    5: LOD3Class.VEGETATION_HIGH,  # High Vegetation
    6: LOD3Class.WALL_PLAIN,  # Building (requires refinement)
    7: LOD3Class.VEGETATION_LOW,  # Low Point (noise)
    8: LOD3Class.OTHER,  # Model Key-point
    9: LOD3Class.WATER,  # Water
    10: LOD3Class.OTHER,  # Rail
    11: LOD3Class.GROUND,  # Road Surface
    12: LOD3Class.OTHER,  # Reserved
    13: LOD3Class.OTHER,  # Wire - Guard
    14: LOD3Class.OTHER,  # Wire - Conductor
    15: LOD3Class.OTHER,  # Transmission Tower
    16: LOD3Class.OTHER,  # Wire-structure Connector
    17: LOD3Class.VEHICLE,  # Bridge Deck (temporary)
    18: LOD3Class.VEGETATION_HIGH,  # High Noise
    64: LOD3Class.OTHER,  # Unknown
    65: LOD3Class.OTHER,  # Unknown
    67: LOD3Class.OTHER,  # Unknown
}

# Reverse mappings
LOD2_TO_ASPRS: Dict[int, int] = {v: k for k, v in ASPRS_TO_LOD2.items()}
LOD3_TO_ASPRS: Dict[int, int] = {v: k for k, v in ASPRS_TO_LOD3.items()}


# ============================================================================
# Lidar HD 7-Class Contiguous Schema (for Pointcept / PTv3 training)
# ============================================================================
# Pointcept-family models (PTv3, PT-v2, SparseConv) require labels in a
# contiguous [0, K-1] range, with an ignore_index for masked points.
# This schema follows the canonical IGN Lidar HD 7-class semseg taxonomy.

LIDARHD_7CL_NAMES: List[str] = [
    "ground",            # 0
    "vegetation_low",    # 1
    "vegetation_medium", # 2
    "vegetation_high",   # 3
    "building",          # 4
    "water",             # 5
    "bridge",            # 6
]

LIDARHD_7CL_IGNORE_INDEX: int = -1

# Maps every known ASPRS / extended code to the contiguous 7-class index,
# or to LIDARHD_7CL_IGNORE_INDEX for noise / unclassified / unsupported codes.
ASPRS_TO_LIDARHD_7CL: Dict[int, int] = {
    # --- Ignored ---
    0: LIDARHD_7CL_IGNORE_INDEX,   # Never classified
    1: LIDARHD_7CL_IGNORE_INDEX,   # Unclassified
    7: LIDARHD_7CL_IGNORE_INDEX,   # Low point (noise)
    8: LIDARHD_7CL_IGNORE_INDEX,   # Reserved
    18: LIDARHD_7CL_IGNORE_INDEX,  # High noise
    20: LIDARHD_7CL_IGNORE_INDEX,  # Ignored ground
    22: LIDARHD_7CL_IGNORE_INDEX,  # Temporal exclusion
    # --- Ground (0) ---
    2: 0,
    11: 0,  # Road surface treated as ground
    # --- Vegetation (1, 2, 3) ---
    3: 1,
    4: 2,
    5: 3,
    # --- Building (4) ---
    6: 4,
    # --- Water (5) ---
    9: 5,
    # --- Bridge (6) ---
    17: 6,
}
# Extended ASPRS codes — all building variants → building class 4
for _code in range(50, 70):
    ASPRS_TO_LIDARHD_7CL[_code] = 4
# Extended vegetation codes (70-79) → mostly veg high, low for bushes/grass/hedge
ASPRS_TO_LIDARHD_7CL.update({
    70: 3,  # Tree
    71: 1,  # Bush
    72: 1,  # Grass
    73: 1,  # Hedge
    74: 3,  # Forest
    75: 2,  # Vineyard
    76: 3,  # Orchard
})
# Extended water codes (80-85)
for _code in range(80, 86):
    ASPRS_TO_LIDARHD_7CL[_code] = 5

# Inverse: contiguous index -> representative ASPRS code (for export / debug)
LIDARHD_7CL_TO_ASPRS: Dict[int, int] = {
    0: 2,   # Ground
    1: 3,   # Low veg
    2: 4,   # Medium veg
    3: 5,   # High veg
    4: 6,   # Building
    5: 9,   # Water
    6: 17,  # Bridge
}


def remap_asprs_to_lidarhd_7cl(labels):
    """Vectorized ASPRS -> contiguous Lidar HD 7-class remap.

    Unknown codes default to LIDARHD_7CL_IGNORE_INDEX.

    Args:
        labels: 1-D integer array of ASPRS codes.

    Returns:
        np.ndarray (int16) of contiguous labels in [0..6] or ignore_index.
    """
    import numpy as np
    labels = np.asarray(labels)
    out = np.full(labels.shape, LIDARHD_7CL_IGNORE_INDEX, dtype=np.int16)
    # Build a flat lookup over the observed range; bounded copy avoids a big LUT.
    for asprs_code, contiguous_id in ASPRS_TO_LIDARHD_7CL.items():
        out[labels == asprs_code] = contiguous_id
    return out


# ============================================================================
# BD TOPO® Nature Attribute Mappings
# ============================================================================

# Building nature to ASPRS classification
BUILDING_NATURE_TO_ASPRS: Dict[str, int] = {
    # Residential
    "Indifférencié": ASPRSClass.BUILDING,
    "Résidentiel": ASPRSClass.BUILDING_RESIDENTIAL,
    "Immeuble": ASPRSClass.BUILDING_RESIDENTIAL,
    "Maison": ASPRSClass.BUILDING_RESIDENTIAL,
    # Commercial
    "Commercial et services": ASPRSClass.BUILDING_COMMERCIAL,
    "Commercial": ASPRSClass.BUILDING_COMMERCIAL,
    # Industrial
    "Industriel": ASPRSClass.BUILDING_INDUSTRIAL,
    "Industriel, agricole ou commercial": ASPRSClass.BUILDING_INDUSTRIAL,
    "Serre": ASPRSClass.BUILDING_AGRICULTURAL,
    # Religious
    "Religieux": ASPRSClass.BUILDING_RELIGIOUS,
    "Chapelle": ASPRSClass.BUILDING_RELIGIOUS,
    "Église": ASPRSClass.BUILDING_RELIGIOUS,
    "Cathédrale": ASPRSClass.BUILDING_RELIGIOUS,
    # Public and services
    "Sportif": ASPRSClass.BUILDING_SPORTS,
    "Enseignement": ASPRSClass.BUILDING_PUBLIC,
    "Santé": ASPRSClass.BUILDING_PUBLIC,
    "Science et recherche": ASPRSClass.BUILDING_PUBLIC,
    "Transport": ASPRSClass.BUILDING_PUBLIC,
    "Gare": ASPRSClass.BUILDING_PUBLIC,
    # Agricultural
    "Agricole": ASPRSClass.BUILDING_AGRICULTURAL,
    # Historic/monument
    "Monument": ASPRSClass.BUILDING_HISTORIC,
    "Château": ASPRSClass.BUILDING_HISTORIC,
    "Fort": ASPRSClass.BUILDING_HISTORIC,
    "Remarquable": ASPRSClass.BUILDING_HISTORIC,
}

# Road nature to ASPRS classification
ROAD_NATURE_TO_ASPRS: Dict[str, int] = {
    "Autoroute": ASPRSClass.ROAD_MOTORWAY,
    "Quasi-autoroute": ASPRSClass.ROAD_MOTORWAY,
    "Route à 2 chaussées": ASPRSClass.ROAD_PRIMARY,
    "Route à 1 chaussée": ASPRSClass.ROAD_SECONDARY,
    "Route empierrée": ASPRSClass.ROAD_TERTIARY,
    "Chemin": ASPRSClass.ROAD_SERVICE,
    "Bretelle": ASPRSClass.ROAD_SERVICE,
    "Rond-point": ASPRSClass.ROAD_ROUNDABOUT,
    "Place": ASPRSClass.ROAD_PEDESTRIAN,
    "Sentier": ASPRSClass.ROAD_PEDESTRIAN,
    "Escalier": ASPRSClass.ROAD_PEDESTRIAN,
    "Piste cyclable": ASPRSClass.ROAD_CYCLEWAY,
    "Parking": ASPRSClass.PARKING,
}

# Vegetation nature to ASPRS classification
VEGETATION_NATURE_TO_ASPRS: Dict[str, int] = {
    "Arbre": ASPRSClass.VEGETATION_TREE,
    "Haie": ASPRSClass.VEGETATION_HEDGE,
    "Bois": ASPRSClass.VEGETATION_FOREST,
    "Forêt fermée de feuillus": ASPRSClass.VEGETATION_FOREST,
    "Forêt fermée de conifères": ASPRSClass.VEGETATION_FOREST,
    "Forêt fermée mixte": ASPRSClass.VEGETATION_FOREST,
    "Forêt ouverte": ASPRSClass.VEGETATION_FOREST,
    "Lande ligneuse": ASPRSClass.VEGETATION_BUSH,
    "Verger": ASPRSClass.VEGETATION_ORCHARD,
    "Vigne": ASPRSClass.VEGETATION_VINEYARD,
    "Peupleraie": ASPRSClass.VEGETATION_TREE,
}

# Water nature to ASPRS classification
WATER_NATURE_TO_ASPRS: Dict[str, int] = {
    "Cours d'eau": ASPRSClass.WATER_RIVER,
    "Plan d'eau": ASPRSClass.WATER_LAKE,
    "Étang": ASPRSClass.WATER_POND,
    "Lac": ASPRSClass.WATER_LAKE,
    "Canal": ASPRSClass.WATER_CANAL,
    "Bassin": ASPRSClass.WATER_POND,
    "Réservoir": ASPRSClass.WATER_POND,
}

# Railway, sports, cemetery, power line, parking, bridge - all map to single codes
RAILWAY_NATURE_TO_ASPRS: Dict[str, int] = {"default": ASPRSClass.RAIL}
SPORTS_NATURE_TO_ASPRS: Dict[str, int] = {"default": ASPRSClass.SPORTS_FACILITY}
CEMETERY_NATURE_TO_ASPRS: Dict[str, int] = {"default": ASPRSClass.CEMETERY}
POWER_LINE_NATURE_TO_ASPRS: Dict[str, int] = {"default": ASPRSClass.POWER_LINE}
PARKING_NATURE_TO_ASPRS: Dict[str, int] = {"default": ASPRSClass.PARKING}
BRIDGE_NATURE_TO_ASPRS: Dict[str, int] = {"default": ASPRSClass.BRIDGE_DECK}


# ============================================================================
# Human-Readable Names
# ============================================================================

ASPRS_CLASS_NAMES: Dict[int, str] = {
    # Standard codes
    0: "Created, Never Classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (Noise)",
    8: "Reserved",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Reserved",
    13: "Wire - Guard (Shield)",
    14: "Wire - Conductor (Phase)",
    15: "Transmission Tower",
    16: "Wire-Structure Connector",
    17: "Bridge Deck",
    18: "High Noise",
    19: "Overhead Structure",
    20: "Ignored Ground",
    21: "Snow",
    22: "Temporal Exclusion",
    # Extended codes
    32: "Motorway",
    33: "Primary Road",
    34: "Secondary Road",
    35: "Tertiary Road",
    36: "Residential Road",
    37: "Service Road",
    38: "Pedestrian Zone",
    39: "Cycleway",
    40: "Parking",
    41: "Sports Facility",
    42: "Cemetery",
    43: "Power Line",
    44: "Agriculture",
    50: "Residential Building",
    51: "Commercial Building",
    52: "Industrial Building",
    53: "Religious Building",
    54: "Public Building",
    55: "Agricultural Building",
    56: "Sports Building",
    57: "Historic Building",
    58: "Roof",
    59: "Wall",
    60: "Facade",
    61: "Chimney",
    62: "Balcony",
    70: "Tree",
    71: "Bush",
    72: "Grass",
    73: "Hedge",
    74: "Forest",
    75: "Vineyard",
    76: "Orchard",
    80: "River",
    81: "Lake",
    82: "Pond",
    83: "Canal",
    84: "Fountain",
    85: "Swimming Pool",
    90: "Railway Track",
    91: "Railway Platform",
    92: "Railway Bridge",
    93: "Railway Tunnel",
    94: "Power Line Overhead",
    95: "Power Pylon",
    96: "Antenna",
    97: "Street Light",
    98: "Traffic Sign",
    99: "Fence",
    100: "Standalone Wall",
    110: "Bench",
    111: "Bin",
    112: "Shelter",
    113: "Bollard",
    114: "Barrier",
    120: "Bare Terrain",
    121: "Gravel",
    122: "Sand",
    123: "Rock",
    124: "Cliff",
    125: "Quarry",
    130: "Car",
    131: "Truck",
    132: "Bus",
    133: "Train",
    134: "Boat",
    135: "Aircraft",
}


# ============================================================================
# Utility Functions
# ============================================================================


def get_class_name(code: int) -> str:
    """
    Get human-readable name for a classification code.

    Args:
        code: ASPRS, LOD2, or LOD3 classification code

    Returns:
        Human-readable name

    Example:
        >>> get_class_name(6)
        'Building'
        >>> get_class_name(32)
        'Motorway'
    """
    return ASPRS_CLASS_NAMES.get(code, f"User Defined ({code})")


def get_class_color(code: int) -> Tuple[int, int, int]:
    """
    Get RGB color for visualization of a classification code.

    Args:
        code: Classification code

    Returns:
        RGB color tuple (0-255)

    Example:
        >>> get_class_color(6)  # Building
        (255, 0, 0)
        >>> get_class_color(2)  # Ground
        (165, 82, 42)
    """
    # Standard ASPRS colors
    color_map = {
        0: (128, 128, 128),  # Never classified - gray
        1: (200, 200, 200),  # Unclassified - light gray
        2: (165, 82, 42),  # Ground - brown
        3: (144, 238, 144),  # Low vegetation - light green
        4: (60, 179, 113),  # Medium vegetation - medium green
        5: (34, 139, 34),  # High vegetation - dark green
        6: (255, 0, 0),  # Building - red
        7: (255, 255, 0),  # Low point - yellow
        8: (128, 128, 128),  # Reserved - gray
        9: (0, 0, 255),  # Water - blue
        10: (128, 0, 128),  # Rail - purple
        11: (0, 0, 0),  # Road - black
        17: (139, 69, 19),  # Bridge - saddle brown
        18: (255, 165, 0),  # High noise - orange
    }

    # Extended colors for roads (32-43)
    if 32 <= code <= 43:
        return (64, 64, 64)  # Dark gray for all roads

    # Extended colors for buildings (50-62)
    if 50 <= code <= 62:
        return (220, 20, 60)  # Crimson for all buildings

    # Extended colors for vegetation (70-76)
    if 70 <= code <= 76:
        return (50, 205, 50)  # Lime green for all vegetation

    # Extended colors for water (80-85)
    if 80 <= code <= 85:
        return (30, 144, 255)  # Dodger blue for all water

    return color_map.get(code, (128, 128, 128))


def get_classification_for_building(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get classification code for a building based on its BD TOPO® nature.

    Args:
        nature: Building nature from BD TOPO® (e.g., "Résidentiel")
        mode: Classification mode

    Returns:
        Classification code

    Example:
        >>> get_classification_for_building("Résidentiel", "asprs_extended")
        50
        >>> get_classification_for_building(None, "asprs_standard")
        6
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.BUILDING

    if nature and nature in BUILDING_NATURE_TO_ASPRS:
        return BUILDING_NATURE_TO_ASPRS[nature]

    return ASPRSClass.BUILDING


def get_classification_for_road(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get classification code for a road based on its BD TOPO® nature.

    Args:
        nature: Road nature from BD TOPO® (e.g., "Autoroute")
        mode: Classification mode

    Returns:
        Classification code
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.ROAD_SURFACE

    if nature and nature in ROAD_NATURE_TO_ASPRS:
        return ROAD_NATURE_TO_ASPRS[nature]

    return ASPRSClass.ROAD_SURFACE


def get_classification_for_vegetation(
    nature: Optional[str] = None,
    height: Optional[float] = None,
    mode: str = ClassificationMode.ASPRS_EXTENDED,
) -> int:
    """
    Get classification code for vegetation based on nature and height.

    Args:
        nature: Vegetation nature from BD TOPO® (e.g., "Arbre")
        height: Vegetation height in meters
        mode: Classification mode

    Returns:
        Classification code
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        # Use height-based classification for standard mode
        if height is not None:
            if height < 0.5:
                return ASPRSClass.LOW_VEGETATION
            elif height < 2.0:
                return ASPRSClass.MEDIUM_VEGETATION
            else:
                return ASPRSClass.HIGH_VEGETATION
        return ASPRSClass.MEDIUM_VEGETATION

    # Extended mode: use nature if available
    if nature and nature in VEGETATION_NATURE_TO_ASPRS:
        return VEGETATION_NATURE_TO_ASPRS[nature]

    # Fallback to height-based
    if height is not None:
        if height < 0.5:
            return ASPRSClass.LOW_VEGETATION
        elif height < 2.0:
            return ASPRSClass.MEDIUM_VEGETATION
        else:
            return ASPRSClass.HIGH_VEGETATION

    return ASPRSClass.MEDIUM_VEGETATION


def get_classification_for_water(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """
    Get classification code for water based on its BD TOPO® nature.

    Args:
        nature: Water nature from BD TOPO® (e.g., "Lac")
        mode: Classification mode

    Returns:
        Classification code
    """
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.WATER

    if nature and nature in WATER_NATURE_TO_ASPRS:
        return WATER_NATURE_TO_ASPRS[nature]

    return ASPRSClass.WATER


def get_classification_for_railway(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_STANDARD
) -> int:
    """Get classification code for railways (always RAIL)."""
    return ASPRSClass.RAIL


def get_classification_for_sports(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """Get classification code for sports facilities."""
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.UNCLASSIFIED
    return ASPRSClass.SPORTS_FACILITY


def get_classification_for_cemetery(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """Get classification code for cemeteries."""
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.UNCLASSIFIED
    return ASPRSClass.CEMETERY


def get_classification_for_power_line(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """Get classification code for power lines."""
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.UNCLASSIFIED
    return ASPRSClass.POWER_LINE


def get_classification_for_parking(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_EXTENDED
) -> int:
    """Get classification code for parking areas."""
    if mode == ClassificationMode.ASPRS_STANDARD:
        return ASPRSClass.UNCLASSIFIED
    return ASPRSClass.PARKING


def get_classification_for_bridge(
    nature: Optional[str] = None, mode: str = ClassificationMode.ASPRS_STANDARD
) -> int:
    """Get classification code for bridges (always BRIDGE_DECK)."""
    return ASPRSClass.BRIDGE_DECK


# ============================================================================
# Feature Requirements (for classification refinement)
# ============================================================================

# Required features for each classification type
WATER_FEATURES = ["height", "planarity", "curvature", "normals"]
ROAD_FEATURES = ["height", "planarity", "curvature", "normals", "ndvi"]
VEGETATION_FEATURES = [
    "ndvi",
    "height",
    "curvature",
    "planarity",
    "sphericity",
    "roughness",
]
BUILDING_FEATURES = ["height", "planarity", "verticality", "ndvi"]

# All unique features needed for ASPRS classification
ALL_CLASSIFICATION_FEATURES = [
    "height",  # Z above ground (meters)
    "planarity",  # Flatness measure [0-1]
    "curvature",  # Surface curvature [0-∞]
    "normals",  # Surface normal vectors [N, 3]
    "ndvi",  # Normalized Difference Vegetation Index [-1, 1]
    "sphericity",  # Shape sphericity [0-1]
    "roughness",  # Surface roughness [0-∞]
    "verticality",  # Wall-like measure [0-1]
]


# Backward compatibility aliases
LOD2_CLASSES = {cls.name.lower(): cls.value for cls in LOD2Class}
LOD3_CLASSES = {cls.name.lower(): cls.value for cls in LOD3Class}


__all__ = [
    # Enums
    "ASPRSClass",
    "LOD2Class",
    "LOD3Class",
    "ClassificationMode",
    # Mappings
    "ASPRS_TO_LOD2",
    "ASPRS_TO_LOD3",
    "LOD2_TO_ASPRS",
    "LOD3_TO_ASPRS",
    "BUILDING_NATURE_TO_ASPRS",
    "ROAD_NATURE_TO_ASPRS",
    "VEGETATION_NATURE_TO_ASPRS",
    "WATER_NATURE_TO_ASPRS",
    "RAILWAY_NATURE_TO_ASPRS",
    "SPORTS_NATURE_TO_ASPRS",
    "CEMETERY_NATURE_TO_ASPRS",
    "POWER_LINE_NATURE_TO_ASPRS",
    "PARKING_NATURE_TO_ASPRS",
    "BRIDGE_NATURE_TO_ASPRS",
    # Names and colors
    "ASPRS_CLASS_NAMES",
    # Utility functions
    "get_class_name",
    "get_class_color",
    "get_classification_for_building",
    "get_classification_for_road",
    "get_classification_for_vegetation",
    "get_classification_for_water",
    "get_classification_for_railway",
    "get_classification_for_sports",
    "get_classification_for_cemetery",
    "get_classification_for_power_line",
    "get_classification_for_parking",
    "get_classification_for_bridge",
    # Feature requirements
    "WATER_FEATURES",
    "ROAD_FEATURES",
    "VEGETATION_FEATURES",
    "BUILDING_FEATURES",
    "ALL_CLASSIFICATION_FEATURES",
    # Lidar HD 7cl contiguous (Pointcept / PTv3)
    "LIDARHD_7CL_NAMES",
    "LIDARHD_7CL_IGNORE_INDEX",
    "ASPRS_TO_LIDARHD_7CL",
    "LIDARHD_7CL_TO_ASPRS",
    "remap_asprs_to_lidarhd_7cl",
    # Backward compatibility
    "LOD2_CLASSES",
    "LOD3_CLASSES",
]
