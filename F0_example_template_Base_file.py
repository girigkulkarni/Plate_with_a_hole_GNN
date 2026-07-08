# -*- coding: mbcs -*-

# =============================================================================
# PARAMETER BLOCK: EDITABLE VALUES
# =============================================================================

# Model Details
MODEL_NAME = 'Model_'+'${counter}'
PART_NAME = 'Plate'
SKETCH_NAME = '__profile__'
MATERIAL_NAME = '${Material}'
BC_LOC = '${bc_loc}'

# Geometry Coordinates (Rectangle)
RECT_P1 = (0.0,0.0)
RECT_P2 = ( ${length} , ${breadth})

# Geometry Coordinates (Circle)
CIRCLE_CENTER = (${length}/2+${Circle_pos_X},${breadth}/2+${Circle_pos_Y})
CIRCLE_P1 = (${length}/2+${Circle_pos_X},${breadth}/2+${Circle_pos_Y}+${Cirlce_radius})

if MATERIAL_NAME=='Steel':
    # Material Properties (E, Nu)
    E_MODULUS = 210000.0
    NU = 0.3
else:
    # Material Properties (E, Nu)
    E_MODULUS = 70000.0
    NU = 0.33

# Analysis Step and Load Parameters
STEP_NAME = 'Step-1'
LOAD_PRESSURE_MAGNITUDE = -${Load} # Applied pressure magnitude (Negative sign indicates direction)
if BC_LOC=='BC_1':
    BC_FIXATION_EDGE = ((7.5, 0.0, 0.0),) # Coordinates for fixed edges
    LOAD_APPLYING_EDGE = ((2.5, ${breadth}, 0.0),) # Coordinates for pressure application
else:
    BC_FIXATION_EDGE = ((0., 10.0, 0.0),) # Coordinates for fixed edges
    LOAD_APPLYING_EDGE = ((${length}, ${breadth}-1, 0.0),) # Coordinates for pressure application
REGION_PICK_COORDS = ((6.921789, 14.61592, 0.0), ) # Coordinates for selecting face/region

# Job Parameters
JOB_NAME = 'l_{0}_b_{1}_Cr_{2}_CrC_X_{3}_CrC_Y_{4}_Ld_{5}_M_{6}_bc_{7}'.format(\
    ${length},
    ${breadth},
    ${Cirlce_radius},
    ${Circle_pos_X},
    ${Circle_pos_Y},
    ${Load},
    '${Material}',
    '${bc_loc}'
)
file_name = 'plate_'+'${counter}'
JOB_MEMORY_PERCENT = 90

# =============================================================================
# LIBRARY IMPORTS
# =============================================================================
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import __main__
executeOnCaeStartup()
# =============================================================================
# STEP 1: SKETCH AND PART CREATION
# =============================================================================
print("--- Creating Sketch and Part ---")
mdb.Model(name=MODEL_NAME, modelType=STANDARD_EXPLICIT)
s = mdb.models[MODEL_NAME].ConstrainedSketch(name=SKETCH_NAME, 
    sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=RECT_P1, point2=RECT_P2)
s.CircleByCenterPerimeter(center=CIRCLE_CENTER, point1=CIRCLE_P1)
p = mdb.models[MODEL_NAME].Part(name=PART_NAME, dimensionality=TWO_D_PLANAR, 
    type=DEFORMABLE_BODY)
p = mdb.models[MODEL_NAME].parts[PART_NAME]
p.BaseShell(sketch=s)
s.unsetPrimaryObject()
p = mdb.models[MODEL_NAME].parts[PART_NAME]
del mdb.models[MODEL_NAME].sketches[SKETCH_NAME]

print("--- Creating Material and Section ---")
# Material Definition
mdb.models[MODEL_NAME].Material(name=MATERIAL_NAME)
mdb.models[MODEL_NAME].materials[MATERIAL_NAME].Elastic(table=((E_MODULUS, NU), ))

# Section Definition
mdb.models[MODEL_NAME].HomogeneousSolidSection(name=MATERIAL_NAME, material=MATERIAL_NAME, 
    thickness=None)

# Apply section to the part
p = mdb.models[MODEL_NAME].parts[PART_NAME]
f = p.faces
faces = f.findAt(REGION_PICK_COORDS)
region = p.Set(faces=faces, name='Set-1')
p = mdb.models[MODEL_NAME].parts[PART_NAME]
p.SectionAssignment(region=region, sectionName=MATERIAL_NAME, offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)

# =============================================================================
# STEP 2: ASSEMBLY AND INSTANTIATION
# =============================================================================
print("--- Setting up Assembly ---")
a = mdb.models[MODEL_NAME].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models[MODEL_NAME].parts[PART_NAME]
a.Instance(name='Plate-1', part=p, dependent=ON)

# =============================================================================
# STEP 3: JOB SETUP (STEPS, LOADS, BCs)
# =============================================================================
print("--- Defining Analysis Steps and Loads ---")

# Static Step Definition
mdb.models[MODEL_NAME].StaticStep(name=STEP_NAME, previous='Initial', nlgeom=ON)

###################
#del mdb.models[MODEL_NAME].fieldOutputRequests['F-Output-1']
######################
# Outputs
mdb.models[MODEL_NAME].FieldOutputRequest(
    name='F-Output-2',
    createStepName=STEP_NAME,
    variables=('S', 'LE', 'U', 'COORD', 'NFORC')
)
####################
# Apply Pressure Load
a = mdb.models[MODEL_NAME].rootAssembly
s1 = a.instances['Plate-1'].edges
side1Edges1 = s1.findAt(LOAD_APPLYING_EDGE)
region = a.Surface(side1Edges=side1Edges1, name='Surf-1')
mdb.models[MODEL_NAME].Pressure(name='Pressure', createStepName=STEP_NAME, 
    region=region, distributionType=UNIFORM, field='', magnitude=10.0, 
    amplitude=UNSET)
mdb.models[MODEL_NAME].loads['Pressure'].setValues(magnitude=LOAD_PRESSURE_MAGNITUDE)

# Apply Fixed Boundary Condition
a = mdb.models[MODEL_NAME].rootAssembly
e1 = a.instances['Plate-1'].edges
edges1 = e1.findAt(BC_FIXATION_EDGE)
region = a.Set(edges=edges1, name='Set-1')
mdb.models[MODEL_NAME].DisplacementBC(name='Fix', createStepName=STEP_NAME, 
    region=region, u1=0.0, u2=0.0, ur3=0.0, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)

# =============================================================================
# STEP 4: MESH GENERATION
# =============================================================================
print("--- Generating Mesh ---")
p = mdb.models[MODEL_NAME].parts[PART_NAME]
f = p.faces
pickedRegions = f.findAt(REGION_PICK_COORDS)
# Mesh pass 1
p.setMeshControls(regions=pickedRegions, elemShape=TRI)
p = mdb.models[MODEL_NAME].parts[PART_NAME]
p.seedPart(size=1.5, deviationFactor=0.01, minSizeFactor=0.1)
p = mdb.models[MODEL_NAME].parts[PART_NAME]
p.generateMesh()
# Assembly Regeneration
a1 = mdb.models[MODEL_NAME].rootAssembly
a1.regenerate()
# =============================================================================
# STEP 5: JOB SUBMISSION
# =============================================================================
print("--- Submitting Job ---")
mdb.Job(name=file_name, model=MODEL_NAME, description=JOB_NAME, type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=JOB_MEMORY_PERCENT, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB)
mdb.jobs[file_name].writeInput(consistencyChecking=OFF)