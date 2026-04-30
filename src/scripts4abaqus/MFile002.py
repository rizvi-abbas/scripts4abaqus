from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import Odb
from driverUtils import executeOnCaeStartup

 
from datetime import datetime
from csv import writer, reader
from inpParser import InputFile

def rotate_vectors_around_axis1(vectors, axis, angle_degrees):
    """
    Rotate multiple vectors around a given axis by a certain angle in degrees.

    :param vectors: An array of vectors to rotate (Nx3 numpy array).
    :param axis: The axis to rotate around (3-element list or numpy array).
    :param angle_degrees: The rotation angle in degrees.
    :return: The rotated vectors as a numpy array.
    """
    np = sys.modules['numpy']
    # Convert angle from degrees to radians
    angle = np.radians(angle_degrees)

    # Normalize the axis
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula components
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Prepare for broadcasting
    cross_product = np.cross(axis, vectors)
    dot_product = np.dot(vectors, axis)

    # Rodrigues' rotation formula
    rotated_vectors = (vectors * cos_theta +
                      cross_product * sin_theta +
                      axis * dot_product[:, np.newaxis] * (1 - cos_theta))
    return rotated_vectors

    
    

def SetByNodeAngle1(partName, startPoint, SetName):
    p = mdb.models['Femur'].parts[partName]
    n = p.nodes
    nodes=n.getClosest(coordinates=((startPoint),))
    nodes=n.sequenceFromLabels((nodes[0].label,))
    faces=nodes[0].getElemFaces()
    i = 0
    while len(nodes) < 1000:
        nodes=faces[i].getNodesByFaceAngle(20)
        #print(len(nodes))
        i += 1
    p.Set(nodes=nodes, name=SetName)


def NodeByBoundingCylinder1(partName, selectDimensions, name, surfaceName=False, setName=False):
    p = mdb.models['Femur'].parts[partName]
    if surfaceName:
        n = p.surfaces[surfaceName].nodes
    elif setName:
        n = p.sets[setName].nodes
    else:
        n = p.nodes
    nodes = n.getByBoundingCylinder(*selectDimensions)
    p.Set(nodes=nodes, name=name)


def NodeByBoundingSphere1(partName, selectDimensions, name, surfaceName=False, setName=False):
    p = mdb.models['Femur'].parts[partName]
    if surfaceName:
        n = p.surfaces[surfaceName].nodes
    elif setName:
        n = p.sets[setName].nodes
    else:
        n = p.nodes
    nodes = n.getByBoundingSphere(*selectDimensions)
    p.Set(nodes=nodes, name=name)


def NodeSpehereCentroidBySet1(partName, setName):
    np = sys.modules['numpy']
    p = mdb.models['Femur'].parts[partName]
    nodes = p.sets[setName].nodes

    coordinates = np.array([node.coordinates for node in nodes])
    spX, spY, spZ = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    A = np.column_stack((2 * spX, 2 * spY, 2 * spZ, np.ones(len(spX))))

    f = spX**2 + spY**2 + spZ**2

    C, residuals, rank, singular_values = np.linalg.lstsq(A, f, rcond=None)

    ##   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return (C[0], C[1], C[2]), radius



def find_furthest_tangent_points1(center1, r1, center2, r2, axis_point1, axis_point2):
    np = sys.modules['numpy']
    # Define axis direction vector
    axis_vector = np.array(axis_point2) - np.array(axis_point1)
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Vector between centers
    c_vector = np.array(center2) - np.array(center1)

    # Normal vector perpendicular to axis and c_vector
    normal_vector = np.cross(axis_vector, c_vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate two possible tangent points for each sphere
    point1_a = np.array(center1) + r1 * normal_vector
    point1_b = np.array(center1) - r1 * normal_vector
    point2_a = np.array(center2) + r2 * normal_vector
    point2_b = np.array(center2) - r2 * normal_vector

    # Calculate distances from axis
    def distance_to_axis(point):
        return np.linalg.norm(np.cross(point - np.array(axis_point1), axis_vector))

    # Select furthest points
    tangent_point1 = point1_a if distance_to_axis(point1_a) > distance_to_axis(point1_b) else point1_b
    tangent_point2 = point2_a if distance_to_axis(point2_a) > distance_to_axis(point2_b) else point2_b

    return tangent_point1, tangent_point2



def shift_point1(point1, point2, point_to_shift, shift_distance):
    np = sys.modules['numpy']
    # Calculate the direction vector from point1 to point2
    direction_vector = np.array(point2) - np.array(point1)

    # Normalize the direction vector
    distance = np.linalg.norm(direction_vector)
    if distance == 0:
        raise ValueError("The two points defining the direction are identical.")
    direction_unit_vector = direction_vector / distance

    # Shift the point_to_shift by the specified distance in the direction of the unit vector
    shifted_point = np.array(point_to_shift) + direction_unit_vector * shift_distance

    return tuple(shifted_point)


def shift_point3(point1, point2, point3, point4, point_to_shift, distance_along_axis, distance_perpendicular):
    np = sys.modules['numpy']
    # Calculate direction vector from point1 to point2 (first axis)
    direction_vector1 = np.array(point2) - np.array(point1)
    distance1 = np.linalg.norm(direction_vector1)

    if distance1 == 0:
        raise ValueError("The two points defining the first axis are identical.")

    # Normalize the direction vector for the first axis
    direction_unit_vector1 = direction_vector1 / distance1

    # Calculate direction vector from point3 to point4 (second axis)
    direction_vector2 = np.array(point4) - np.array(point3)
    distance2 = np.linalg.norm(direction_vector2)

    if distance2 == 0:
        raise ValueError("The two points defining the second axis are identical.")

    # Normalize the direction vector for the second axis
    direction_unit_vector2 = direction_vector2 / distance2

    # Create a rotation matrix to rotate 90 degrees about direction_unit_vector1
    rotation_axis = direction_unit_vector1
    angle = np.pi / 2  # 90 degrees in radians
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = rotation_axis

    # Rodrigues' rotation formula
    rotation_matrix = np.array([
        [cos_angle + ux * ux * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
        [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy * uy * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
        [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz * uz * (1 - cos_angle)]
    ])

    # Rotate direction_unit_vector2
    rotated_vector = np.dot(rotation_matrix, direction_unit_vector2)

    # Calculate the shift vector
    shift_vector = (direction_unit_vector1 * distance_along_axis +
                    rotated_vector * distance_perpendicular)

    # Apply shift to the point
    shifted_point = np.array(point_to_shift) + shift_vector

    return tuple(shifted_point)


def getNormal1(point1,point2):
    np = sys.modules['numpy']
    axis = np.array(point2)-np.array(point1)
    axis = axis / np.linalg.norm(axis)
    return axis

    
    
def set_view_rotation1(view_name):
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    v_primary_start = np.array(session.viewports['Viewport: 1'].odbDisplay.viewCuts[view_name].normal)
    v_secondary_start = np.array(session.viewports['Viewport: 1'].odbDisplay.viewCuts[view_name].axis2)

    v_primary_target = np.array([0, 0, 1])
    v_secondary_target = np.array([0, 1, 0])

    def get_orthonormal_basis(v_fwd, v_up_hint):
        """Creates an orthogonal coordinate system from two vectors."""
        z = v_fwd / np.linalg.norm(v_fwd)
        x = np.cross(v_up_hint, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack((x, y, z))

    basis_start = get_orthonormal_basis(v_primary_start, v_secondary_start)
    basis_target = get_orthonormal_basis(v_primary_target, v_secondary_target)
    
    rot_matrix = basis_target @ basis_start.T

    euler_angles = R.from_matrix(rot_matrix).as_euler('XYZ', degrees=True)
    rx_deg, ry_deg, rz_deg = euler_angles
    
    session.viewports['Viewport: 1'].view.rotate(xAngle=rx_deg, yAngle=ry_deg, zAngle=rz_deg, mode=TOTAL)

def NodeSphereCentroidShiftBySet1(partName, setName):
    np = sys.modules['numpy']
    # Access the specified part and set within the model
    p = mdb.models['Femur'].parts[partName]
    nodes = p.sets[setName].nodes

    # Extract the coordinates of the nodes
    coordinates = np.array([node.coordinates for node in nodes])
    spX, spY, spZ = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Calculate mean point of the nodes
    meanX, meanY, meanZ = np.mean(spX), np.mean(spY), np.mean(spZ)

    # Construct the A matrix for the least squares calculation
    A = np.column_stack((2 * spX, 2 * spY, 2 * spZ, np.ones(len(spX))))

    # f vector representing squared distances from the origin
    f = spX**2 + spY**2 + spZ**2

    # Solve the linear least squares problem to find the sphere centroid
    C, residuals, rank, singular_values = np.linalg.lstsq(A, f, rcond=None)

    # Calculate the radius of the sphere
    radius = np.sqrt((C[0]**2) + (C[1]**2) + (C[2]**2) + C[3])

    # Original centroid
    cx, cy, cz = C[0], C[1], C[2]

    # Vector from centroid to mean point
    direction = np.array([meanX - cx, meanY - cy, meanZ - cz])
    norm_direction = np.linalg.norm(direction)

    # Normalize direction vector
    if norm_direction != 0:
        unit_direction = direction / norm_direction
    else:
        unit_direction = np.zeros(3)  # No shift if the direction is a zero vector

    # Shift the centroid towards the mean point by the radius
    shifted_centroid = (cx + radius * unit_direction[0], 
                        cy + radius * unit_direction[1], 
                        cz + radius * unit_direction[2])

    return shifted_centroid



def SetReferencePoint1(name, point):
    a = mdb.models['Femur'].rootAssembly
    r = a.referencePoints
    RF1=a.ReferencePoint(point=point)
    a.Set(referencePoints=(r[RF1.id], ), name=name)



def SetAxisByTwoNodes1(point1, point2):
    a = mdb.models['Femur'].rootAssembly
    a.DatumAxisByTwoPoint(point1=a.sets[point1].referencePoints[0], point2=a.sets[point2].referencePoints[0])



def BoundaryCondition1(part='Cortical without Lesion'):
    a = mdb.models['Femur'].rootAssembly
    region1=a.sets['Hip Joint Centre']
    region2=a.instances[part+'-1'].sets['Load-Face']
    mdb.models['Femur'].Coupling(name='Constraint-1', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=OFF, ur3=OFF)
    for i,v in enumerate(['M','L']):
        region1=a.sets['Condyle Centre-%s'%v]
        region2=a.instances[part+'-1'].sets['Fixed-Face-%s'%v]
        mdb.models['Femur'].Coupling(name='Constraint-%d'%(i+2), controlPoint=region1, 
            surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
            localCsys=None, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=OFF, ur3=OFF) 
    for i,v in enumerate(['01']):
        region1=a.sets[f"Muscle-{v} Point"]
        region2=a.instances[part+'-1'].sets[f"Muscle-{v}"]
        mdb.models['Femur'].Coupling(name='Constraint-%d'%(i+4), controlPoint=region1, 
            surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
            localCsys=None, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=OFF, ur3=OFF) 
            
            
def BoundaryCondition4(name,region,BCtype = 'EncastreBC'):
    if BCtype == 'EncastreBC':
        mdb.models['Femur'].EncastreBC(name=name, createStepName='Initial', 
            region=region, localCsys=None)
    elif BCtype == 'PinnedBC':
        mdb.models['Femur'].PinnedBC(name=name, createStepName='Initial', 
            region=region, localCsys=None)
    elif BCtype == 'ZasymmBC':
        mdb.models['Femur'].ZasymmBC(name=name, createStepName='Initial', 
            region=region, localCsys=None)

def BoundaryCondition5(name,region,magnitude,vector):
    np = sys.modules['numpy']
    vector = vector / np.linalg.norm(vector)
    mdb.models['Femur'].ConcentratedForce(name=name, createStepName='Step-1', 
        region=region, cf1=magnitude*vector[0], cf2=magnitude*vector[1], cf3=magnitude*vector[2],
        distributionType=UNIFORM, field='', localCsys=None)

def BoundaryCondition6(name,region,magnitude,vector):
    vector = [float(v) for v in vector]
    mdb.models['Femur'].DisplacementBC(name=name, createStepName='Step-1', 
        region=region, u1=magnitude*vector[0], u2=magnitude*vector[1], u3=magnitude*vector[2], ur1=UNSET, ur2=UNSET, ur3=UNSET, 
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)


def getSetReferencePoint(setName):
    a = mdb.models['Femur'].rootAssembly
    return a.getCoordinates(a.sets[setName].referencePoints[0])

def getSetCoordinates(job,setName,frame=-1):
    np = sys.modules['numpy']
    odb = session.openOdb(name=f'{job}.odb')
    fieldOutput = odb.steps['Step-1'].frames[frame].fieldOutputs['COORD']
    region=odb.rootAssembly.nodeSets[setName]
    return np.stack([v.data for v in fieldOutput.getSubset(region=region).values])



def LoadCase1(caseName):
    np = sys.modules['numpy']
    Hip_Joint_Centre = getSetReferencePoint('Hip Joint Centre')
    Knee_Joint_Centre = getSetReferencePoint('Knee Joint Centre')
    Shaft_Centre_P = getSetReferencePoint('P')
    Shaft_Centre_D = getSetReferencePoint('D')
    
    a=mdb.models['Femur'].rootAssembly
    for k in mdb.models['Femur'].boundaryConditions.keys():
        del mdb.models['Femur'].boundaryConditions[k]
    for k in mdb.models['Femur'].loads.keys():
        del mdb.models['Femur'].loads[k]
    #for k in mdb.models['Femur'].constraints.keys():
    #    del mdb.models['Femur'].constraints[k]
    BCtype = 'EncastreBC'
    v1 = getNormal1(Hip_Joint_Centre,Knee_Joint_Centre)
    R1 = rotation_matrix_from_vectors1(np.array([0,0,-1]),v1)
    R2 = rotation_matrix_from_vectors1(np.array([-0.91,0.21,0.36]),v1)
    match caseName:
        case 'Load Case 07':
            region = a.sets['Hip Joint Centre']
            BoundaryCondition5('Load-1',region,5000,v1)
        case 'Load Case 13':
            region = a.sets['Hip Joint Centre']
            BoundaryCondition6('Displacement-1',region,1.0, v1)
        case 'Load Case 14':
            region = a.sets['Hip Joint Centre']
            BoundaryCondition5('Load-1',region,5000,R1@np.array([0.31,0.15,-0.94]))
            region = a.sets[f"Muscle-{1:02d} Point"]
            BoundaryCondition5('Load-2',region,2000,R1@np.array([-0.27,-0.12,0.45]))
        case 'Load Case 15':
            region = a.sets['Hip Joint Centre']
            BoundaryCondition5('Load-1',region,5000,R1@np.array([0.31,0.15,-0.94]))
            region = a.sets[f"Muscle-{1:02d} Point"]
            BoundaryCondition5('Load-2',region,1000,R1@np.array([-0.27,-0.12,0.45]))
        case 'Load Case 08':
            region = a.sets['Hip Joint Centre']
            BoundaryCondition5('Load-1',region,5000,R1@np.array([0.31,0.15,-0.94]))
            region = a.sets[f"Muscle-{1:02d} Point"]
            BoundaryCondition5('Load-2',region,3000,R1@np.array([-0.27,-0.12,0.45]))
            SetReferencePoint1('Force_D', np.array(Hip_Joint_Centre) + R1@np.array([0.31,0.15,-0.94]))
            SetAxisByTwoNodes1('Hip Joint Centre','Force_D')
        case 'Load Case 09':
            region = a.sets['Hip Joint Centre']
            BoundaryCondition5('Load-1',region,5000,R2@np.array([-0.94,0.22,0.28]))
            region = a.sets[f"Muscle-{1:02d} Point"]
            BoundaryCondition5('Load-2',region,3000,R2@np.array([-0.94,-0.34,0.00]))
            SetReferencePoint1('Force_D', np.array(Hip_Joint_Centre) + R2@np.array([-0.94,0.22,0.28]))
            SetAxisByTwoNodes1('Hip Joint Centre','Force_D')
            #BoundaryCondition4("BC-04",region)
        case str() if caseName.startswith("07Angle "):
            v3 = rotate_vectors_around_axis1(np.array([v1]), np.cross(getNormal1(Shaft_Centre_P,Shaft_Centre_D),v1), 5)
            v4 = rotate_vectors_around_axis1(v3, v1, int(caseName[8:]))[0]
            print(f"v1:{v1}; v4:{v4}", file=sys.stderr, end='')
            region = a.sets['Hip Joint Centre']
            BoundaryCondition5('Load-1',region,5000,v4)
    
    for i,v in enumerate(['M','L']):
        region = a.sets[f"Condyle Centre-{v}"]
        BoundaryCondition4(f"BC-{i+2}",region,BCtype)



def rotation_matrix_from_vectors1(u, v):
    np = sys.modules['numpy']
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    cross = np.cross(u, v)
    dot = np.dot(u, v)
    if np.allclose(cross, 0):  # vectors are parallel or anti-parallel
        return np.eye(3) if dot > 0 else -np.eye(3)
    skew = np.array([[0, -cross[2], cross[1]],
                     [cross[2], 0, -cross[0]],
                     [-cross[1], cross[0], 0]])
    R = np.eye(3) + skew + skew @ skew * ((1 - dot) / (np.linalg.norm(cross)**2))
    return R
    

def solve2(jobName, description=''):
    mdb.Job(name=jobName, model='Femur', description=description, type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, 
        numDomains=4, numGPUs=0)
    mdb.jobs[jobName].writeInput(consistencyChecking=OFF)
    del mdb.jobs[jobName]

def solve3(jobName, path1=''):
    mdb.JobFromInputFile(name=jobName, inputFileName=path1+jobName+'.inp', 
        type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
        memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, 
        numDomains=4, numGPUs=0)
    mdb.jobs[jobName].submit(consistencyChecking=OFF)
    mdb.jobs[jobName].waitForCompletion()








def createDF1(job,info,output_file,caseName='None',subject='None',group='None',frame=-1):
    np = sys.modules['numpy']
    # Write header if creating new file
    header = ['Subject','Group','Load Case','Lesion','Position: R','Position: Z','Position: Y','Position: X','Position: V','Position: I','Displacement-Head: U-U1','Displacement_U-U1','Displacement_U-U2','Displacement_U-U3','Reaction_Medial','Reaction_Lateral','TimeStamp','Weighted Average Elasticity',]
    new_row = {}

    odb = session.openOdb(name=f'{job}.odb')

    field_output = odb.steps['Step-1'].frames[frame].fieldOutputs['U']
    #new_row['Displacement_U-Magnitude'] = field_output.getSubset(region=odb.rootAssembly.nodeSets['Hip Joint Centre']).values[0].magnitude
    new_row['Displacement-Head: U-U1'], new_row['Displacement-Head: U-U2'], new_row['Displacement-Head: U-U3'] = field_output.getSubset(region=odb.rootAssembly.nodeSets['Hip Joint Centre']).values[0].data

    field_output = odb.steps['Step-1'].frames[frame].fieldOutputs['RT']
    #new_row['Reaction_Medial'] = field_output.getSubset(region=odb.rootAssembly.nodeSets['Condyle Centre-M']).values[0].magnitude
    new_row['Reaction-Medial: RT-RT1'], new_row['Reaction-Medial: RT-RT2'], new_row['Reaction-Medial: RT-RT3'] = field_output.getSubset(region=odb.rootAssembly.nodeSets['Condyle Centre-M']).values[0].data
    #new_row['Reaction_Lateral'] = field_output.getSubset(region=odb.rootAssembly.nodeSets['Condyle Centre-L']).values[0].magnitude
    new_row['Reaction-Lateral: RT-RT1'], new_row['Reaction-Lateral: RT-RT2'], new_row['Reaction-Lateral: RT-RT3'] = field_output.getSubset(region=odb.rootAssembly.nodeSets['Condyle Centre-L']).values[0].data


    if group in ['a-p','m-l','neck']:
        odb = session.openOdb(name='Job-6.odb')
        field_output = odb.steps['Step-1'].frames[frame].fieldOutputs['elasticity']
        n1=[v.data for v in field_output.getSubset(region=odb.rootAssembly.instances['PART-1-1'].elementSets['LESIONELEMENTS']).values]
        field_output = odb.steps['Step-1'].frames[frame].fieldOutputs['EVOL']
        w1=[v.data for v in field_output.getSubset(region=odb.rootAssembly.instances['PART-1-1'].elementSets['LESIONELEMENTS']).values]
        new_row['Weighted Average Elasticity'] = np.sum(np.prod(np.array([n1,w1]),axis=0))/np.sum(w1)
    else:
        new_row['Weighted Average Elasticity'] = ''
    
    with open(output_file, 'a') as f:
        writer(f, lineterminator='\n').writerow([
            subject,
            group,
            caseName,
            f"Mode {frame}",
            info[0],
            info[1][0],
            info[1][1],
            info[1][2],
            info[1][3],
            info[2],
            info[3],
            new_row['Displacement-Head: U-U1'],
            new_row['Displacement-Head: U-U2'],
            new_row['Displacement-Head: U-U3'],
            new_row['Reaction-Medial: RT-RT1'],
            new_row['Reaction-Medial: RT-RT2'],
            new_row['Reaction-Medial: RT-RT3'],
            new_row['Reaction-Lateral: RT-RT1'],
            new_row['Reaction-Lateral: RT-RT2'],
            new_row['Reaction-Lateral: RT-RT3'],
            datetime.now().strftime("%Y%m%d%H%M%S"),
            new_row['Weighted Average Elasticity'],
            ])



class InpFileValuesReader1():
    np = sys.modules['numpy']
    def __init__(self, inputFileName):                
        if not inputFileName.exists() and inputFileName.with_suffix('.inp.zip').exists():
            import zipfile
            with zipfile.ZipFile(inputFileName.with_suffix('.inp.zip'), 'r') as f:
                f.extract(inputFileName.name, inputFileName.parent)
            
        self.inp = InputFile(f"{inputFileName}")
        self.inp.parse(organize=True,usePyArray=True)
        self.values = {}
        self.values['solidsection'] = {v.parameter['elset']:v.parameter['material'] for v in self.inp.keywords if v.name == 'solidsection'}
        self.values['node'] = [v.data for v in self.inp.keywords if v.name == 'node']
        self.values['element'] = {v.parameter['elset']:v.data for v in self.inp.keywords if v.name == 'element'}
        self.values['GSvalues'] = { v.parameter['elset']: int(re.search("GS=-?\\d+", v.comments[-2]).group()[3:]) for v in self.inp.keywords if v.name == 'solidsection' }
        
    def __getitem__(self, index):
        required = {index}
        if 'mat_vol' in required:
            required |= {'element','element_vollist'}
        if 'element_vollist' in required:
            required |= {'element_nodelist'}
        if 'element_nodelist' in required:
            required |= {'node_list','all_element'}
        if 'all_element' in required:
            required |= {'element'}
        if 'node_list' in required:
            required |= {'node'}
        
        required = {v for v in required if v not in self.values}
        
        if 'node_list' in required:
            self.values['node_list'] = {int(v[0]):(v[1:4]) for w in self.values['node'] for v in w}
        if 'all_element' in required:
            self.values['all_element'] = {v[0]:(w[0],v[1:]) for w in self.values['element'].items() for v in w[1]}
        if 'element_nodelist' in required:
            self.values['element_nodelist'] = {k:np.array([self.values['node_list'][v] for v in w[1]]) for k,w  in self.values['all_element'].items()}
        if 'element_vollist' in required:
            self.values['element_vollist'] = dict(zip(self.values['element_nodelist'].keys(),list(tetrahedron_volumes(np.stack(self.values['element_nodelist'].values(),axis=0)))))
        if 'mat_vol' in required:
            self.values['mat_vol']={k:sum([self.values['element_vollist'][u] for u in list(v.reshape(-1))]) for k,v in self.values['element'].items()}
        return self.values[index]

def getGS2E1(readInpFileValues1, s):
    np = sys.modules['numpy']
    Houns_U = np.array(list(readInpFileValues1['GSvalues'].values()))
    select = {
        'A' : (1453,6799),
        'C' : (6381,6975),
        'E' : (4882,6657),
    }
    a,b = select[s]
    rho_QCT = a*1e-5 + b*1e-7 * Houns_U
    rho_QCT = -1.45E-02 + 6.80E-04 * Houns_U
    rho_ash = 6.33E-02 + 8.87E-01 * rho_QCT
    rho_app = rho_ash / 0.626
    rho_app = np.clip(rho_app, 0.01, None)
    return dict(zip(readInpFileValues1['solidsection'].values(), list(6850 * rho_app**1.49)))




def ElementByBoundingSphere1(partName, selectDimensions, ElementsName):
    p = mdb.models['Femur'].parts[partName]
    e = p.elements
    elements = e.getByBoundingSphere(*selectDimensions)
    p.Set(elements=elements, name=ElementsName)

def NodeByBoundingSphere1(partName, selectDimensions, name, surfaceName=False, setName=False):
    p = mdb.models['Femur'].parts[partName]
    if surfaceName:
        n = p.surfaces[surfaceName].nodes
    elif setName:
        n = p.sets[setName].nodes
    else:
        n = p.nodes
    nodes = n.getByBoundingSphere(*selectDimensions)
    p.Set(nodes=nodes, name=name)


def ElementByElementLabel1(source, target, frame=-1):
    jobName1, instanceName1, name1 = source
    jobName2, instanceName2, name2 = target

    odb = session.openOdb(name=f"{jobName1}.odb")
    field_output = odb.steps['Step-1'].frames[frame].fieldOutputs['EVOL']
    elementLabels = [v.elementLabel for v in field_output.getSubset(
        region=odb.rootAssembly.instances[instanceName1].elementSets[name1]).values]

    odb = session.openOdb(name=f"{jobName2}.odb")
    odb.rootAssembly.instances[instanceName2].ElementSetFromElementLabels(name=name2,elementLabels=elementLabels)
    
    
def writeFieldReport1(job,output_file):
    odb = session.openOdb(name=job)
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    leaf = dgo.LeafFromPartInstance(partInstanceName=('ASSEMBLY', ))
    session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)
    session.fieldReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES)
    session.writeFieldReport(
        fileName=output_file,
        append=OFF, sortItem='Element Label', odb=odb, step=0, frame=1, 
        outputPosition=INTEGRATION_POINT, variable=(('E', INTEGRATION_POINT), ), 
        stepFrame=SPECIFY)
    session.viewports['Viewport: 1'].setValues(displayedObject=None)
    odb.close()
    

def create_viewports1(metadata, views=['Neck-Axis','Neck-Plane','Anteversion-Plane','Shaft-Axis','Shaft-Plane']):
    for v in views:
        if v in session.viewports['Viewport: 1'].odbDisplay.viewCuts:
            del session.viewports['Viewport: 1'].odbDisplay.viewCuts[v]
            
    if 'Neck-Axis' in views:            
        session.viewports['Viewport: 1'].odbDisplay.ViewCut(name='Neck-Axis', shape=CYLINDER, origin=tuple(eval(metadata['Neck_Centre'])),
            cylinderAxis=tuple(eval(metadata['axis_05'])), 
            followDeformation=True, referenceFrame=CURRENT_FRAME)
    if 'Neck-Plane' in views:
        session.viewports['Viewport: 1'].odbDisplay.ViewCut(name='Neck-Plane', shape=PLANE, origin=tuple(eval(metadata['Neck_Centre'])),
            normal=tuple(eval(metadata['axis_05'])),
            axis2=tuple(eval(metadata['axis_06'])),
            followDeformation=True)
    if 'Anteversion-Plane' in views:
        session.viewports['Viewport: 1'].odbDisplay.ViewCut(name='Anteversion-Plane', 
            shape=PLANE, origin=tuple(eval(metadata['Neck_Centre'])),
            normal=tuple(eval(metadata['axis_07'])),
            axis2=tuple(eval(metadata['axis_06'])),
            followDeformation=True)
    if 'Shaft-Axis' in views:
        session.viewports['Viewport: 1'].odbDisplay.ViewCut(name='Shaft-Axis', shape=CYLINDER, origin=tuple(eval(metadata['Hip_Joint_Centre'])),
            cylinderAxis=tuple(eval(metadata['axis_06'])), 
            followDeformation=True, referenceFrame=CURRENT_FRAME)
    if 'Shaft-Plane' in views:
        session.viewports['Viewport: 1'].odbDisplay.ViewCut(name='Shaft-Plane', shape=PLANE, origin=tuple(eval(metadata['Hip_Joint_Centre'])),
            normal=tuple(eval(metadata['axis_06'])),
            axis2=tuple(eval(metadata['axis_02'])),
            followDeformation=True)
        
        
def create_image1(fileName,legend=False):
    fileName.parent.mkdir(parents=True, exist_ok=True)
    session.printOptions.setValues(vpDecorations=OFF)
    session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(legend=ON if legend else OFF)
    if fileName.suffix == '.png':
        session.printToFile(fileName=f"{fileName}", format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))
    elif fileName.suffix == '.eps':
        session.epsOptions.setValues(resolution=DPI_1200, shadingQuality=EXTRA_COARSE)
        session.printToFile(fileName=f"{fileName}", format=EPS, canvasObjects=(session.viewports['Viewport: 1'], ))
######################################


def create_odb_from_inp1(inp_filepath, odb_filepath):
    np = sys.modules['numpy']
    reader = InpFileValuesReader1(inp_filepath)
    for odb in session.odbs.values():
        odb.close()
    odb = Odb(name='Generated_ODB', analysisTitle='Imported from INP', path=odb_filepath)
    part = odb.Part(name='PART-1', embeddedSpace=THREE_D, type=DEFORMABLE_BODY)
    step = odb.Step(name='Step-1', description='', domain=TIME, timePeriod=1.0)
    frame = step.Frame(incrementNumber=1, frameValue=0., description='Increment 1')
    combined_nodes = np.vstack(reader['node'])
    node_labels = tuple(combined_nodes[:, 0].astype(int).tolist())
    node_coords = tuple(map(tuple, combined_nodes[:, 1:4].tolist()))
    part.addNodes(labels=node_labels, coordinates=node_coords)
    
    combined_array = np.vstack(list(reader.values['element'].values())).astype(int)
    elem_labels = tuple(combined_array[:, 0].tolist())
    connectivity = tuple(map(tuple, combined_array[:, 1:].tolist()))
    elem_map = {5: 'C3D4', 7: 'C3D6', 9: 'C3D8', 11: 'C3D10'}
    part.addElements(labels=elem_labels, connectivity=connectivity, type=elem_map[combined_array.shape[1]])
    
    instance = odb.rootAssembly.Instance(name='PART-1-1', object=part)
    odb.save()
    odb.close()
    
def add_result_to_odb1(odbFile, file_path, fields):
    for odb in session.odbs.values():
        odb.close()
    odb = session.openOdb(name=odbFile,readOnly=False)
    #odb = openOdb(path=odbFile, readOnly=False)
    for field in fields:
        data = TensorsGet_data1(file_path, f"{field}.npy")
        data = tuple((v,) for v in data)
        labels = tuple(range(1,len(data)+1))
        createFeild1(odb,'PART-1-1',labels,data,field,'')
    odb.save()
    odb.close()
    
def createEqStrainRatio1(name, odbFile, subject, lesion):
    np = sys.modules['numpy']
    file_path0 = f"/exports/eddie/scratch/s2215928/femur/{subject}/Cadaver{subject}.zip"
    file_path1 = f"/exports/eddie/scratch/s2215928/femur/{subject}/{lesion}.zip"
    
    odb = session.openOdb(name=odbFile)
    odb.close()
    odb = session.openOdb(name=odbFile,readOnly=False)
    data0 = TensorsGet_data1(file_path0, 'E_mises.npy')
    data1 = TensorsGet_data1(file_path1, 'E_mises.npy')
    
    data = np.where(data0 > data1, (data0 / data1) - 1, 1 - (data1 / data0))
    data = tuple((v,) for v in data)
    
    labels = tuple(range(1,len(data)+1))
    createFeild1(odb,'PART-1-1',labels,data,name,'')
    odb.save()
    odb.close()

def createFeild1(odb,instance,labels,data,name,description):
    tmpField = odb.steps['Step-1'].frames[-1].FieldOutput(name=name, description=description, type=SCALAR)
    tmpField.addData(position=INTEGRATION_POINT, instance=odb.rootAssembly.instances[instance], labels=labels, data=data)



def calculateEqStrain1(array):
    np = sys.modules['numpy']
    e11 = array[:, 0]
    e22 = array[:, 1]
    e33 = array[:, 2]
    
    e12 = array[:, 3] / 2.0
    e23 = array[:, 4] / 2.0
    e31 = array[:, 5] / 2.0

    return (2.0 / 3.0) * np.sqrt( (e11**2 + e22**2 + e33**2 - e11*e22 - e22*e33 - e33*e11) + 3.0 * (e12**2 + e23**2 + e31**2) )



def TensorsAppend_data1(file_path, data_name, data_value, unique_string=False):
    np = sys.modules['numpy']
    import zipfile
    import io
    if unique_string:
        names = [f"{data_name}_{v}" for v in ['stringKey','stringData']]
        values = list(np.unique(data_value.to_numpy().astype('U'), return_inverse=True))
    else:
        names = [data_name]
        values = [data_value]
    for data_name, data_value in zip(names,values):
        buffer = io.BytesIO()
        np.save(buffer, data_value.squeeze())
        with zipfile.ZipFile(file_path, 'a', compression=zipfile.ZIP_DEFLATED) as f:
            f.writestr(f"{data_name}.npy", buffer.getvalue())

def TensorsAppend_metadata1(file_path, metadata):
    import zipfile
    import json
    with zipfile.ZipFile(file_path, 'a', compression=zipfile.ZIP_DEFLATED) as f:
        f.writestr('metadata.json', json.dumps(metadata))

def TensorsGet_metadata1(file_path):
    import zipfile
    import json
    with zipfile.ZipFile(file_path, 'r') as z:
        data = z.read('metadata.json')
        return json.loads(data)

def TensorsGet_data1(file_path, internal_file):
    import zipfile
    np = sys.modules['numpy']
    
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open(internal_file) as f:
            return np.load(f)


            
def writeFieldTensors1(job,output_file,fields,metadata={}):
    from pathlib import Path
    import numpy as np
    file_path = Path(output_file)
    file_path.unlink(missing_ok=True)
    position = set()
    odb = session.openOdb(name=job)
    if 'SectionName' in fields:
        fields.remove('SectionName')
        #append_data(file_path,'SectionName',df[['Section Name']],unique_string=True)

    
    apply_func_all = {
        ('E','mises'): (calculateEqStrain1, 'data'),
    }
    
    for field in fields:
        field2 = field.split('_')
        if len(field2) == 2:
            apply_func, field2[1] = apply_func_all.get(tuple(field2), (None, field2[1]))
            data = np.array([getattr(v,field2[1]) for v in odb.steps['Step-1'].frames[-1].fieldOutputs[field2[0]].values])
            if apply_func is not None: data = apply_func(data)
            TensorsAppend_data1(file_path, field, data )
            position.add(odb.steps['Step-1'].frames[-1].fieldOutputs[field2[0]].values[0].position)

    for coord in list(position):
        TensorsAppend_data1(file_path,f"COORD_{coord}",np.array([getattr(v,'data') for v in odb.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=coord).values]) )
    #TensorsAppend_data1(file_path,'COORD_NODAL',np.array([getattr(v,'data') for v in odb.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values]) )
    #TensorsAppend_data1(file_path,'COORD_INTEGRATION_POINT',np.array([getattr(v,'data') for v in odb.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=INTEGRATION_POINT).values]) )
    
    

    metadata
    TensorsAppend_metadata1(file_path, metadata)
    odb.close()    
    
######################################

def getMedialRatio1(jobName, frame=-1):
    odb = session.openOdb(name=f"{jobName}.odb")
    field_output = odb.steps['Step-1'].frames[frame].fieldOutputs['RT']
    m = field_output.getSubset(region=odb.rootAssembly.nodeSets['Condyle Centre-M']).values[0].magnitude
    l = field_output.getSubset(region=odb.rootAssembly.nodeSets['Condyle Centre-L']).values[0].magnitude
    return m/(m+l)
    
    
def getDisplace1(jobName,frame=-1):
    odb = session.openOdb(name=f"{jobName}.odb")
    field_output = odb.steps['Step-1'].frames[frame].fieldOutputs['U']
    return field_output.getSubset(region=odb.rootAssembly.nodeSets['Hip Joint Centre']).values[0].data
    
    
def calculateAxisPlaneAngles1(A, B, C, D, E, F):
    np = sys.modules['numpy']
    # Calculate the normal vector to the plane (A, B)
    N = np.array(B) - np.array(A)

    # Calculate CD
    CD = np.array(D) - np.array(C)

    # Project CD onto the plane
    CD_proj = CD - np.dot(CD, N) / np.dot(N, N) * N

    # Calculate EF
    EF = np.array(F) - np.array(E)

    # Angle between EF and the plane
    cos_theta_plane = np.dot(EF, N) / (np.linalg.norm(EF) * np.linalg.norm(N))
    theta_plane = np.arccos(np.abs(cos_theta_plane))

    # Projection of EF on the plane
    EF_proj = EF - np.dot(EF, N) / np.dot(N, N) * N

    # Normalize vectors
    CD_proj_norm = CD_proj / np.linalg.norm(CD_proj)
    EF_proj_norm = EF_proj / np.linalg.norm(EF_proj)

    # Angle between the projection of EF on the plane and CD_proj
    cos_theta_proj = np.dot(CD_proj_norm, EF_proj_norm)
    theta_proj = np.arccos(cos_theta_proj)

    # Return the angles in degrees
    return np.degrees(theta_plane), np.degrees(theta_proj)
