#abaqus viewer -noGUI odb_reader.py 
# import odb abaqus library
from odbAccess import openOdb
import numpy as np

# open odb file with the binary data
odb = openOdb('biaxial_job0.odb')

# save the displacements and stresses from abaqus
step1 = odb.steps['Step-1']
n_frames = len(step1.frames)
displacement = []
time = []
stress = []
for i in range(n_frames):
    time.append(step1.frames[i].frameValue)
    displacement.append(step1.frames[i].fieldOutputs['U'].values[309].data)
    stress.append(step1.frames[i].fieldOutputs['S'].values[309].data)

time_np = np.array(time,copy=True)
displacement_np = np.array(displacement,copy=True)
stress_np = np.array(stress,copy=True)

# write file with displacements and stresses
file1=open("biaxial_results.dat","w")
for i in range(n_frames):
    file1.write("{} {} {} {} {}\n".format(time[i],displacement_np[i][0],displacement_np[i][1],stress_np[i][0],stress_np[i][1]))

file1.close()

