#abaqus viewer -noGUI odb_reader.py 
# import odb abaqus library
from odbAccess import openOdb

# open odb file with the binary data
odb = openOdb('chorda_test.odb')

# save the displacements and stresses from abaqus
step1 = odb.steps['Step-1']
n_frames = len(step1.frames)
displacement = []
time = []
stress = []
for i in range(n_frames):
    time.append(step1.frames[i].frameValue)
    displacement.append(step1.frames[i].fieldOutputs['U'].values[1].data)
    stress.append(step1.frames[i].fieldOutputs['S'].values[1].data)

# write file with displacements and stresses
file1=open("chorda_results.dat","w")
for i in range(n_frames):
    file1.write("{} {} {}\n".format(time[i],displacement[i][2],stress[i][2]))

file1.close()
