
def AvgChange(data):
	differences = []
	for x in range(1,len(data)):
		differences.append(data[x]-data[x-1])
	return sum(differences)/len(differences)
