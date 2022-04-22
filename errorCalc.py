"""
This program should be used to compute error metrics comparing real and predicted
porosity measures resulted from each iteration of some exec file.
The current error metrics are: MAE and RMSE

The two input files lines should follow the pattern:
X Y Z Value

X, Y and Z should be integers and Value will be converted to a float

Both files should order its lines by the X,Y and Z values

At the end, the error metrics are printed with the pattern:

ErrorMetricName: value
""" 

import sys
from timeit import default_timer as timer

def printUsage():
    print("Error! Argc is not 3. Check Usage!")
    print("Usage: python errorCalc.py realValuesFileName predictedValuesFileName")

def printErrors(realValuesFileName, predictedValuesFileName):
    rmse = 0
    mae = 0
    
    rmse, mae = computeErrors(realValuesFileName, predictedValuesFileName)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

def computeErrors(realValuesFileName, predictedValuesFileName):
    """
    Compute the RMSE and MAE.
    Assumes that all points in both files are ordered by X, Y and Z positions

    realValuesFileName, predictedValuesFileName: the input file names
    Assumes that the predictedValuesFile has a header on it
    return: RMSE, MAE
    """
    partialRMSESum = 0
    partialMAESum = 0
    valuesCount = 0
    predEmptyLines = 0

    REALFILESEP = " "
    PREDFILESEP = ","

    with open(realValuesFileName, 'r') as realValuesFile, open(predictedValuesFileName, 'r') as predValuesFile:

        #First read to pass header
        predValuesFile.readline()
        currRealValueDict = getStructuredLineFrom(realValuesFile, REALFILESEP)

        #predValuesFile probably has much less lines than realValuesFile
        for line in predValuesFile:

            if line.strip() != "":
            
                currPredValueDict = getStructuredLineFromLine(line, PREDFILESEP)

                while not isTheSamePoint(currPredValueDict, currRealValueDict):
                    
                    currRealValueDict = getStructuredLineFrom(realValuesFile, REALFILESEP)
                
                #Found a matching point on realValuesFile
                predDiff = currRealValueDict['phi']-currPredValueDict['phi']
                partialRMSESum+=(predDiff)**2
                partialMAESum+= abs(predDiff)
                valuesCount += 1
            else:
                predEmptyLines +=1

    print(f"Pontos contabilizados: {valuesCount}")
    print(f"Linhas Vazias Pred: {predEmptyLines}")

    rmse = (partialRMSESum/valuesCount)**(1/2)
    mae = partialMAESum/valuesCount

    return rmse, mae

def getStructuredLineFrom(myFile, sep=" "):
    """
    Returns a Dict with x,y,z,phi as keys based on the line read from file.
    See getStructuredLineFromLine(line)
    myFile: The file that will read a line
    """
    line = myFile.readline()
    return getStructuredLineFromLine(line, sep)

def getStructuredLineFromLine(line, sep):
    """
    Returns a Dict with x,y,z,phi as keys based on the line read from file.
    
    The split assumes that the values are in order: x y z ... phi

    line: A line from file that has not been split
    sep: The file separator.
    """
    lineSplit = line.split(sep)
    structLine = {}
    try:
        structLine['x'] = int(lineSplit[0])
        structLine['y'] = int(lineSplit[1])
        structLine['z'] = int(lineSplit[2])
        structLine['phi'] = float(lineSplit[-1])
    except:
        print(f"linha que fugiu do padrão: {line}")

    return structLine
    
def isTheSamePoint(point1, point2):
    """
    Compares the two Points and returns if they are the same.
    point1, point2: Both must have x,y and z keys. These keys are used to compare both points
    """
    return (point1["x"] == point2['x'] and point1["y"] == point2['y'] and point1["z"] == point2['z'])

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        printUsage()
    else:
    
        realValuesFileName = sys.argv[1]
        predictedValuesFileName = sys.argv[2]

        start = timer()
        printErrors(realValuesFileName, predictedValuesFileName)
        end = timer()
        print(f"Elapsed Time: {end-start}")
