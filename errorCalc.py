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
from sklearn.metrics import mean_squared_error, mean_absolute_error

def printUsage():
    print("Error! Argc is not 3. Check Usage!")
    print("Usage: python errorCalc.py realValuesFileName predictedValuesFileName")

def getStructuredLineFromLine(line):
    """
    Returns a Dict with X,Y,Z,Value as keys based on the line read from file.
    The split assumes that values are separated with a space.
    Also, the split assumes that the values are in order: X Y Z Value

    line: A line from file that has not been split
    """
    #Split based on spaces
    lineSplit = line.split()
    structLine = {}
    structLine['X'] = int(lineSplit[0])
    structLine['Y'] = int(lineSplit[1])
    structLine['Z'] = int(lineSplit[2])
    structLine['Value'] = float(lineSplit[3])
    return structLine

def getStructuredLineFrom(myFile):
    """
    Returns a Dict with X,Y,Z,Value as keys based on the line read from file.
    See getStructuredLineFromLine(line)
    myFile: The file that will read a line
    """
    line = myFile.readline()
    return getStructuredLineFromLine(line)

def isTheSamePoint(point1, point2):
    """
    Compares the two Points and returns if they are the same.
    point1, point2: Both are Dicts that must have X,Y and Z keys. These keys are used to compare both points
    """
    return (point1["X"] == point2['X'] and point1["Y"] == point2['Y'] and point1["Z"] == point2['Z'])

def getRealAndPredictedValues(realValuesFileName, predictedValuesFileName):
    """
    Returns two lists with the predicted values and its corresponding real values from the files
    Assumes that all points in both files are ordered by X, Y and Z positions

    realValuesFileName, predictedValuesFileName: the input file names
    return: predictedValues, realValues
    """
    predictedValues = []
    realValues = []

    with open(realValuesFileName, 'r') as realValuesFile, open(predictedValuesFileName, 'r') as predValuesFile:
        #Reads a line from realValuesFile
        currentRealValueDict = getStructuredLineFrom(realValuesFile)

        #predValuesFile probably has much less lines than realValuesFile
        for line in predValuesFile:
            
            currPredValueDict = getStructuredLineFromLine(line)

            while not isTheSamePoint(currPredValueDict, currentRealValueDict):
                #Reads other line from realValuesFile
                currentRealValueDict = getStructuredLineFrom(realValuesFile)
            
            #Found a matching point on realValuesFile
            predictedValues.append(currPredValueDict['Value'])
            realValues.append(currentRealValueDict['Value'])
    
    return predictedValues, realValues

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        printUsage()
    else:
    
        realValuesFileName = sys.argv[1]
        predictedValuesFileName = sys.argv[2]

        predictedValues = []
        realValues = []
        predictedValues, realValues = getRealAndPredictedValues(realValuesFileName, predictedValuesFileName)

        #print(f"Predicted Values Count: {len(predictedValues)}")
        #print(f"Real Values Count: {len(realValues)}")
        
        #squared = False means Root Mean Squared
        rmse = mean_squared_error(realValues, predictedValues, squared=False)
        mae = mean_absolute_error(realValues, predictedValues)

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")