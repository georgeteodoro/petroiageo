"""
This program should be used to compute error metrics comparing real and predicted
porosity measures resulted from each iteration of some exec file.
The current error metrics are: MAE and RMSE

The two input files lines should follow the pattern:
X Y Z ... Value

The two input files don't need to have headers. If it does, a column name must not be a number (int or float)

X, Y and Z should be integers and Value will be converted to a float

-----IMPORTANT:-----
Both files should be ordered by the X,Y and Z values.
If you want to set a file separator equals to a space when running this script, use \s as an argument

At the end, the error metrics are printed with the pattern:

ErrorMetricName: value
""" 

import sys
from timeit import default_timer as timer
import argparse
import pathlib

def getStructuredLineFromLine(line, sep=" "):
    """
    Returns a Dict with X,Y,Z,Value as keys based on the line read from file.
    Also, the split assumes that the values are in order: X Y Z Value

    line: A line from file that has not been split
    sep: The line separator
    """
    lineSplit = line.split(sep)
    structLine = {}
    try:
        structLine['X'] = int(lineSplit[0])
        structLine['Y'] = int(lineSplit[1])
        structLine['Z'] = int(lineSplit[2])
        structLine['Value'] = float(lineSplit[-1])
    except:
        print(f"ERROR: Original Line: {line}, sep: {sep}")
        raise
    
    return structLine

def getStructuredLineFrom(myFile, fileSep):
    """
    Returns a Dict with X,Y,Z,Value as keys based on the line read from file.
    See getStructuredLineFromLine(line)
    myFile: The file that will read a line
    fileSep: The file line separator
    """
    line = myFile.readline()
    return getStructuredLineFromLine(line, fileSep)

def isTheSamePoint(point1, point2):
    """
    Compares the two Points and returns if they are the same.
    point1, point2: Both are Dicts that must have X,Y and Z keys. These keys are used to compare both points
    """
    return (point1["X"] == point2['X'] and point1["Y"] == point2['Y'] and point1["Z"] == point2['Z'])

def isType(type:str, value:str) -> bool:
    """
    Tests if a given value can be cast to a value of the type provided
    type: A valid type to cast value. Should be 'int' or 'float'
    value: The value to try to cast on
    """
    try:
        if type == 'int':
            valueConv = int(value)
        elif type == 'float':
            valueConv = float(value)
    except:
        return False
    else:
        return True

def isHeader(line:str, sep:str) -> bool:
    """
    Tests if a line is considered a header. A line is considered to be a header if it does
    not have a column with a value that can be cast to int or float
    Example: a,b,c is a header but 1,b,c and 'a 5.67 c' are not headers
    line: A line
    sep: The line separator 
    """
    for value in line.split(sep):
        if isType('int', value) or isType('float', value):
            return False
        
    return True

def jumpToNextLineIfStartWithHeader(file, sep:str):
    """
    Jump the header line of file if it is identified as having one
    file: An opened file
    sep: The file line separator
    """
    startPos = file.tell()
    line = file.readline()
    
    if not isHeader(line, sep):
        #this line was not a header. Should go back a line so it doesn't mess with 
        #future line reads
        file.seek(startPos)        

def computeErrors(realValuesFileName, realValuesFileSep, predictedValuesFileName, predictedValuesFileSep):
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

    with open(realValuesFileName, 'r') as realValuesFile, open(predictedValuesFileName, 'r') as predValuesFile:

        jumpToNextLineIfStartWithHeader(realValuesFile, realValuesFileSep)
        jumpToNextLineIfStartWithHeader(predValuesFile, predictedValuesFileSep)

        #Reads a line from realValuesFile
        currRealValueDict = getStructuredLineFrom(realValuesFile, realValuesFileSep)

        #predValuesFile probably has much less lines than realValuesFile
        line = predValuesFile.readline()

        while line not in [None, '']:

            line = line.rstrip('\n').strip()
            currPredValueDict = getStructuredLineFromLine(line, predictedValuesFileSep)
            while not isTheSamePoint(currPredValueDict, currRealValueDict):
                #Reads other line from realValuesFile
                try:
                    currRealValueDict = getStructuredLineFrom(realValuesFile, realValuesFileSep)
                except Exception:
                    print("Got to the end of the real values file!")
                    raise

            #Found a matching point on realValuesFile
            predDiff = currRealValueDict['Value']-currPredValueDict['Value']
            partialRMSESum+=(predDiff)**2
            partialMAESum+= abs(predDiff)
            valuesCount += 1
            
            line = predValuesFile.readline()

    print(f"Pontos contabilizados: {valuesCount}")
    rmse = (partialRMSESum/valuesCount)**(1/2)
    mae = partialMAESum/valuesCount

    return rmse, mae    

def treatInputSepIfSpace(inputSep:str) -> str:
    if inputSep == '\s':
        inputSep = " "
    
    return inputSep

def file_exists(file_path:str) -> bool:
    file = pathlib.Path(file_path)
    return file.exists()

def config_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--real-file', type=str, help="The real values file path", required=True)
    arg_parser.add_argument("--pred-file", type=str, help="The predicted values file path", required=True)

    DEFAULT_PRED_SEP = ","
    DEFAULT_REAL_SEP = " "
    real_file_sep_help = f"The file separator for the real values file. Default: '{DEFAULT_REAL_SEP}'"
    arg_parser.add_argument("--real-file-sep", type=str, default=DEFAULT_REAL_SEP,
                            help=real_file_sep_help)
    pred_file_sep_help = f"The file separator for the predicted values file. Default: '{DEFAULT_PRED_SEP}'"
    arg_parser.add_argument("--pred-file-sep", type=str, default=DEFAULT_PRED_SEP,
                            help=pred_file_sep_help)
    
    return arg_parser

if __name__ == "__main__":

    my_arg_parser = config_arg_parser()
    
    user_args = my_arg_parser.parse_args()
    
    if not file_exists(user_args.real_file):
        print(f"{user_args.real_file} não existe!")
        sys.exit(-1)
    
    if not file_exists(user_args.pred_file):
        print(f"{user_args.pred_file} não existe!")
        sys.exit(-1)
        
    realValuesFileName = user_args.real_file
    realValuesFileSep = user_args.real_file_sep
    predictedValuesFileName = user_args.pred_file
    predictedValuesFileSep = user_args.pred_file_sep

    realValuesFileSep = treatInputSepIfSpace(realValuesFileSep)
    predictedValuesFileSep = treatInputSepIfSpace(predictedValuesFileSep)

    start = timer()
    rmse, mae = computeErrors(realValuesFileName, realValuesFileSep, predictedValuesFileName, predictedValuesFileSep)
    end = timer()
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Elapsed Time: {end-start}")
