import sys
import pandas as pd

if __name__ == "__main__":
    predictedValuesFileName = sys.argv[1]
    predValues = pd.read_csv(predictedValuesFileName, header=0)
    predValues.sort_values(by=['x','y','z'], inplace=True)
    print("Ordenou")
    sortedFileName = predictedValuesFileName.split(".")[0]+"_sorted.csv"
    print(f"Salvando em: {sortedFileName}.")
    predValues.to_csv(sortedFileName, index=False)