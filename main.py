from Modeledonnee import DataLoad

data=DataLoad()

data.missing_values()
data.normalisation()
data.verifNA()
data.shape()
data.info()
data.describe()

data.boxplot()
data.DEBoxplot()
data.showHisto()
data.showCorrelation()

data.regressionlinearDE()
data.regressionlinearFR()

data.RIDGEregression()

data.RIDGEregressionFR()
data.LASSOregression()
data.LASSOregressionFR()
data.ElasticNetregression()
data.ElasticNetregressionFR()
data.DecisionTreeregression()
data.DecisionTreeregressionFR()
data.RandomForestregression()
data.RandomForestregressionFR()
