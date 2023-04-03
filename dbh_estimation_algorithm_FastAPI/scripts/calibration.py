import statsmodels.formula.api as smf
import pandas as pd

calibiration_data = pd.read_csv("data/static/dbh_algorithm_calibration_data.csv")

model = smf.ols('measured_dbh ~ ratio', data=calibiration_data)
model = model.fit()

# prediction function
def getPrediction(ratio):
    try:
        new_X = float(ratio)
        pred = model.predict({"ratio": new_X})
        return round(pred.tolist()[0],2)
    except:
        return None