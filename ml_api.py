from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class model_input(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# loading the saved model
diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: model_input):
    input_data = input_parameters.dict()

    preg = input_data['Pregnancies']
    glu = input_data['Glucose']
    bp = input_data['BloodPressure']
    st = input_data['SkinThickness']
    ins = input_data['Insulin']
    bmi = input_data['BMI']
    dpf = input_data['DiabetesPedigreeFunction']
    age = input_data['Age']

    input_list = [preg, glu, bp, st, ins, bmi, dpf, age]

    prediction = diabetes_model.predict([input_list])

    if prediction[0] == 0:
        return {"prediction": "The person is non-diabetic"}
    else:
        return {"prediction": "The person is diabetic"}
