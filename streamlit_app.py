import streamlit as st
import pickle
import pandas as pd 
import xgboost as xgb
from xgboost import DMatrix
model = pickle.load(open("./xgb_model.pkl", "rb"))


def show_predict_page():
    
    st.write("""# Patient Survival Prediction App""")
    st.write("""### Enter information for prediction """)        
    
    gcs_verbal_apache = st.selectbox('GCS Verbal Apache', options=[1.0,2.0,3.0,4.0,5.0] )
    gcs_motor_apache = st.selectbox('GCS Motor Apache', options=[1.0,2.0,3.0,4.0,5.0,6.0] )
    ventilated_apache = st.selectbox('Ventilated Apache ', options=['yes', 'no'] )
    if ventilated_apache == "yes":
        ventilated_apache = 1.0
    else:
        ventilated_apache = 0.0

    gcs_eyes_apache = st.selectbox('GCS Eyes Apache', options=[1.0,2.0,3.0,4.0] )


    apache_4a_hospital_death_prob = st.slider( 'Apache Hospital Death Probability', 0.01, 0.99, 0.01 )
    apache_4a_icu_death_prob = st.slider( 'Apache ICU Death Probability', 0.01, 0.99, 0.01)
    d1_sysbp_noninvasive_min = st.slider( 'SysBP NonInvassive Min', 41.0, 160.0, 1.0 )
    d1_sysbp_min = st.slider( 'SysBP Min ', 41.0, 160.0, 1.0 )
    
    ok = st.button("Predict")
    
    columns_name = ['gcs_verbal_apache', 'gcs_motor_apache', 'ventilated_apache', 'gcs_eyes_apache', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'd1_sysbp_noninvasive_min', 'd1_sysbp_min']

    values = {'gcs_verbal_apache':[gcs_verbal_apache], 'gcs_motor_apache':[gcs_motor_apache], 'ventilated_apache':[ventilated_apache], 
              'gcs_eyes_apache':[gcs_eyes_apache], 'apache_4a_hospital_death_prob':[apache_4a_hospital_death_prob], 'apache_4a_icu_death_prob':[apache_4a_icu_death_prob],
                'd1_sysbp_noninvasive_min':[d1_sysbp_noninvasive_min], 'd1_sysbp_min':[d1_sysbp_min]}

    df = pd.DataFrame(values)
    df = df[["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob", "gcs_motor_apache", "gcs_verbal_apache", "ventilated_apache", "gcs_eyes_apache", "d1_sysbp_min", "d1_sysbp_noninvasive_min"]]
    DM_df = DMatrix(df)
    
    if ok:
        predictions = model.predict(DM_df)
        pred = predictions[0]

        if pred<0.5:
            st.write("#### Prediction : 1 | Patient Death" )
        else:
            st.write("#### Prediction : 0 | Patient Survived" )

show_predict_page()
