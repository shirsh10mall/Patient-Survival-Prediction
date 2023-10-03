import streamlit as st
import pickle
import pandas as pd 
import xgboost as xgb
from xgboost import DMatrix
model = pickle.load(open("./xgb_model.pkl", "rb"))


def show_predict_page():
    
    st.write("""# Patient Survival Prediction App""")
    st.write("""### Enter information for prediction """)        
    
    gcs_verbal_apache = st.selectbox('Product Group ', options=[1.0,2.0,3.0,4.0,5.0] )
    gcs_motor_apache = st.selectbox('Product Content ', options=[1.0,2.0,3.0,4.0,5.0,6.0] )
    ventilated_apache = st.selectbox('Unit ', options=['yes', 'no'] )
    if ventilated_apache == "yes":
        ventilated_apache = 1.0
    else:
        ventilated_apache = 0.0

    gcs_eyes_apache = st.selectbox('Dosage Form ', options=[1.0,2.0,3.0,4.0] )


    apache_4a_hospital_death_prob = st.slider( 'Apache Hospital Death Probaboloty', 0.01, 0.99, 0.01 )
    apache_4a_icu_death_prob = st.slider( 'Lowest Competitor Price ', 0.01, 0.99, 0.01)
    d1_sysbp_noninvasive_min = st.slider( 'Product Price', 41.0, 160.0, 1.0 )
    d1_sysbp_min = st.slider( 'Manufacrurer ID ', 41.0, 160.0, 1.0 )
    
    ok = st.button("Predict")
    
    columns_name = ['gcs_verbal_apache', 'gcs_motor_apache', 'ventilated_apache', 'gcs_eyes_apache', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'd1_sysbp_noninvasive_min', 'd1_sysbp_min']

    values = {'gcs_verbal_apache':[gcs_verbal_apache], 'gcs_motor_apache':[gcs_motor_apache], 'ventilated_apache':[ventilated_apache], 
              'gcs_eyes_apache':[gcs_eyes_apache], 'apache_4a_hospital_death_prob':[apache_4a_hospital_death_prob], 'apache_4a_icu_death_prob':[apache_4a_icu_death_prob],
                'd1_sysbp_noninvasive_min':[d1_sysbp_noninvasive_min], 'd1_sysbp_min':[d1_sysbp_min]}

    df = pd.DataFrame(values)
    df = df[["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob", "gcs_motor_apache", "gcs_verbal_apache", "gcs_eyes_apache", "ventilated_apache", "d1_sysbp_min", "d1_sysbp_noninvasive_min"]]
    DM_df = DMatrix(df)

    if ok:
        predictions = model.predict(DM_df)
        predictions = predictions[0]

        st.write("#### Prediction : "+str( predictions[0] ) )

show_predict_page()
