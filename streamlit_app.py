
import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('tinter_model.joblib')
scaler = joblib.load('tinter_scaler.joblib')
feature_cols = ['L_meas', 'a_meas', 'b_meas', 'Standard_L', 'Standard_A', 'Standard_B', 'dL', 'da', 'db']
target_cols = ['0T_SPL_BLK', '0T_VIOLET', '0T_WHITE', '0T_WHITE_BASE', 'APP_VIOLET_23', 'APP_WHITE', 'AS_SUCH', 'B_BLUE', 'BEETA_BLUE', 'BETA_BLUE', 'BLACK', 'BLUE', 'BLUE_211', 'BLUE_385', 'BLUE211', 'CARBAN_BLACK', 'CARBON_BLACK', 'DPP_RED', 'GREEN', 'GREEN_7', 'GREEN7', 'M._CHROME', 'MIDDLE_CHROME', 'Middle_Chrome', 'OT_BLACK', 'RED_3097', 'RED_OXIDE_3097', 'RED3097', 'RED8097', 'Red_Oxide', 'S._CHROME', 'SCARLET_CHROME', 'SDP_SPL_BLACK', 'SDP_YELLOW_10C242', 'SDP_YELLOW_OXIDE', 'SPL_BLACK', 'SPL_BLK', 'SPL._BLACK', 'SR_TINTER_BLK', 'VIOLET', 'VIOLET_23', 'VIOLET-23', 'WHITE', 'WHITE_BASE', 'YELLOW', 'YELLOW_10C242', 'YELLOW_OXD', 'YELLOW_OXIDE', 'YELOLOW_OXIDE', 'ac_GREEN_7', 'ac_white', 'ac_yellow_oxide', 'b_blue', 'beta_blue', 'black', 'black_ot', 'blue', 'blue_385', 'blue211', 'carbon_black', 'green', 'green_7', 'green7', 'lemon_chrome', 'middle_chrome', 'ot_white', 'ot_yellow', 'red_3097', 'red3097', 'sdp_black', 'sdp_green_7', 'sdp_spl_black', 'sdp_white', 'sdp_yellow_oxd', 'sdp_yellow_oxide', 'sp_black', 'spl_Black', 'spl_black', 'spl_blk', 'spl._black', 'violet', 'violet_23', 'violet23', 'white', 'white_base', 'yellow', 'yellow_oxd', 'yellow_oxide', 'yellow_oxide_ot']

st.title("Tinter Amount Predictor")

L = st.number_input("Measured L")
a = st.number_input("Measured a")
b = st.number_input("Measured b")

Std_L = st.number_input("Standard L")
Std_a = st.number_input("Standard a")
Std_b = st.number_input("Standard b")

if st.button("Predict"):
    dL = Std_L - L
    da = Std_a - a
    db = Std_b - b
    X = np.array([[L,a,b,Std_L,Std_a,Std_B,dL,da,db]])
    Xs = scaler.transform(X)
    ypred = model.predict(Xs)[0]
    result = pd.DataFrame({"Tinter": target_cols, "Predicted %": ypred})
    st.dataframe(result)
