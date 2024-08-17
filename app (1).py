import streamlit as st
import pandas as pd
import pickle as pk

st.set_page_config(page_title='Laptop price predict',layout='wide',page_icon='💻')
model=pk.load(open('lap.pkl','rb'))
lap=pd.read_csv('laptop.csv')
html_temp = """
		<div style="background-color:#F7C66A;padding:08px;border-radius:08px">
		<h1 style="color:white;text-align:center;">💻 Laptop Price Predictor..</h1>
		</div>
		"""
st.markdown(html_temp,unsafe_allow_html=True)
c1,c2=st.columns(2)
with c1:
    Company=st.selectbox('Brand',lap['Company'].unique())
    TypeName=st.selectbox('Type',lap['TypeName'].unique())
    Ram=st.selectbox('RAM(in GB)',lap['Ram'].unique())
    OpSys=st.selectbox('OS',lap['OpSys'].unique())
    Weight=st.selectbox('Weight of the laptop ',lap['Weight'].unique())
with c2:    
    Touchscreen=st.selectbox('Touchscreen(yes 1/no 0)',lap['Touchscreen'].unique())
    HDD=st.selectbox('HDD(in GB)',lap['HDD'].unique())
    SSD=st.selectbox('SSD(in GB)',lap['SSD'].unique())
    GPU=st.selectbox('GPU',lap['GPU'].unique())
    input=pd.DataFrame([[Company,TypeName,Ram,OpSys,Weight,Touchscreen,HDD,SSD,GPU]],
                    columns=['Company','TypeName','Ram','OpSys','Weight','Touchscreen','HDD','SSD','GPU']
                    )
    with st.expander('View Selected Configuration'):
        st.dataframe(input)
    res=model.predict(input)
    if st.button('predict....'):
        st.header('The predicted price of this configuration')
        st.success(res)

st.markdown("________________________")

f1,f2=st.columns(2)
with f1:
    with st.container():
        st.header('🚀 The project involves')
        st.markdown(""" 
                    - data exploration,  
                    - visualization, 
                    - feature engineering, and  
                    - The implementation of a linear regression model to predict laptop prices based on various features.
                    """)
        st.subheader('🧐Data Exploration')
        st.markdown("""
                    The dataset consists 1303 rows and 15 columns which include
                    - Company	-Name of the laptop
                    - TypeName	-Brand name
                    - Ram	    -Amount of RAM (in GB)
                    - OpSys	    -Name Of operation system
                    - Weight	-Weight of laptop 
                    - Touchscreen -TouchScreen availability(yes or no)	
                    - CpuCompany   -CPU brad name 
                    - ClockSpeed   -Clockspead of Laptop 	
                    - Flash Storage	-About Flash Storage
                    - HDD	        -Hard Disk size Drive(in GB)
                    - Hybrid	    -Aboud Hyprid
                    - SSD	        -Solid State Drive size(in GB)
                    - GPU	        -GPU name
                    - Ppi	        -Pixels per inch
                    - Price         -Price of the laptop
                    """)
    with f2:
        with st.container():
            st.subheader('🧹 feature engineering')
            st.markdown("""
                        - OneHotEncoder for Categorical values [Company,TypeName,OpSys,GPU]
                        """)
            st.subheader('💡Data Modeling')
            st.markdown("""
                        - A linear regression model was used to predict the final price of the laptops.
                        - The features used for the model included Company,TypeName,Ram,OpSys,Weight,Touchscreen,HDD,SSD,GPU
                        """)
            st.subheader('⚖️Model Training and Testing')
            st.markdown("""
                        The data was split into training and testing sets (75% training and 25% testing).
                        The model was trained on the training data and tested on the testing data.

                        """)
            st.subheader('🌟Model Evaluation')
            st.markdown("""
                        - The model's performance was evaluated using the following metrics:
                        - R-squared (R2 Score): 0.729
                        - Mean Absolute Error (MAE): 14588.03
                        """)
            