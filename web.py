import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("bigmart_model.pkl", "rb"))

st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒")

st.title("BigMart Sales Prediction")
st.write("Enter the product details below to predict sales.")

item_identifier = st.number_input("Item Identifier (Encoded)", 0)
item_weight = st.number_input("Item Weight", 0.0)
item_fat_content = st.number_input("Item Fat Content (Encoded)", 0)
item_visibility = st.number_input("Item Visibility", 0.0)
item_type = st.number_input("Item Type (Encoded)", 0)
item_mrp = st.number_input("Item MRP", 0.0)
outlet_identifier = st.number_input("Outlet Identifier (Encoded)", 0)
outlet_establishment_year = st.number_input("Outlet Establishment Year", 1985, 2025)
outlet_size = st.number_input("Outlet Size (Encoded)", 0)
outlet_location_type = st.number_input("Outlet Location Type (Encoded)", 0)
outlet_type = st.number_input("Outlet Type (Encoded)", 0)

if st.button("Predict Sales"):
    input_data = pd.DataFrame([[item_identifier,
        item_weight,
        item_fat_content,
        item_visibility,
        item_type,
        item_mrp,
        outlet_identifier,
        outlet_establishment_year,
        outlet_size,
        outlet_location_type,
        outlet_type]],
        columns=[
        "Item_Identifier",
        "Item_Weight",
        "Item_Fat_Content",
        "Item_Visibility",
        "Item_Type",
        "Item_MRP",
        "Outlet_Identifier",
        "Outlet_Establishment_Year",
        "Outlet_Size",
        "Outlet_Location_Type",
        "Outlet_Type"
    ])

    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: ₹{prediction[0]:.2f}")