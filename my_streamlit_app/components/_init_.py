import streamlit as st
import pandas as pd

# Define the path to your sample data
SAMPLE_DATA_PATH = "data/sample_data.csv"

def upload_sample_data():
    """
    Function to upload sample data from the file path.
    """
    return pd.read_csv(SAMPLE_DATA_PATH)

def main():
    st.title("Customer Satisfaction Dashboard")

    # Button to upload the sample data
    if st.button("Upload Sample Data"):
        st.write("Uploading sample data...")
        df = upload_sample_data()
        st.write(df)  # Display the sample data
    else:
        # Option to upload a file if the button is not clicked
        uploaded_file = st.file_uploader("Upload Engagement and Experience Scores CSV Files", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df)  # Display the uploaded data
    
if __name__ == "__main__":
    main()
    
