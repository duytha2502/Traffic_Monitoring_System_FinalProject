import streamlit as st

st.markdown(
    """
    <style>
    .css-d1b1ld.edgvbvh6 
    {
        visibility: hidden;
    }
    .css-1v8iw7l.eknhn3m4
    {
        visibility: hidden;
    }
    .metric_label 
    {
        text-align: center;
    }
    .metric_number
    {
        text-align: center;
        color: red;
    }

    .ea3mdgi5 {
        width: 1300px;
        margin-top: -50px 
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Welcome to Traffic analysis and monitoring system! 👋")
st.caption("Follow these steps to start a tracking system")

st.subheader("Step 1: Initialize setting of the process")
st.image("img/about1.png")
st.markdown("---")

st.subheader("Step 2: Checking upload video, setting value and start tracking")
st.image("img/about2.png")
st.markdown("---")

st.subheader("Step 3: Wait and here you are! Processed video and metrics :sparkles:")
col1, col2 = st.columns(2)
with col1:
    st.image("img/about3.png")
with col2:
    st.image("img/about4.png")
st.markdown("---")

st.subheader("Step 4: See more detail of the processed video")
st.image("img/about5.png")
