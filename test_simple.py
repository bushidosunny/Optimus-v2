import streamlit as st

st.write("Hello World!")
st.write("If you see this, Streamlit is working!")

if st.button("Click me"):
    st.balloons()
    st.success("Button clicked!")