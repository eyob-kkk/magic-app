
import streamlit as st, pandas as pd, joblib, time, os

model = joblib.load("magic_model.joblib")

st.title("MAGIC Telescope: Gamma or Hadron?")
st.write("Enter the measurements of one detected event.")

row = {
    "fLength": st.number_input("fLength", 0.0, 400.0, 37.0),
    "fWidth": st.number_input("fWidth", 0.0, 300.0, 17.0),
    "fSize": st.number_input("fSize", 1.0, 6.0, 2.74),
    "fConc": st.number_input("fConc", 0.0, 1.0, 0.35),
    "fConc1": st.number_input("fConc1", 0.0, 1.0, 0.20),
    "fAsym": st.number_input("fAsym", -500.0, 600.0, 4.0),
    "fM3Long": st.number_input("fM3Long", -400.0, 300.0, 15.0),
    "fM3Trans": st.number_input("fM3Trans", -250.0, 200.0, 0.0),
    "fAlpha": st.number_input("fAlpha", 0.0, 90.0, 18.0),
    "fDist": st.number_input("fDist", 0.0, 520.0, 191.0),
}

if st.button("Predict"):
    start = time.perf_counter()
    data = pd.DataFrame([row])
    data["length_width_ratio"] = data["fLength"] / (data["fWidth"] + 0.000001)
    prob = model.predict_proba(data)[0, 1]
    ms = (time.perf_counter() - start) * 1000
    st.metric("Probability of gamma", f"{prob:.1%}")
    st.write("Prediction:", "Gamma (signal)" if prob >= 0.5 else "Hadron (background)")
    st.caption(f"Response time: {ms:.1f} ms")
    log = pd.DataFrame([{**row, "prob_gamma": prob, "latency_ms": ms}])
    log.to_csv("monitor_log.csv", mode="a", header=not os.path.exists("monitor_log.csv"), index=False)