
import joblib
import pandas as pd
import streamlit as st

FEATURES = ["pclass", "sex", "age", "fare"]

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.title("Would this passenger have survived?")

st.write("An educational model trained on the Titanic passenger dataset.")

pclass = st.selectbox("Ticket Class", [1, 2, 3])

sex = st.radio("Sex", ["female", "male"])

age = st.slider("Age", 1, 80, 30)

fare = st.slider("Fare Paid", 0, 500, 30)

if st.button("Predict"):
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("The model file is missing. Please add model.joblib.")
        st.stop()
    except Exception as e:
        st.error(f"Unable to load the model: {e}")
        st.stop()

    row = pd.DataFrame(
        [[pclass, 1 if sex == "female" else 0, age, fare]],
        columns=FEATURES
    )

    chance = model.predict_proba(row)[0][1]

    st.metric("Estimated Chance of Survival", f"{chance:.0%}")

    st.caption("A learning exercise based on historical Titanic data. This is not advice.")
