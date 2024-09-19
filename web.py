import streamlit as st
from agents.context_retrieval import information_lookup
from utils.load_model import load_blindness_detection, get_prediction, load_llm_chain


def ui_image(model, transform, session_state):
    st.write("## Upload your Diabetic Retinopathy Image here")
    with st.form("Blindness Dectetion Form"):
        image = st.file_uploader(
            label="Your image: ",
            label_visibility="collapsed",
            key="image",
        )

        if image:
            _, col2, _ = st.columns(3)
            with col2:
                st.image(image, use_column_width="auto")

        submitted = st.form_submit_button("Predict")

    if submitted and image:
        st.write("## Diagnosis")
        with st.spinner("Please wait..."):
            prediction_name, probs, decision_map = get_prediction(
                model=model, image_path=image, transform=transform
            )
            st.write(
                f'<div style="text-align: right; font-weight: 600; padding: 0px 0px 20px 20px">There are {round(probs[prediction_name], 3) * 100}% that you are {prediction_name}.',
                unsafe_allow_html=True,
            )
            st.bar_chart(probs, x_label="Probability", horizontal=True)

            _, col2, _ = st.columns(3)
            with col2:
                st.image(decision_map, caption="Model Activation Map.")

            session_state["probs"] = probs


def ui_question(chain, session_state):

    st.write("## Ask questions")

    with st.form("Large Language Assistant Form"):
        text = st.text_area(
            "question",
            key="question",
            height=100,
            placeholder="Enter question here",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Submit")

    if text and submitted:
        with st.spinner("Please wait..."):

            context = information_lookup(text)
            diagnosis = session_state.get("probs", "None")
            llm_output = chain.invoke(
                {"user_input": text, "diagnosis": diagnosis, "context": context}
            )

            st.write("### Diabetic Retinopathy Assistant: ")
            st.write(
                f'<div style="background-color: #262730; border-radius: 0.5rem; padding: 10px 15px 10px 15px;">{llm_output["text"]}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":

    # Setup session state
    session_state = st.session_state

    # Load blindness detection and large language model
    model, transform = load_blindness_detection()
    chain = load_llm_chain(temperature=0.0)

    # Load UI
    ui_image(model, transform, session_state)
    ui_question(chain, session_state)
