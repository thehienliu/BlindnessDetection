import streamlit as st

# from langchain_openai import ChatOpenAI


def on_api_key_change():
    api_key = ss.get("api_key")
    print(api_key)


def on_image_key_change():
    image = ss.get("image")
    print(image)


def ui_api_key():
    st.write("## Enter your OpenAI API key")
    st.text_input(
        "OpenAI API key",
        type="password",
        key="api_key",
        label_visibility="collapsed",
        on_change=on_api_key_change,
    )


def ui_image():
    st.write("## Upload your Diabetic Retinopathy Image here")
    image = st.file_uploader(
        label="Your image: ",
        label_visibility="collapsed",
        on_change=on_image_key_change,
        key="image",
    )

    if image:
        _, col2, _ = st.columns(3)
        with col2:
            st.image(image, use_column_width="auto")


def ui_question():
    st.write("## Ask questions")
    st.text_area(
        "question",
        key="question",
        height=100,
        placeholder="Enter question here",
        label_visibility="collapsed",
        disabled=False,
    )


if __name__ == "__main__":

    ss = st.session_state
    ui_api_key()
    ui_image()
    ui_question()
