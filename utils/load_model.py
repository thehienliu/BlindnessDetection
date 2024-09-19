import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms
from private_key import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from utils.transforms import CricleCrop, Normalize
from tools.search import get_google_search_information
from models.blindness_detection import BlindnessDetection
from tools.retrieve import get_diabetic_retinopathy_context
from langchain.agents import initialize_agent, AgentType


def load_llm_chain(temperature: float = 0.0):

    # Setup Large Language Model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY, temperature=temperature
    )

    # Setup prompt template
    prompt_template = PromptTemplate(
        template="""
        You are an AI assistant specialized in diabetic retinopathy diagnosis and treatment.
        Use the following pieces of retrieved context and user's diagnosis to response the user's input.
        Your response should be polite, empathetic, and informative.
        If you don't know the answer, just say that you don't know.
        User's Input: {user_input}
        Diagnosis: {diagnosis}
        Context: {context}""",
        input_variables=["user_input", "diagnosis", "context"],
    )

    # Initial agent
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain


def load_blindness_detection():
    config = OmegaConf.load("web_materials/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlindnessDetection(
        module_name=config.model.extractor_module_name,
        hidden_size=config.model.hidden_size,
        output_size=config.model.output_size,
    )

    model.load_state_dict(
        torch.load("web_materials/model.pt", map_location=device, weights_only=True)
    )
    model.to(device)

    transform = transforms.Compose(
        [
            CricleCrop(device=device),
            transforms.Resize(
                (config.transform.input_size, config.transform.input_size)
            ),
            Normalize(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def get_decision_map(transform_image, model):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    model.eval()
    with torch.no_grad():
        output1 = model.extractor(transform_image)
        output2 = model.avgpool(output1)
        output3 = model.classifier[0](output2)
        output4 = model.classifier[1](output3)
        output5 = model.classifier[2](output4)
        output6 = model.classifier[3](output5)

    # Get the most contribution weight to prediction
    prediction = output6.argmax(dim=-1).item()
    contribution = model.classifier[3].weight.data[prediction]

    # Get the most contribution previous weight to prediction
    contribution_idx2 = (contribution * output5).argmax(dim=-1).item()
    feature_map_weight = model.classifier[1].weight.data[contribution_idx2]

    decision_map = feature_map_weight[None, :, None, None] * output1

    img = inv_normalize(transform_image[0])
    map = torch.nn.functional.interpolate(decision_map, (224, 224), mode="bicubic").sum(
        1
    )

    # Normalize the map to range [0, 1]
    map_min, map_max = map.min(), map.max()
    normalized_map = (map - map_min) / (map_max - map_min)

    # Convert normalized map to a numpy array
    normalized_map = normalized_map.squeeze().numpy()

    # Apply a colormap
    colored_map = plt.cm.jet(
        normalized_map
    )  # Colormap is applied to the normalized map
    colored_map = np.delete(colored_map, 3, 2)  # Remove alpha channel

    # Optionally overlay the colormap on the original image
    final_map = np.transpose(colored_map, (2, 0, 1)) * 0.25 + img.numpy() * 0.75

    return np.transpose(final_map, (1, 2, 0)), prediction


def get_prediction(model, image_path, transform):

    label_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    img = Image.open(image_path)
    x = transform(img).unsqueeze(0)
    decision_map, _ = get_decision_map(x, model)

    model.eval()
    with torch.no_grad():
        output = model(x)

    prediction = output.argmax(dim=-1).item()
    prediction_name = label_names[prediction]
    probs = {
        label_names[i]: p for i, p in enumerate(F.softmax(output, dim=-1)[0].tolist())
    }

    return prediction_name, probs, decision_map
