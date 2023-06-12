import os
import subprocess
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors.blip_processors import BlipCaptionProcessor
import platform

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if device is not arm / Apple Silicon, install LAVIS from pip
# else install from source

if platform.processor() == "arm":
    installation_instructions = ["pip install salesforce-lavis"]
else:
    installation_instructions = [
        "cd ~/.cache/autodistill/ && git clone https://github.com/salesforce/LAVIS",
        "cd ~/.cache/autodistill/LAVIS && pip install -r requirements.txt",
        "cd ~/.cache/autodistill/LAVIS && python setup.py build develop --user",
    ]


@dataclass
class Model(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology

        if not os.path.exists(f"{HOME}/.cache/autodistill/LAVIS"):
            for command in installation_instructions:
                subprocess.run(command, shell=True)

        model, vis_processors, _ = load_model_and_preprocess(
            "blip_feature_extractor", model_type="base", is_eval=True, device=DEVICE
        )

        self.model = model
        self.vis_processors = vis_processors

    def predict(self, input: str) -> sv.Detections:
        image = Image.open(input).convert("RGB")

        image = self.vis_processors["eval"](image).unsqueeze(0).to(DEVICE)

        classes = self.ontology.classes()

        if len(classes) == 1:
            classes.append("something else")

        text_processor = BlipCaptionProcessor(prompt="A picture of ")

        cls_prompt = [text_processor(cls_nm) for cls_nm in classes]

        sample = {"image": image, "text_input": cls_prompt}

        image_features = self.model.extract_features(
            sample, mode="image"
        ).image_embeds_proj[:, 0]
        text_features = self.model.extract_features(
            sample, mode="text"
        ).text_embeds_proj[:, 0]

        sims = (image_features @ text_features.t())[0] / self.model.temp
        probs = torch.nn.Softmax(dim=0)(sims).tolist()

        top_k = 1

        top_k_idx = torch.topk(sims, top_k).indices.tolist()

        return sv.Classifications(
            class_id=np.array([top_k_idx[0]]),
            confidence=np.array([probs[top_k_idx[0]]]),
        )
