import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
import torch

from autodistill.classification import ClassificationBaseModel
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if device is not arm / Apple Silicon, install LAVIS from pip
# else install from source


@dataclass
class BLIP(ClassificationBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology

        if platform.processor() != "arm":
            subprocess.run("pip install salesforce-lavis", shell=True)
        elif not os.path.exists(f"{HOME}/.cache/autodistill/LAVIS"):
            installation_instructions = [
                "cd ~/.cache/autodistill/ && git clone https://github.com/salesforce/LAVIS",
                "cd ~/.cache/autodistill/LAVIS && pip install -r requirements.txt",
                "cd ~/.cache/autodistill/LAVIS && python setup.py build develop --user",
            ]
            for command in installation_instructions:
                subprocess.run(command, shell=True)

        if platform.processor() == "arm":
            sys.path.append(f"{HOME}/.cache/autodistill/LAVIS")

        from lavis.models import load_model_and_preprocess
        from lavis.processors.blip_processors import BlipCaptionProcessor

        model, vis_processors, _ = load_model_and_preprocess(
            "blip_feature_extractor", model_type="base", is_eval=True, device=DEVICE
        )

        self.model = model
        self.vis_processors = vis_processors
        self.tokenizer = BlipCaptionProcessor

    def predict(self, input: Any) -> sv.Classifications:
        image = load_image(input, return_format="PIL").convert("RGB")

        image = self.vis_processors["eval"](image).unsqueeze(0).to(DEVICE)

        classes = self.ontology.classes()

        text_processor = self.tokenizer(prompt="A picture of ")

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

        class_ids = np.arange(len(classes))

        return sv.Classifications(
            class_id=np.array(class_ids),
            confidence=np.array(probs),
        )
