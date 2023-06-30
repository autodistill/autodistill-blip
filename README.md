<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png?4"
      >
    </a>
  </p>
</div>

# Autodistill BLIP Module

This repository contains the code supporting the BLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[BLIP](https://github.com/salesforce/LAVIS), developed by Salesforce, is a computer vision model that supports visual question answering and zero-shot classification. Autodistill supports classifying images using BLIP.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [BLIP Autodistill documentation](https://autodistill.github.io/autodistill/base_models/blip/).

## Installation

To use BLIP with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-blip
```

## Quickstart

```python
from autodistill_blip import BLIP

# define an ontology to map class names to our BLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = BLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```


## License

This project is licensed under a [3-Clause BSD license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!