# Sketch to photo faces using CycleGAN

CycleGAN provides a framework to train image-to-image translation with unpaired
datasets using cycle consistency loss. While results are great in many applications.
And Face photo-sketch synthesis is one of the challenging applications that CycleGAN
can give better results and this help a lot in face recognition systems. In this work, we
implemented a cycleGAN to translate from Real face images to sketch and vice versa.

Check the cycleGAN paper [arxiv](https://arxiv.org/abs/1703.10593).\
Check the used dataset [CUHK](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install -U git+https://github.com/albumentations-team/albumentations
pip install torch torchvision
```

## Usage

Before you run the [main.py](https://github.com/AymenSe/The-generation-of-colored-faces-from-sketches-using-CycleGAN/blob/main/main.py) file make sure to update the [config.yaml](https://github.com/AymenSe/The-generation-of-colored-faces-from-sketches-using-CycleGAN/blob/main/config/config.yaml) file.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[GPL v3.0](https://choosealicense.com/licenses/gpl-3.0/)