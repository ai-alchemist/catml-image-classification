# Image Classification Training and Inference

This project was created as a fully-featured demonstration of training a neural net and then performing image classification with it. It avoids hardcoding things like dataset paths and premature optimizations like the number of objects that the neural net can classify.

The simplest way to run the program is by doing:
```
python3 image_classification_from_scratch.py path_to_dataset
```

Information about other arguments can be found by running

```
python3 image_classification_from_scratch.py -h
```

`inference_from_model.py` can be used to run inference using trained neural nets without having to go through the entire training process every single time.

---
