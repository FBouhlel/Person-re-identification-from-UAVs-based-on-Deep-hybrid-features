Person re-identification from UAVs based on Deep hybrid features: Application for intelligent video surveillance
Journal : Signal, Image and Video Processing
Authors: Fatma Bouhlel, Hazar Mliki and Mohamed Hammami

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- torchvision
- scipy
- numpy
- scikit_learn


## Data

The data structure would look like:
```
data/
    bounding_box_train/
    bounding_box_test/
    query/
```

## Train

You can specify more parameters in opt.py

```
python main.py --mode train --data_path <path/to/PRAI-1581> 
```

## Evaluate

Use pretrained weight or your trained weight

```
python main.py --mode evaluate --data_path <path/to/PRAI-1581> --weight <path/to/weight_name.pt> 
```

=
