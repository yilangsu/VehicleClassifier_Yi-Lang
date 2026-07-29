# Vehicle Classifier

A convolutional neural network (CNN) built with PyTorch to classify images into 8 vehicle categories: **Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, Van**.

Designed to run in Google Colab with data and results stored on Google Drive.

## Project Structure

```
VehicleClassifier_Yi Lang/
├── vehicle_classification_Yi_Lang.ipynb   # Main notebook (data prep, training, evaluation)
├── Data/
│   ├── Raw/                               # Input zip (vehicle_classification.zip)
│   └── Processed/                         # Saved train/test split indices
└── Results/
    ├── Model/                             # Saved model checkpoints (best_model.pt, final_model.pt)
    ├── Logs/                              # Per-epoch training log + training curves plot
    └── Summary/                           # Final results summary
```

## Model

A simple CNN (`VehicleCNN`) with 3 convolutional blocks (32 → 64 → 128 filters, each followed by ReLU + MaxPool) feeding into a fully connected classifier with dropout.

- Input: 64x64 RGB images, normalized to [-1, 1]
- Train/test split: 80/20 (seeded for reproducibility)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Batch size: 32

## How to Run

1. Open `vehicle_classification_Yi_Lang.ipynb` in Google Colab.
2. Mount Google Drive and place `vehicle_classification.zip` in `Data/Raw/`.
3. Run the notebook cells in order — data extraction, training, evaluation, and plotting all run automatically and save outputs to `Results/`.

## Results

Trained for 10 epochs on 26,378 images (21,102 train / 5,276 test).

| Metric | Value |
|---|---|
| Final Training Accuracy | 91.89% |
| Final Testing Accuracy | 81.48% |
| Best Testing Accuracy | 82.73% (epoch 7) |

See [`Results/Logs/training_curves.png`](VehicleClassifier_Yi%20Lang/Results/Logs/training_curves.png) for loss/accuracy curves and [`Results/Summary/final_summary.txt`](VehicleClassifier_Yi%20Lang/Results/Summary/final_summary.txt) for the full run summary.
