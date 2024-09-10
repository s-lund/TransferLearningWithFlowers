# Flower Classification: Transfer Learning vs. Custom CNN

This project explores two approaches to flower classification using deep learning techniques, focusing on scenarios with limited training data.

## Dataset

We use the [Flowers "Five Classes" dataset](https://www.kaggle.com/datasets/lara311/flowers-five-classes/data) from Kaggle, which contains images of five different flower types.

## Approaches

### 1. Transfer Learning with EfficientNet-B3

- Utilize a pre-trained EfficientNet-B3 model for feature extraction
- Fine-tune the model on our flower dataset
- Validate the results using train/test split

### 2. Custom CNN based on AlexNet

- Implement a convolutional neural network inspired by the AlexNet architecture
- Train the model from scratch on a dataset more closely related to flower images
- Validate the results using train/test split

## Project Structure

```
├── data/
│   └── flowers-five-classes/
├── notebooks/
│   ├── 1_transfer_learning_efficientnet.ipynb
│   ├── 2_custom_cnn_alexnet.ipynb
│   └── 3_comparison.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── plotting.py
│   └── train_test.py
├── results/
│   ├── transfer_learning_results.csv
│   └── custom_cnn_results.csv
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/s-lund/transfer-learning-with-flowers.git
   cd transfer-learning-with-flowers
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `data/` directory.

## Usage

1. Run the transfer learning notebook:
   ```
   jupyter notebook notebooks/1_transfer_learning_efficientnet.ipynb
   ```

2. Run the custom CNN notebook:
   ```
   jupyter notebook notebooks/2_custom_cnn_alexnet.ipynb
   ```

## Results

We compare the performance of both approaches:

1. Transfer Learning with EfficientNet-B3
2. Custom CNN based on AlexNet

Detailed results and analysis can be found in the respective notebook files.

## Conclusion

[Add your conclusions here after completing the project]

## Future Work

- Experiment with other pre-trained models (e.g., ResNet, VGG)
- Implement data augmentation techniques to improve model generalization
- Explore ensemble methods to combine the strengths of both approaches

## Contributing

Feel free to open issues or submit pull requests with improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
