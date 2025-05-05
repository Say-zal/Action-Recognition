# Action Recognition using Vision Transformer (ViT)

This project implements action recognition on the HMDB dataset using the Vision Transformer (ViT).

## Features
- Train and evaluate a ViT model for video classification.
- Web interface for uploading videos and predicting actions.

## Project Structure

```
action-recognition-vit
├── src
│   ├── models
│   │   └── vit.py          # Implementation of the Vision Transformer model
│   ├── training
│   │   ├── train.py        # Training script for the ViT model
│   │   └── dataset.py      # Dataset class for loading and preprocessing video data
│   ├── evaluation
│   │   └── evaluate.py     # Evaluation script for assessing model performance
│   ├── web
│   │   ├── app.py          # Web application for user interaction
│   │   ├── templates
│   │   │   └── index.html  # HTML template for the web interface
│   │   └── static
│   │       └── styles.css  # CSS styles for the web interface
│   └── utils
│       └── helpers.py      # Utility functions for data processing and visualization
├── requirements.txt         # List of project dependencies
├── README.md                # Project documentation
└── .gitignore               # Files and directories to ignore in Git
```

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
```

## Usage

1. **Training the Model**: 
   To train the Vision Transformer model, run the following command:

   ```bash
   python src/training/train.py
   ```

2. **Evaluating the Model**: 
   After training, you can evaluate the model's performance using:

   ```bash
   python src/evaluation/evaluate.py
   ```

3. **Running the Web Interface**: 
   To start the web application, execute:

   ```bash
   python src/web/app.py
   ```

   Then, navigate to `http://127.0.0.1:5000` in your web browser to upload videos and view classification results.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.