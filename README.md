# 1. Create venv
C:/Users/shlag/AppData/Local/Programs/Python/Python313/python.exe -m venv venv

# 2. Activate venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install tensorflow matplotlib pillow scikit-learn seaborn

# 4. Generate data
python generate_data.py

# 5. Train model + generate graphs
python train.py

# 6. Predict sample image
python -c "from predict import predict_image; print('Predicted category for', predict_image('data/validation/invoice/invoice_0.png'))"