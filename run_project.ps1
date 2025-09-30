# Activate virtual environment (create if not exists)
if (-Not (Test-Path "venv")) { py -3.13 -m venv venv }
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install tf-nightly pillow numpy matplotlib scikit-learn seaborn

# Generate synthetic data
python generate_data.py

# Train model + generate graphs
python train.py

# Sample prediction
$sample = "data/validation/invoice/invoice_0.png"
python -c "from predict import predict_image; print('Predicted category for `"$sample`":', predict_image(r'$sample'))"

Write-Host "✅ Project completed. Graphs in reports/, model in models/."
