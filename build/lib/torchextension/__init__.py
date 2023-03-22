# === Glory Be To God ====

"""
    torchextension.

    from torch import nn 
    from torchextension.metrics import MAE
    from torchextension.data_generator import DataGenerator

    # Create demo data....
    train_data = DataGenerator(100, 8)
    test_data = DataGenerator(100, 8, train_sample=False)

    model = Sequential([
    # Input layer
        nn.Linear(in_features=8, out_features=32),
        nn.ReLU(),  # Activation function
        nn.Dropout(.4),
        
        # First hidden layer
        nn.Linear(in_features=32, out_features=32),
        nn.ReLU(),  # Activation function
        nn.Dropout(.4),  # Drop same pixel
        
        # Output layer
        nn.Linear(in_features=32, out_features=1)
        ])
        # Compile the model .........
        model.compile(
        optimize=torch.optim.Adam(model.parameters()),
        loss=nn.MSELoss(),
        metrics=MAE(),
        device=None
        )
        
    train_dataloader = DataLoader(dataset=train_data, shuffle=True)
    valid_dataloader = DataLoader(dataset=test_data, shuffle=False)

    # Fit the data .........
    history = model.fit(
        train_dataloader,
        epochs=10,
        verbose=False,
        validation_data=valid_dataloader
        )
    print(model.predict(test_load))
"""

__version__ = "0.0.1"
__author__ = "Mutesasira Denis"
