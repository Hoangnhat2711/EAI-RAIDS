import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np # For dummy data generation
from data.data_transforms import get_image_transforms # Import image transforms

# Placeholder for data loading functions

def load_adult_census_data(test_size=0.2, random_state=42):
    """
    Loads and preprocesses the Adult Census dataset.

    This function handles loading the raw Adult Census data (adult.data and adult.test),
    concatenates them, cleans the target variable, identifies sensitive attributes,
    and applies preprocessing (numerical scaling, categorical one-hot encoding).
    It then splits the data into training and testing sets, returning features (X),
    target (y), and sensitive attributes (s) as pandas DataFrames.

    Args:
        test_size (float, optional): The proportion of the dataset to include in the test split.
                                     Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training target labels.
            - y_test (pd.Series): Testing target labels.
            - s_train (pd.DataFrame): Training sensitive attributes (one-hot encoded).
            - s_test (pd.DataFrame): Testing sensitive attributes (one-hot encoded).
        Returns (None, None, None, None, None) if data files are not found.
    """
    print("Loading and preprocessing Adult Census data...")

    # Define column names based on UCI Adult Census documentation
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    # Load training data
    try:
        df_train = pd.read_csv('data/raw/adult.data', header=None, names=column_names, skipinitialspace=True)
        # Load test data (UCI provides a separate test set, first row is sometimes metadata, adjust as needed)
        df_test = pd.read_csv('data/raw/adult.test', header=None, names=column_names, skipinitialspace=True, skiprows=1)
    except FileNotFoundError:
        print("Error: Adult Census files (adult.data, adult.test) not found in data/raw/")
        print("Please download them from UCI Machine Learning Repository and place them there.")
        return None, None, None, None, None, None

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Clean target variable
    df['income'] = df['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    df['income'] = (df['income'] == '>50K').astype(int) # Convert to 0/1

    # Define sensitive attributes
    sensitive_features = ['sex', 'race']

    # Define categorical and numerical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [f for f in categorical_features if f not in ['income'] + sensitive_features]

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features = [f for f in numerical_features if f not in ['income']]
    
    # Create preprocessing pipelines for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], 
        remainder='passthrough' # Keep other columns (like sensitive attributes)
    )

    # Separate features (X), target (y), and sensitive attributes (s)
    X = df.drop('income', axis=1)
    y = df['income']

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    onehot_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(onehot_feature_names) + sensitive_features
    
    # Convert X_processed back to DataFrame to easily extract sensitive attributes
    # This part might need adjustment based on how 'remainder' handles column order
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names[:X_processed.shape[1]]) # Slice to match actual columns

    # Extract and one-hot encode sensitive attributes for 's_processed'
    sensitive_preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), sensitive_features)
        ],
        remainder='drop' # Only keep sensitive features
    )
    s_processed = sensitive_preprocessor.fit_transform(df[sensitive_features])
    # Convert s_processed to DataFrame to align indices if necessary for train_test_split
    s_processed_df = pd.DataFrame(s_processed.toarray(), 
                                  columns=sensitive_preprocessor.named_transformers_['onehot']
                                                            .get_feature_names_out(sensitive_features))

    # Ensure y and s_processed_df have the same index as X_processed_df for splitting
    y = y.reset_index(drop=True)
    s_processed_df = s_processed_df.reset_index(drop=True)
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_processed_df, y, s_processed_df, test_size=test_size, random_state=random_state
    )

    print("Adult Census data loaded and preprocessed.")
    return X_train, X_test, y_train, y_test, s_train, s_test

def load_compas_data(test_size=0.2, random_state=42):
    """
    Loads and preprocesses the COMPAS dataset.

    This function loads the 'compas-scores-two-years.csv' dataset, applies standard
    filtering criteria used in fairness research, identifies sensitive attributes,
    and preprocesses features (numerical scaling, categorical one-hot encoding).
    It then splits the data into training and testing sets, returning features (X),
    target (y), and sensitive attributes (s) as pandas DataFrames.

    Args:
        test_size (float, optional): The proportion of the dataset to include in the test split.
                                     Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training target labels.
            - y_test (pd.Series): Testing target labels.
            - s_train (pd.DataFrame): Training sensitive attributes (one-hot encoded).
            - s_test (pd.DataFrame): Testing sensitive attributes (one-hot encoded).
        Returns (None, None, None, None, None) if data file is not found.
    """
    print("Loading and preprocessing COMPAS data...")

    try:
        df = pd.read_csv('data/raw/compas-scores-two-years.csv', skipinitialspace=True)
    except FileNotFoundError:
        print("Error: COMPAS file (compas-scores-two-years.csv) not found in data/raw/")
        print("Please download it and place it there.")
        return None, None, None, None, None, None

    # Apply filters based on common practice in fairness literature
    df = df[(df['days_since_input'] <= 365 * 2) & 
            (df['c_charge_degree'] != 'O') & 
            (df['decile_score'] != -1) & 
            (df['score_text'] != 'N/A')]

    # Define target variable (recidivism: 0 = no recid, 1 = recid)
    df['two_year_recid'] = df['two_year_recid'].astype(int)
    y = df['two_year_recid']

    # Define sensitive attributes
    # For COMPAS, common sensitive attributes include race, sex.
    # age_cat is also often treated as sensitive.
    sensitive_features = ['race', 'sex', 'age_cat']

    # Select features (excluding target, sensitive, and other irrelevant columns)
    features_to_drop = [
        'id', 'name', 'first', 'last', 'compas_screening_date', 
        'dob', 'age_cat', 'c_charge_degree', 'c_charge_desc', 'c_offense_date',
        'r_offense_date', 'r_charge_desc', 'r_jail_date', 'r_fade_in', 'r_offense_num',
        'r_sec_art', 'r_fel_bod', 'vr_charge_degree', 'vr_offense_date',
        'vr_charge_desc', 'v_type_of_assessment', 'v_decile_score', 'v_score_text',
        'priors_count.1', 'decile_score', 'score_text', 'raw_score', 'violent_recid',
        'is_violent_recid', 'days_since_input', 'end', 'two_year_recid'
    ] + sensitive_features # Also drop sensitive features from X to prevent data leakage

    X = df.drop(columns=features_to_drop, errors='ignore')
    # Fill missing values for numerical features before preprocessing
    X = X.fillna(X.mean(numeric_only=True))

    # Define categorical and numerical features for preprocessing
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Create preprocessing pipelines for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Ensure sensitive features are one-hot encoded separately if needed for 's' output later
    # Here we are one-hot encoding them to be part of the X_processed_df for splitting, 
    # but for 's_processed' we will use the original non-encoded sensitive features.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    onehot_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(onehot_feature_names)

    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

    # Extract and one-hot encode sensitive attributes for 's_processed'
    # We need to one-hot encode sensitive features to convert them to numerical format
    # before converting to tensor for PyTorch Dataset.
    sensitive_preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), sensitive_features)
        ],
        remainder='drop' # Only keep sensitive features
    )
    s_processed = sensitive_preprocessor.fit_transform(df[sensitive_features])
    # Convert s_processed to DataFrame to align indices if necessary for train_test_split
    s_processed_df = pd.DataFrame(s_processed.toarray(), 
                                  columns=sensitive_preprocessor.named_transformers_['onehot']
                                                            .get_feature_names_out(sensitive_features))

    # Ensure y and s_processed_df have the same index as X_processed_df for splitting
    y = y.reset_index(drop=True)
    s_processed_df = s_processed_df.reset_index(drop=True)
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_processed_df, y, s_processed_df, test_size=test_size, random_state=random_state
    )

    print("COMPAS data loaded and preprocessed.")
    return X_train, X_test, y_train, y_test, s_train, s_test

def load_fairface_data(image_size=(64, 64), num_channels=3, num_classes=7, num_sensitive_groups=7, num_samples=1000, test_size=0.2, random_state=42):
    """
    (Simulated) Loads and preprocesses the FairFace dataset by generating dummy data.

    This function simulates the loading and preprocessing of an image dataset like FairFace.
    It generates random image tensors, corresponding labels, and sensitive attributes.
    In a real-world scenario, this would involve reading image files and their annotations.

    Args:
        image_size (tuple, optional): The (height, width) of the simulated images. Defaults to (64, 64).
        num_channels (int, optional): The number of color channels (e.g., 3 for RGB). Defaults to 3.
        num_classes (int, optional): The number of output classes for the task. Defaults to 7 (for FairFace races/ethnicities).
        num_sensitive_groups (int, optional): The number of sensitive attribute categories. Defaults to 7.
        num_samples (int, optional): The total number of dummy samples to generate. Defaults to 1000.
        test_size (float, optional): The proportion of the dataset to include in the test split.
                                     Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - X_train (torch.Tensor): Training image features (N, C, H, W).
            - X_test (torch.Tensor): Testing image features (N, C, H, W).
            - y_train (torch.Tensor): Training target labels.
            - y_test (torch.Tensor): Testing target labels.
            - s_train (torch.Tensor): Training sensitive attributes (one-hot encoded).
            - s_test (torch.Tensor): Testing sensitive attributes (one-hot encoded).
    """
    print(f"Generating dummy FairFace data with image_size={image_size}, num_samples={num_samples}...")
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Simulate image data: (N, C, H, W)
    X = torch.randn(num_samples, num_channels, image_size[0], image_size[1])
    # Simulate labels
    y = torch.randint(0, num_classes, (num_samples,))
    # Simulate sensitive attributes (one-hot encoded)
    s = torch.nn.functional.one_hot(torch.randint(0, num_sensitive_groups, (num_samples,)), num_classes=num_sensitive_groups).float()

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=test_size, random_state=random_state
    )

    print("Dummy FairFace data generated and split.")
    return X_train, X_test, y_train, y_test, s_train, s_test

class AdultCensusDataset(Dataset):
    """
    Custom PyTorch Dataset for the Adult Census data.

    This dataset wraps the preprocessed Adult Census features (X),
    target labels (y), and sensitive attributes (s) into a format
    compatible with PyTorch DataLoaders.
    """
    def __init__(self, X, y, s):
        """
        Initializes the AdultCensusDataset.

        Args:
            X (pd.DataFrame): Features DataFrame.
            y (pd.Series): Target labels Series.
            s (pd.DataFrame): Sensitive attributes DataFrame (one-hot encoded).
        """
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.s = torch.tensor(s.values, dtype=torch.float32) # Sensitive attributes are now one-hot encoded float

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features, target label, and sensitive attributes.
        """
        return self.X[idx], self.y[idx], self.s[idx]

class CompasDataset(Dataset):
    """
    Custom PyTorch Dataset for the COMPAS data.

    This dataset wraps the preprocessed COMPAS features (X),
    target labels (y), and sensitive attributes (s) into a format
    compatible with PyTorch DataLoaders.
    """
    def __init__(self, X, y, s):
        """
        Initializes the CompasDataset.

        Args:
            X (pd.DataFrame): Features DataFrame.
            y (pd.Series): Target labels Series.
            s (pd.DataFrame): Sensitive attributes DataFrame (one-hot encoded).
        """
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.s = torch.tensor(s.values, dtype=torch.float32) # Sensitive attributes, potentially multi-hot

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features, target label, and sensitive attributes.
        """
        return self.X[idx], self.y[idx], self.s[idx]

class FairFaceDataset(Dataset):
    """
    Custom PyTorch Dataset for the FairFace data.

    This dataset wraps FairFace image features (X), target labels (y),
    and sensitive attributes (s) into a format compatible with PyTorch DataLoaders.
    It applies specified image transformations.
    """
    def __init__(self, X, y, s, transform=None):
        """
        Initializes the FairFaceDataset.

        Args:
            X (torch.Tensor): Image features tensor (N, C, H, W).
            y (torch.Tensor): Target labels tensor.
            s (torch.Tensor): Sensitive attributes tensor (one-hot encoded).
            transform (torchvision.transforms.Compose, optional): Image transformations to apply.
                                                                Defaults to None.
        """
        self.X = X
        self.y = y
        self.s = s
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image, target label, and sensitive attributes.
        """
        image = self.X[idx]
        label = self.y[idx]
        sensitive_attr = self.s[idx]

        if self.transform:
            # For torchvision transforms, input should be PIL Image or torch.Tensor
            # Since X is already a tensor, we can apply transforms directly if they expect tensor input
            # or convert to PIL Image if needed by a specific transform.
            # For simplicity, assuming transforms can handle tensor directly or will be handled by ToTensor() if not already there.
            image = self.transform(image)

        return image, label, sensitive_attr

def get_adult_census_dataloader(batch_size=32, test_size=0.2, random_state=42):
    """
    Loads, preprocesses, and creates PyTorch DataLoaders for the Adult Census dataset.

    Args:
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        test_size (float, optional): The proportion of the dataset to include in the test split.
                                     Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the testing set.
            - input_dim (int): Dimensionality of the input features.
            - num_sensitive_attrs (int): Number of sensitive attribute columns (one-hot encoded).
        Returns (None, None, None, None) if data loading fails.
    """
    X_train, X_test, y_train, y_test, s_train, s_test = load_adult_census_data(test_size=test_size, random_state=random_state)

    if X_train is None:
        return None, None, None, None

    train_dataset = AdultCensusDataset(X_train, y_train, s_train)
    test_dataset = AdultCensusDataset(X_test, y_test, s_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine input dimension for the model
    input_dim = X_train.shape[1]
    # Determine number of sensitive attributes for evaluation
    num_sensitive_attrs = s_train.shape[1] if len(s_train.shape) > 1 else 1

    print(f"Adult Census DataLoaders created. Input Dim: {input_dim}, Num Sensitive Attrs: {num_sensitive_attrs}")

    return train_loader, test_loader, input_dim, num_sensitive_attrs

def get_compas_dataloader(batch_size=32, test_size=0.2, random_state=42):
    """
    Loads, preprocesses, and creates PyTorch DataLoaders for the COMPAS dataset.

    Args:
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        test_size (float, optional): The proportion of the dataset to include in the test split.
                                     Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the testing set.
            - input_dim (int): Dimensionality of the input features.
            - num_sensitive_attrs (int): Number of sensitive attribute columns (one-hot encoded).
        Returns (None, None, None, None) if data loading fails.
    """
    X_train, X_test, y_train, y_test, s_train, s_test = load_compas_data(test_size=test_size, random_state=random_state)

    if X_train is None:
        return None, None, None, None

    train_dataset = CompasDataset(X_train, y_train, s_train)
    test_dataset = CompasDataset(X_test, y_test, s_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    num_sensitive_attrs = s_train.shape[1] if len(s_train.shape) > 1 else 1

    print(f"COMPAS DataLoaders created. Input Dim: {input_dim}, Num Sensitive Attrs: {num_sensitive_attrs}")

    return train_loader, test_loader, input_dim, num_sensitive_attrs

def get_fairface_dataloader(batch_size=32, image_size=(64, 64), num_channels=3, num_classes=7, num_sensitive_groups=7, num_samples=1000, test_size=0.2, random_state=42):
    """
    (Simulated) Loads, preprocesses, and creates PyTorch DataLoaders for the FairFace dataset.

    This function simulates the data loading and preprocessing pipeline for an image dataset.
    It generates dummy image data and creates DataLoaders with specified transformations.

    Args:
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        image_size (tuple, optional): The (height, width) of the simulated images. Defaults to (64, 64).
        num_channels (int, optional): The number of color channels (e.g., 3 for RGB). Defaults to 3.
        num_classes (int, optional): The number of output classes for the task. Defaults to 7.
        num_sensitive_groups (int, optional): The number of sensitive attribute categories. Defaults to 7.
        num_samples (int, optional): The total number of dummy samples to generate. Defaults to 1000.
        test_size (float, optional): The proportion of the dataset to include in the test split.
                                     Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the testing set.
            - input_shape (tuple): Shape of the input image data (channels, height, width).
            - num_sensitive_attrs (int): Number of sensitive attribute columns (one-hot encoded).
    """
    X_train, X_test, y_train, y_test, s_train, s_test = load_fairface_data(
        image_size=image_size,
        num_channels=num_channels,
        num_classes=num_classes,
        num_sensitive_groups=num_sensitive_groups,
        num_samples=num_samples,
        test_size=test_size,
        random_state=random_state
    )

    # Get image transforms
    image_transforms = get_image_transforms(image_size=image_size)

    train_dataset = FairFaceDataset(X_train, y_train, s_train, transform=image_transforms)
    test_dataset = FairFaceDataset(X_test, y_test, s_test, transform=image_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_shape = (num_channels, image_size[0], image_size[1])
    num_sensitive_attrs = s_train.shape[1] if len(s_train.shape) > 1 else 1

    print(f"FairFace DataLoaders created. Input Shape: {input_shape}, Num Sensitive Attrs: {num_sensitive_attrs}")

    return train_loader, test_loader, input_shape, num_sensitive_attrs

# Add more data loading functions as needed
