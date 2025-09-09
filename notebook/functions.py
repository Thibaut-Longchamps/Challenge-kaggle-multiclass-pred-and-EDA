import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway, levene

def perform_anova(df, metric_column, category_column):
    # Déterminer le nombre minimum d'observations parmi tous les niveaux de la variable catégorielle
    min_obs = df[category_column].value_counts().min()

    # Créer des échantillons équilibrés avec le même nombre d'observations
    samples = [df[df[category_column] == stage][metric_column].sample(n=min_obs, replace=True) for stage in df[category_column].unique()]

    # Effectuer l'ANOVA
    result = stats.f_oneway(*samples)

    # Afficher les résultats
    print(f"\nPour la colonne {metric_column}:")
    print("Statistique de test F :", result.statistic)
    print("P-valeur :", result.pvalue)

    # Interprétation de la p-valeur
    alpha = 0.05  # Niveau de signification
    if result.pvalue < alpha:
        print("La p-valeur est inférieure à", alpha, "=> On rejette l'hypothèse nulle.")
        
        # Exécuter le test de Tukey pour les comparaisons post hoc
        tukey_results = pairwise_tukeyhsd(df[metric_column], df[category_column])
        
        # Afficher les résultats du test de Tukey
        print(tukey_results.summary())
    else:
        print("La p-valeur est supérieure à", alpha, "=> On ne peut pas rejeter l'hypothèse nulle.")
        
# -----------------------------------------------------------------------------         
    
def drop_column(df, column_name, df_name="DataFrame"):
    if column_name in df.columns:
        df.drop(columns=[column_name], inplace=True)
        print(f"The '{column_name}' column has been removed from {df_name}.")
    else:
        print(f"The column '{column_name}' doesn't exist in {df_name}.")
        
# -----------------------------------------------------------------------------        
        
import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_heatmap(df, df_name = 'DataFrame'):
    # Créer une figure avec un sous-graphique
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Heatmap
    sns.heatmap(df.isna(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Values')

    # Vérification des valeurs manquantes
    if df.isna().any().any():
        plt.show()
    else:
        plt.close()  # Fermer la figure car la heatmap est vide
        print(f"No missing values in {df_name}.")


# -----------------------------------------------------------------------------      

# Manage outliers with Winsorize method

def winsorize_outliers_dataframe(df, lower_percentile=0.05, upper_percentile=0.95):
    """
    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    lower_percentile (float): The lower percentile from which winsorizing is applied.
    upper_percentile (float): The upper percentile from which winsorizing is applied.

    Returns:
    pd.DataFrame: The DataFrame with the winsorized outliers.
    """
    for column in df.columns:
        # Check if the columns is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Calculate lower and upper limits based on percentiles
            lower_limit = df[column].quantile(lower_percentile)
            upper_limit = df[column].quantile(upper_percentile)
            # Replace values outside the specified percentile range with limits
            df[column] = np.where(df[column] < lower_limit, lower_limit, np.where(df[column] > upper_limit, upper_limit, df[column]))
    # return the modified Dataframe
    return df

# --------------------------------------------------------------------------------

# Manage outliers with standard deviation method

def remove_outliers_std_dataframe(df, threshold=3):
    """
    Removes outliers using the standard deviation for all columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    threshold (float): The number of standard deviations above which to remove outliers.

    Returns:
    pd.DataFrame: The DataFrame without the outliers.
    """
    # Apply outlier suppression to each column
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        mean_val = df[column].mean()
        std_val = df[column].std()

        # Calculate limits
        lower_limit = mean_val - threshold * std_val
        upper_limit = mean_val + threshold * std_val

        # Delete outliers / underliers
        df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

    return df

# --------------------------------------------------------------------------------------

# Manage outliers with IQR method

def remove_outliers_iqr_inplace(df, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed (modified in place).
    multiplier (float): The multiplier factor to determine the IQR boundaries.
    """
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Calculate quantiles for each numeric column
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate IQR boundaries for each numeric column
    lower_limit = Q1 - multiplier * IQR
    upper_limit = Q3 + multiplier * IQR

    # Remove rows containing outliers
    df.drop(df[(df[numeric_columns] < lower_limit) | (df[numeric_columns] > upper_limit)].index, inplace=True) # Replace Value instead of cretae a new object 
    
    # ---------------------------------------------------------------------------------------------------------------------
    
    def Separate_target_features(df):

        # Separate target variable Y from features X
        print("Separating labels from features...")
        target_variable = "Status"

        X = df.drop(target_variable, axis = 1)
        Y = df.loc[:,target_variable]

        print("...Done.")
        print()

        print('Y : ')
        print(Y.head())
        print()
        print('X :')
        print(X.head())
        
        return X, Y
    
    # ------------------------------------------------------------------------------------------------------------------------
    
    def separate_cat_num_variable(X):
        """
        Separates features of a DataFrame into numeric and categorical types.

        Parameters:
        X (pd.DataFrame): The DataFrame containing features.

        Returns:
        tuple: A tuple containing lists of numeric and categorical feature names.
        """
        numeric_features = []
        categorical_features = []

        for i, t in X.dtypes.items():
            if ('float' in str(t)) or ('int' in str(t)):
                numeric_features.append(i)
            else:
                categorical_features.append(i)

        print('Found numeric features:', numeric_features)
        print('Found categorical features:', categorical_features)

        return numeric_features, categorical_features
    
    # -------------------------------------------------------------------------------------------------------------------------
    
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    def Create_pipeline_preprocessor(numeric_features, categorical_features):
        
        """
        Create a papeline in order to apply features preprocessing
        
        Parameters : 
        numeric features : list of numeric columns
        categorical_features : list of categorical features
        
        Returns :
        Display preprocessor with tranformations

        """
    
        # Create pipeline for numeric features
        numeric_transformer = SimpleImputer(strategy='mean') # missing values will be replaced by columns mean (here we don't have missing values)
        categorical_transformer = OneHotEncoder(drop='first') # encode with 0 and 1 categorical variable, the argument drop = first the first level of each categorical characteristic is excluded to avoid collinearity

        # Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features), # no need numeric_transformer no missing/NaN values
                ('cat', categorical_transformer, categorical_features)
            ])
        
        print(preprocessor)
        return preprocessor

# ---------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

def apply_preprocessing(preprocessor, X_train, Y_train, X_test, Y_test):
    
    '''
        Apply preprocessing steps on training and test sets.

    Parameters:
    preprocessor (object): The preprocessor object (e.g., ColumnTransformer) for feature transformations.
    X_train (DataFrame or ndarray): Training features.
    Y_train (Series or ndarray): Training labels.
    X_test (DataFrame or ndarray): Test features.
    Y_test (Series or ndarray): Test labels.

    Returns:
    X_train (ndarray): Processed training features.
    Y_train (ndarray): Encoded training labels.
    X_test (ndarray): Processed test features.
    Y_test (ndarray): Encoded test labels.
    
    '''
       
    # Preprocessings on train set
    print("Performing preprocessings on train set...")
    print(X_train.head())
    X_train = preprocessor.fit_transform(X_train)
    print('...Done.')
    print(X_train[0:5]) # MUST use this syntax because X_train is a numpy array and not a pandas DataFrame anymore
    print()

    # Label encoding
    print("Encoding labels...")
    print(Y_train.head())
    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    print("...Done")
    print(Y_train[0:5])

    # Preprocessings on test set
    print("Performing preprocessings on test set...")
    print(X_test.head()) 
    X_test = preprocessor.transform(X_test) # Don't fit again !! The test set is used for validating decisions
    # we made based on the training set, therefore we can only apply transformations that were parametered using the training set.
    # Otherwise this creates what is called a leak from the test set which will introduce a bias in all your results.
    print('...Done.')
    print(X_test[0:5,:]) # MUST use this syntax because X_test is a numpy array and not a pandas DataFrame anymore
    print()

    # Label encoding
    print("Encoding labels...")
    print(Y_test[0:5])
    Y_test = encoder.transform(Y_test)
    print("...Done")
    print(Y_test[0:5])
    
    return X_train, Y_train, X_test, Y_test

# ------------------------------------------------------------------------------------------------------------------------------

def check_nan_values(data):
    '''
    Check if there are any NaN values in the given dataset.

    Parameters:
    data (DataFrame or ndarray): The dataset to check for NaN values.

    Returns:
    None
    '''

    # Check if there are any NaN values in the dataset
    has_nan_values = np.any(np.isnan(data))

    if has_nan_values:
        print("The dataset contains NaN values.")
    else:
        print("No NaN values in the dataset.")
        
#  --------------------------------------------------------------------------------------------------------------------------------------

from sklearn.utils import resample

def oversampling(X_train, Y_train):
    
    '''
    Perform oversampling to balance the classes in the dataset.

    Parameters:
    X_train (ndarray): Features of the training set.
    Y_train (ndarray): Labels of the training set.

    Returns:
    X_train (ndarray): Resampled features after oversampling.
    Y_train (ndarray): Resampled labels after oversampling.
    '''
    
    # Display the distribution of classes before oversampling
    unique, counts = np.unique(Y_train, return_counts=True)
    print("Avant l'oversampling :", dict(zip(unique, counts)))
    
    # Concatenate features and labels
    data_x_y_train = np.column_stack((X_train, Y_train))
    
    # Separate majority and minority classes
    class_majority_C = data_x_y_train[data_x_y_train[:, -1] == 0]
    class_minority_D = data_x_y_train[data_x_y_train[:, -1] == 2]
    class_minority_CL = data_x_y_train[data_x_y_train[:, -1] == 1]
    
    # Determine the size of the majority class
    size_majority = len(class_majority_C)
    
    # Perform oversampling for each minority class
    class_minority_D_resampled = resample(class_minority_D, replace=True, n_samples=size_majority, random_state=0)
    class_minority_CL_resampled = resample(class_minority_CL, replace=True, n_samples=size_majority, random_state=0)
    
    # Concatenate the resampled minority classes with the majority class again
    data_resampled = np.vstack((class_minority_D_resampled, class_majority_C, class_minority_CL_resampled))
    
    # Separate features and labels after oversampling
    X_train, Y_train = data_resampled[:, :-1], data_resampled[:, -1]

    # Display the distribution of classes after oversampling
    unique, counts = np.unique(Y_train, return_counts=True)
    print("Après l'oversampling :", dict(zip(unique, counts)))
    
    return X_train, Y_train

# -----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, Y, cv, scoring, data):
    '''
    Plot the learning curve for a given model.

    Parameters:
    model: The machine learning model for which the learning curve will be plotted.
    X: The features of the dataset.
    Y: The labels of the dataset.
    cv: Number of cross-validation folds.
    scoring: The scoring metric for evaluation.
    data: A label or identifier for the dataset.

    Returns:
    None
    '''

    # Generate learning curves
    N, train_score, val_score = learning_curve(model, X, Y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1, 10), n_jobs=-1)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot the training and validation scores
    plt.plot(N, train_score.mean(axis=1), label=f'Train Score: {data}')
    plt.plot(N, val_score.mean(axis=1), label=f'Validation Score: {data}')

    # Add labels and title
    plt.legend()
    plt.xlabel('Training Size')
    plt.ylabel(scoring)
    plt.title('Learning Curve')

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    # ------------------------------------------------------------------------------------------------------------
    from sklearn.metrics import confusion_matrix, classification_report
    
    def evaluation(model):
        '''
        Evaluate a machine learning model by fitting it to the training data and making predictions on the training and test sets.

        Parameters:
        model: The machine learning model to be evaluated.

        Returns:
        None
        '''

        # Fit the model on the training set
        model.fit(X_train, Y_train)

        # Predictions on the training set
        print("Predictions on the training set...")
        Y_train_pred = model.predict(X_train)
        print("...Done.")

        # Predictions on the test set
        print("Predictions on the test set...")
        Y_test_pred = model.predict(X_test)
        print("...Done.")

        # Display confusion matrix and classification report for the training set
        print("Confusion Matrix and Classification Report for the Training Set:")
        print(confusion_matrix(Y_train, Y_train_pred))
        print(classification_report(Y_train, Y_train_pred))
        print()

        # Display confusion matrix and classification report for the test set
        print("Confusion Matrix and Classification Report for the Test Set:")
        print(confusion_matrix(Y_test, Y_test_pred))
        print(classification_report(Y_test, Y_test_pred))
        
        # ------------------------------------------------------------------------------------------------------------------