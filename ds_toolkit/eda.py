
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


def statistics(data):
    """Calculate comprehensive descriptive statistics for numerical data.
    
    Computes various statistical metrics including central tendency, dispersion, and distribution shape for all numeric columns in the input DataFrame.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing numerical columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with the following statistics for each numeric column:
        - non-null : int
            Count of non-null values
        - range : float
            Difference between maximum and minimum values
        - min : float
            Minimum value
        - quant25 : float
            First quartile (25th percentile)
        - median : float
            Median (50th percentile)
        - quant75 : float
            Third quartile (75th percentile)
        - max : float
            Maximum value
        - mean : float
            Arithmetic mean
        - std : float
            Standard deviation
        - skew : float
            Skewness (measure of distribution asymmetry)
        - kurtosis : float
            Kurtosis (measure of distribution tailedness)
            
    Notes
    -----
    - All metrics are rounded to 1 decimal place
    - Only columns with int or float datatypes are analyzed
    - NaN values are automatically excluded from calculations
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2.5, 3.5, 4.5, 5.5, 6.5],
    ...     'C': ['a', 'b', 'c', 'd', 'e']
    ... })
    >>> stats = statistics(df)
    >>> print(stats.columns)
    Index(['non-null', 'range', 'min', 'quant25', 'median', 'quant75', 'max', 'mean', 'std', 'skew', 'kurtosis'], dtype='object')
    """
    # Select only numeric columns
    num_data = data.select_dtypes(include=['int', 'float'])

    # Central Tendency
    mean = num_data.apply(np.mean)

    # Quantiles
    q25 = num_data.quantile(0.25)
    q50 = num_data.quantile(0.5)
    q75 = num_data.quantile(0.75)
    range_ = num_data.apply(lambda x: x.max() - x.min())
    count = num_data.count()

    # Dispersion
    min_ = num_data.apply(min)
    max_ = num_data.apply(max)
    std = num_data.apply(np.std)

    # Distribution Shape
    skew = num_data.apply(lambda x: x.skew())
    kurtosis = num_data.apply(lambda x: x.kurtosis())

    metrics = pd.DataFrame({
        'non-null': count,
        'range': range_,
        'min': min_,
        'quant25': q25,
        'median': q50,
        'quant75': q75,
        'max': max_,
        'mean': mean,
        'std': std,
        'skew': skew,
        'kurtosis': kurtosis
    })
    
    return np.round(metrics, 1)


def inspect_outliers(df, numeric_columns):
    """Visualize and analyze outliers in numeric features using box plots and IQR method.
    
    Creates a 4-column grid of box plots for numeric features, with outlier statistics displayed below each plot. Uses the Interquartile Range (IQR) method for outlier detection. 
    The visualization includes the original data scale and prevents scientific notation for better readability.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the numeric features to analyze numeric_columns : list
        List of column names containing numeric data to analyze for outliers
        
    Returns
    -------
    None
        Displays a matplotlib figure with:
        - Grid of box plots (4 columns)
        - Outlier statistics below each plot
        - Original data scale (no scientific notation)
        
    Notes
    -----
    Visualization details:
    - Box plots arranged in a 4-column grid
    - White background text box for statistics
    - Increased vertical spacing between plots (h_pad=3)
    - Empty subplots are removed if number of features isn't divisible by 4
    
    For each feature, displays:
    - Box plot showing distribution and outliers
    - Number and percentage of outliers
    - Lower and upper bounds for outlier detection
    
    The IQR method defines outliers as:
    - Lower outliers: < Q1 - 1.5 * IQR
    - Upper outliers: > Q3 + 1.5 * IQR
    where:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    
    Examples
    --------
    >>> numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    >>> inspect_outliers(df, numeric_cols)
    """
    
    # Create subplots for each numeric column
    n_cols = 4
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten()
    
    # Plot data
    for idx, column in enumerate(numeric_columns):

        # Calculate outlier statistics using IQR method
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        
        # Create box plot
        sns.boxplot(data=df, y=column, ax=axes[idx])
        axes[idx].set_title(f'Box Plot - {column}')
        
        # Format y-axis to prevent scientific notation
        axes[idx].yaxis.set_major_formatter(ScalarFormatter())
        axes[idx].ticklabel_format(style='plain', axis='y')
        
        # Add statistics text below plot
        stats_text = f'Outliers: {len(outliers)} ({(len(outliers)/len(df))*100:.1f}%)\nBounds: [{lower_bound:,.1f} | {upper_bound:,.1f}]'
        axes[idx].text(0.5, -0.12, stats_text, 
                      horizontalalignment='center',
                      transform=axes[idx].transAxes,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
     # Remove empty subplots
    for idx in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout(h_pad=3)
    plt.show()